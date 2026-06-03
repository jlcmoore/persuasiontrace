"""
src/api/api.py

Author: Jared Moore
Date: July, 2025

An API and the logic to run participant experiments, possibly with just one
participant and either an LLM or a simple program or between two participants.
Stores intermediate results in a sql database.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import markdown
import websockets
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.functions import count
from sqlmodel import Session, SQLModel, create_engine, select
from starlette.websockets import WebSocketState
from typing_extensions import Annotated

from experiment.condition import PARTICIPANT_PROPOSITION_PROMPT, Condition, Roles
from experiment.endpoints import (
    PARTICIPANT_PROPOSITION_ERROR_REASON,
    moderate_content,
    rephrase_participant_decision,
)
from experiment.llm_utils import call_llm
from experiment.round import LLM_HUMAN_LIKE_PROMPT_TEMPLATE, Round
from experiment.utils import (
    normalize_message_highlight,
    normalize_serial_sentence_values,
    token_time_totals_verbose,
)
from simulation.target import (
    BayesianNetwork,
    SimulatedTarget,
    TargetPersona,
    node_beliefs_from_trace_payload,
    susceptibilities_for_persona,
)

from . import message_processing
from .current_round_helpers import assign_participants, choose_participant_conditions
from .message_processing import ConnectionManager
from .sql_model import (
    CONNECT_ARGS,
    SQLITE_URL,
    ExternalUser,
    FlaggedResponse,
    Participant,
    Proposition,
    RoundORM,
    SentMessage,
    SentMessageBase,
    ensure_schema_compatibility,
)
from .sql_queries import (
    choose_condition,
    choose_proposition_for_participants,
    get_last_sent_message,
    get_paired_participant,
    get_participant_rounds,
    get_round_types_remaining,
    get_user_messages,
    populate_tables,
)
from .utils import (
    MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER,
    ServerSettings,
    round_over_state,
)

WAITING_TOO_LONG_MSG = "Participant {id} has waited too long."

##### App startup

logger = logging.getLogger(__name__)

# This is for websockets
manager = ConnectionManager()


# Sentence segmentation provided by Round.split_into_sentences for consistency


@lru_cache
def get_settings():
    """Loads in the settings from disc and caches them"""
    return ServerSettings()


global_engine = create_engine(SQLITE_URL, echo=False, connect_args=CONNECT_ARGS)


def get_session():
    """A FastAPI dependency. Yields the global session on call."""
    with Session(global_engine) as session:
        yield session


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Defines the start up and clean up necessary when running the server through FastAPI.
    NB: Everything before `yield` occurs before the server starts up and after `yield`
    occurs on shutdown.
    """
    settings = get_settings()

    # Ensure existing DBs have any newly added columns before usage
    ensure_schema_compatibility(global_engine)
    SQLModel.metadata.create_all(global_engine)
    with Session(global_engine) as session:
        propositions = session.exec(select(Proposition)).all()
        # Only populate the tables if they are not already there
        if not propositions:
            populate_tables(
                session=session,
                conditions=settings.condition_num_rounds.keys(),
                propositions_filenames=settings.propositions_full_filenames,
            )

    yield
    # NB: After this yield we clean up any resources
    # If dev_environment, overwrite the database

    if settings.dev_environment:
        SQLModel.metadata.drop_all(global_engine)


if get_settings().dev_environment:
    app = FastAPI(lifespan=lifespan)
else:
    # Don't expose the docs in production
    app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)

    # 1) Mount a "static" folder
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")


###############
##### API
###############


## Shared error messasges


def get_round_error(session: Session, round_id: int) -> RoundORM:
    """A helper to get the round by round_id and raise errors"""
    rd_orm = session.get(RoundORM, round_id)
    if not rd_orm:
        message = "Round does not exist"
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)
    return rd_orm


## HTTP Request Classes


class FeedbackRequest(BaseModel):
    participant_id: int
    feedback: str


class ParticipantInitRequest(BaseModel):
    id: str
    turnstile_token: str | None = None


class ParticipantRoundRequest(BaseModel):
    participant_id: int

    # This request is only availble if the server is running on development.
    # They force the particpant to be assinged
    # to the following conditions, if possible.
    is_target: bool | None = None
    llm_persuader: str | None = None
    llm_target: str | None = None

    proposition: str | None = None
    persuader_supports_proposition: bool | None = None
    continuous_measure: str | None = None

    def force_round_settings(self) -> bool:
        for name, _ in type(self).model_fields.items():
            if name == "participant_id":
                continue
            if getattr(self, name) is not None:
                return True
        return False


class ParticipantRequest(BaseModel):
    id: int


class ParticipantPropositionRequest(BaseModel):
    """Request payload for participant proposition setup."""

    participant_id: int
    decision: str | None = None


def validate_turnstile_token(
    token: str,
    secret: str,
    remote_ip: str | None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """
    Validate a Turnstile token with Cloudflare's siteverify API.

    Args:
        token: The Turnstile response token from the client.
        secret: The Turnstile secret key.
        remote_ip: The visitor's IP address, if available.
        timeout_seconds: The timeout in seconds for the validation request.

    Returns:
        A dict containing the JSON response from the siteverify API.
    """
    payload = {"secret": secret, "response": token}
    if remote_ip:
        payload["remoteip"] = remote_ip
    data = urlencode(payload).encode("utf-8")
    request = UrlRequest(
        "https://challenges.cloudflare.com/turnstile/v0/siteverify",
        data=data,
        method="POST",
    )
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        logger.error("Turnstile validation request failed: %s", exc)
        return {"success": False, "error-codes": ["internal-error"]}

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        logger.error("Turnstile validation returned invalid JSON: %s", exc)
        return {"success": False, "error-codes": ["internal-error"]}


def get_participant(session: Session, participant_id: int) -> Participant:
    """Tries to get the participant. Throws an error if they don't exist"""
    participant = session.get(Participant, participant_id)

    if participant is None:
        message = f"Participant {participant_id} does not exist."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    return participant


def _participant_proposition_counts(
    session: Session, participant_id: int, required_count: int
) -> tuple[int, int, int]:
    """
    Return (total_required, completed_count, remaining_count).
    """
    count_stmt = select(count(Proposition.id)).where(
        Proposition.participant_id == participant_id
    )
    completed = int(session.exec(count_stmt).one())
    remaining = max(required_count - completed, 0)
    return required_count, completed, remaining


def _participant_proposition_status(
    session: Session,
    participant: Participant,
    settings: ServerSettings,
) -> dict[str, Any]:
    """
    Return status payload for participant-proposition setup.
    """
    if not settings_require_participant_propositions(settings):
        return {"enabled": False}
    required, completed, remaining = _participant_proposition_counts(
        session, participant.id, settings.rounds_per_participant
    )
    return {
        "enabled": True,
        "required_count": required,
        "completed_count": completed,
        "remaining_count": remaining,
        "prompt": markdown.markdown(
            PARTICIPANT_PROPOSITION_PROMPT,
            extensions=["md_in_html"],
        ),
    }


def settings_require_participant_propositions(settings: ServerSettings) -> bool:
    """Return True if any configured condition uses participant propositions."""
    return any(
        condition.participant_proposition
        for condition in settings.condition_num_rounds.keys()
    )


def participant_round_validate(
    request: ParticipantRoundRequest,
    settings: ServerSettings,
):
    """Raises various validation errors for ParticipantRoundRequests"""
    if not settings.dev_environment and request.force_round_settings():
        raise HTTPException(
            status_code=400,
            detail="You may only change the kind of round to play in development.",
        )
    if request.force_round_settings():
        if request.is_target is None:
            raise HTTPException(
                status_code=400,
                detail=("You must pass is_target"),
            )
        if request.is_target and not request.llm_persuader:
            raise HTTPException(
                status_code=400,
                detail="You must give an llm persuader when the participant is the target",
            )
        if not request.is_target and not request.llm_target:
            raise HTTPException(
                status_code=400,
                detail="You must give an llm target when the participant is the persuader",
            )


## API proper


@app.post("/participant_instructions/")
def participant_instructions(
    request: ParticipantRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Returns the condition-specific instructions for the participant.
    """
    logger.info(request)
    participant = get_participant(session, request.id)

    assert participant
    assert participant.condition

    condition = Condition(**participant.condition)

    return markdown.markdown(
        condition.instructions(
            is_target=participant.is_target(),
            is_human=True,
            round_time_limit=settings.round_time_limit,
            max_audio_seconds=settings.max_audio_seconds,
        ),
        extensions=["md_in_html"],
    )


@app.post("/participant_propositions/")
def participant_propositions(
    request: ParticipantPropositionRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
) -> dict[str, Any]:
    """
    Create or query participant-proposition setup state.
    """
    participant = get_participant(session, request.participant_id)
    status = _participant_proposition_status(session, participant, settings)
    if not status.get("enabled") or request.decision is None:
        return status

    if status["remaining_count"] <= 0:
        status["status"] = "ok"
        return status

    decision = request.decision.strip()
    if not decision:
        status["status"] = "error"
        status["reason"] = PARTICIPANT_PROPOSITION_ERROR_REASON
        return status

    if moderate_content(decision):
        status["status"] = "error"
        status["reason"] = PARTICIPANT_PROPOSITION_ERROR_REASON
        return status

    try:
        result = rephrase_participant_decision(decision)
    except ValueError:
        logger.exception("Participant proposition model returned invalid output")
        status["status"] = "error"
        status["reason"] = PARTICIPANT_PROPOSITION_ERROR_REASON
        return status

    if result.get("status") != "ok":
        status["status"] = "error"
        error_code = result.get("code")
        error_message = result.get("message")
        if error_code and error_message:
            status["code"] = error_code
            status["reason"] = error_message
            return status
        reason = result.get("reason")
        if reason:
            status["reason"] = (
                f"{PARTICIPANT_PROPOSITION_ERROR_REASON} Reason: {reason}."
            )
        else:
            status["reason"] = PARTICIPANT_PROPOSITION_ERROR_REASON
        return status

    proposition_text = result.get("proposition", "").strip()
    if not proposition_text:
        status["status"] = "error"
        status["reason"] = PARTICIPANT_PROPOSITION_ERROR_REASON
        return status

    if session.get(Proposition, proposition_text):
        status["status"] = "error"
        status["reason"] = PARTICIPANT_PROPOSITION_ERROR_REASON
        return status

    prop = Proposition(
        id=proposition_text,
        original_text=decision,
        factual_domain=False,
        proposition_is_correct=None,
        control_dialogue=False,
        participant_id=participant.id,
    )
    session.add(prop)
    session.commit()

    status = _participant_proposition_status(session, participant, settings)
    status["status"] = "ok"
    status["proposition"] = proposition_text
    return status


@app.post("/participant_init/")
def participant_init(
    request: ParticipantInitRequest,
    http_request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> None:
    """
    Check if this ID already exists. if not, add in the user.
    Regardless, returns the participant id (session id, effectively) of the given user.
    """
    logger.info(request)

    secret = os.getenv("TURNSTILE_SECRET_KEY", "").strip()
    if secret:
        if not request.turnstile_token:
            raise HTTPException(
                status_code=400, detail="Missing Turnstile verification token."
            )
        forwarded_ip = http_request.headers.get("CF-Connecting-IP")
        if not forwarded_ip:
            forwarded_ip = http_request.headers.get("X-Forwarded-For")
        if forwarded_ip and "," in forwarded_ip:
            forwarded_ip = forwarded_ip.split(",")[0].strip()
        remote_ip = forwarded_ip or (
            http_request.client.host if http_request.client else None
        )
        validation = validate_turnstile_token(
            request.turnstile_token, secret, remote_ip
        )
        if not validation.get("success"):
            error_codes = validation.get("error-codes") or []
            raise HTTPException(
                status_code=400,
                detail=f"Turnstile verification failed: {', '.join(error_codes)}",
            )

    user = session.exec(
        select(ExternalUser).where(ExternalUser.external_id == request.id)
    ).first()

    if user:
        # if they call init twice, signal a real conflict rather than silently succeed
        detail = f"Participant with external id={request.id} already initialized"
        logger.error(detail)
        raise HTTPException(status_code=409, detail=detail)

    # External User has already been added.
    user = ExternalUser(external_id=request.id)
    session.add(user)
    session.commit()
    session.refresh(user)

    participant = Participant(
        id=user.id,
    )
    session.add(participant)
    session.commit()
    session.refresh(participant)

    return {"participant_id": user.id}


@app.post("/participant_ready/")
def participant_ready(
    request: ParticipantRequest,
    session: Annotated[Session, Depends(get_session)],
) -> None:
    """
    Add the participant, `request.participant_id` to the waiting room,
    updating the time they entered on each call.
    NB: do not call repeatedly -- this timestamp is used
    to indicate if the participant is in a current round.
    Raises an exception if the participant does not exist.
    Raises an exception if the participant is already in a round.
    """
    logger.info(request)

    participant = get_participant(session, request.id)
    # reject if already in an unfinished round
    if participant.current_round:
        rd = session.get(RoundORM, participant.current_round)
        if rd and not rd.finished():
            detail = f"Participant {participant.id} is already in an active round"
            logger.error(detail)
            raise HTTPException(status_code=400, detail=detail)

    participant.entered_waiting_room = datetime.now(timezone.utc)
    participant.current_round = None
    session.add(participant)
    session.commit()
    session.refresh(participant)


@app.post("/current_round/")
def current_round(
    request: ParticipantRoundRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Returns the current round the participant is in as a dict with keys:
    - round_id (int): the current round id
    - is_target (bool): whether the participant is the target
    - prompt (str): the initial prompt to show the participant

    Returns an empty dict if the participant cannot yet be assigned.

    If the participant is not in a round attempts to assign them to one,
    possibly pairing them with another paticipant.

    Raises an exception if:

    - the participant does not exist
    - the participant has no more rounds to play
    """
    logger.info(request)

    try:
        # NB: We are locking on the participant here and will until the transaction is completed.
        participant = session.exec(
            select(Participant)
            .where(Participant.id == request.participant_id)
            .with_for_update(skip_locked=True)
        ).first()

        if participant is None:
            message = f"p={request.participant_id} does not exist."
            logger.error(message)
            raise HTTPException(status_code=400, detail=message)

        if settings_require_participant_propositions(settings):
            _, completed, remaining = _participant_proposition_counts(
                session, participant.id, settings.rounds_per_participant
            )
            if remaining > 0:
                message = (
                    "Participant propositions are required before joining the lobby."
                )
                logger.error(
                    "p=%s missing participant propositions (%s/%s)",
                    participant.id,
                    completed,
                    settings.rounds_per_participant,
                )
                raise HTTPException(status_code=400, detail=message)

        participant_round_validate(request, settings)
        # Is the participant already in a round?
        if participant.current_round:
            logger.debug("p=%s is already in a round.", participant.id)
            rd_orm = session.get(RoundORM, participant.current_round)

            if rd_orm.processing_msg:
                message = "Call back once the last message is processed."
                raise HTTPException(status_code=400, detail=message)
        else:
            # These vary just over each round
            chosen_participant = None
            if request.force_round_settings():
                target_id = request.participant_id if request.is_target else None

                persuader_id = request.participant_id if not request.is_target else None
                #
                proposition = session.get(Proposition, request.proposition)

                if not proposition:
                    raise HTTPException(
                        status_code=400,
                        detail=f"p={participant.id} Invalid proposition, {proposition}, passed",
                    )

                # No human-human rounds are allowed here.
                chosen_condition = Condition(
                    roles=Roles(
                        llm_persuader=request.llm_persuader,
                        llm_target=request.llm_target,
                        human_persuader=persuader_id is not None,
                        human_target=target_id is not None,
                    ),
                    persuader_supports_proposition=request.persuader_supports_proposition,
                    factual_domain=proposition.factual_domain,
                    proposition_is_correct=proposition.proposition_is_correct,
                    continuous_measure=request.continuous_measure,
                    on_reflection=False,
                    control_dialogue=False,  # TODO: should find a more substative fix
                )

                participant.condition = chosen_condition.model_dump()

                # Calling this to assign the Particpant to the condition values
                choose_participant_conditions(
                    participant,
                    None,
                    chosen_condition,
                )
            else:

                # TODO: could rounds instead be part of the Participant object using that
                # sneaky relationship attribute?
                rounds = get_participant_rounds(participant, session)
                if len(rounds) >= settings.rounds_per_participant:
                    message = f"p={participant.id} has no more rounds to play."
                    logger.error(message)
                    raise HTTPException(
                        status_code=400,
                        detail=message,
                    )

                # Tabulate which conditions we have yet to fill
                paired_rounds_remaining_set, non_paired_rounds_remaining_set = (
                    get_round_types_remaining(session, settings.condition_num_rounds)
                )
                if (
                    not paired_rounds_remaining_set
                    and not non_paired_rounds_remaining_set
                ):
                    # TODO: (low priority) consider failing silently here?
                    message = "There are no rounds to play in general"
                    logger.error(message)
                    raise HTTPException(status_code=400, detail=message)

                # Only look for another participant if we still need to fill that condition.
                if len(paired_rounds_remaining_set) > 0 and (
                    not participant.condition
                    or participant.condition
                    and Condition(**participant.condition).is_paired_human()
                ):
                    ## Get an existing participant, if it exists
                    chosen_participant = get_paired_participant(
                        participant,
                        session,
                        settings,
                    )
                    if participant.condition and chosen_participant.condition:
                        assert participant.condition == chosen_participant.condition

                if (
                    participant.waiting_time()
                    > MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER
                    * settings.waiting_room_timeout
                ):
                    # The participant has been waiting too long regardless of the condition.
                    # Send a signal that the experiment should end for them.
                    participant.waited_too_long = True
                    session.add(participant)
                    session.commit()

                    message = WAITING_TOO_LONG_MSG.format(id=participant.id)
                    logger.error(message)
                    raise HTTPException(
                        status_code=400,
                        detail=message,
                    )

                chosen_condition = choose_condition(
                    participant=participant,
                    chosen_participant=chosen_participant,
                    paired_rounds_remaining_set=paired_rounds_remaining_set,
                    non_paired_rounds_remaining_set=non_paired_rounds_remaining_set,
                    overassign_non_paired_conditions=settings.overassign_non_paired_conditions,
                    condition_num_rounds=settings.condition_num_rounds,
                    waiting_room_timeout=settings.waiting_room_timeout,
                )

                waiting_response = {
                    "waiting_time": participant.waiting_time().total_seconds()
                }
                if not chosen_condition:
                    logger.info(
                        "p=%s waiting_time=%s",
                        participant.id,
                        waiting_response["waiting_time"],
                    )
                    # Tell the user to wait longer
                    return waiting_response

                ## Choose condition assignments for these participants, if necessary
                choose_participant_conditions(
                    participant,
                    chosen_participant,
                    chosen_condition,
                )

                persuader_id, target_id = assign_participants(
                    participant, chosen_participant
                )

            proposition = choose_proposition_for_participants(
                participant, chosen_participant, session
            )

            proposition_control = None
            if chosen_condition.control_dialogue:
                proposition_control = choose_proposition_for_participants(
                    participant, chosen_participant, session, control_dialogue=True
                )

            is_target = participant.id == target_id

            # TODO: make sure the participants are in the same condition

            rd_orm = RoundORM(
                persuader_id=persuader_id,
                target_id=target_id,
                llm_persuader=chosen_condition.roles.llm_persuader,
                llm_target=chosen_condition.roles.llm_target,
                simulated_target=chosen_condition.roles.simulated_target,
                proposition=proposition.id,
                proposition_during_round=(
                    proposition_control.id
                    if chosen_condition.control_dialogue
                    else None
                ),
                persuader_supports_proposition=None,
                turn_limit=chosen_condition.turn_limit,
                minimum_turns=chosen_condition.minimum_turns,
                continuous_measure=chosen_condition.continuous_measure,
                on_reflection=chosen_condition.on_reflection,
                synthetic_audio=chosen_condition.synthetic_audio,
                use_audio=chosen_condition.use_audio,
                show_transcript=chosen_condition.show_transcript,
                control_dialogue=chosen_condition.control_dialogue,
                participant_proposition=chosen_condition.participant_proposition,
                enable_node_belief_survey=(chosen_condition.enable_node_belief_survey),
                no_early_end=chosen_condition.no_early_end,
            )

            # If there's a simulated target, instantly set up its initial cognitive trace
            if chosen_condition.roles.simulated_target:
                bn = BayesianNetwork(**proposition.bayesian_network)
                output_constraints = LLM_HUMAN_LIKE_PROMPT_TEMPLATE.format(
                    max_audio_seconds=chosen_condition.max_audio_seconds,
                    max_message_chars=chosen_condition.max_message_chars,
                )
                persona_value = chosen_condition.roles.simulated_target_persona
                persona = (
                    TargetPersona(persona_value) if persona_value is not None else None
                )
                persona_for_sampling = persona or TargetPersona.RANDOM
                sim_target = SimulatedTarget(
                    bn=bn,
                    llm_model=chosen_condition.roles.simulated_target,
                    output_constraints=output_constraints,
                    persona=persona_for_sampling,
                    use_rhetorical_dimensions=(
                        not chosen_condition.simulated_target_no_rhetoric
                    ),
                )
                rd_orm.simulated_target_trace = sim_target.model_dump()
            elif (
                chosen_condition.roles.llm_target
                and chosen_condition.llm_target_use_bayes_structure
            ):
                persona_value = chosen_condition.roles.simulated_target_persona
                persona = (
                    TargetPersona(persona_value) if persona_value is not None else None
                )
                persona_for_sampling = persona or TargetPersona.RANDOM
                rd_orm.simulated_target_trace = {
                    "susceptibilities": susceptibilities_for_persona(
                        persona_for_sampling
                    ),
                    "persona": persona.value if persona is not None else None,
                }

            session.add(rd_orm)
            session.commit()
            session.refresh(rd_orm)

            ## This round is ready to go. Take the participants out of the waiting room.
            # NB: Have to wait until after refresh as the ID initializes the round id
            if chosen_participant:
                chosen_participant.entered_waiting_room = None
                chosen_participant.current_round = rd_orm.id
                session.add(chosen_participant)

            participant.entered_waiting_room = None
            participant.current_round = rd_orm.id
            session.add(participant)

            session.commit()

        is_target = request.participant_id == rd_orm.target_id
        logger.info("r=%s p=%s is target %s", rd_orm.id, participant.id, is_target)
        last_sent_msg: SentMessage = get_last_sent_message(
            session, is_target=None, round_id=rd_orm.id
        )
        return _current_round(
            session,
            rd_orm,
            is_target=is_target,
            last_sent_msg=last_sent_msg,
            settings=settings,
        )

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in current_round: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred") from e
    except HTTPException:
        session.rollback()
        raise


def _current_round(
    session: Session,
    rd_orm: RoundORM,
    is_target: bool,
    last_sent_msg: SentMessage | None,
    settings: ServerSettings,
) -> dict[str, Any]:
    """
    Returns the relevant information about this round.
    """
    if rd_orm.simulated_target and rd_orm.simulated_target_trace:
        changed = False
        if rd_orm.target_initial_belief is None:
            sim_target = SimulatedTarget(**rd_orm.simulated_target_trace)
            initial_belief = sim_target.get_belief_state(0)
            rd_orm.target_initial_belief = initial_belief
            rd_orm.persuader_supports_proposition = initial_belief < 0.5
            changed = True
        if rd_orm.target_initial_node_beliefs is None:
            initial_node_beliefs = _simulated_target_node_beliefs_from_trace(
                rd_orm.simulated_target_trace,
                initial=True,
            )
            if initial_node_beliefs is not None:
                rd_orm.target_initial_node_beliefs = initial_node_beliefs
                changed = True
        if (
            rd_orm.target_final_belief is not None
            and rd_orm.target_final_node_beliefs is None
        ):
            final_node_beliefs = _simulated_target_node_beliefs_from_trace(
                rd_orm.simulated_target_trace,
                initial=False,
            )
            if final_node_beliefs is not None:
                rd_orm.target_final_node_beliefs = final_node_beliefs
                changed = True
        if changed:
            session.add(rd_orm)
            session.commit()

    rd = rd_orm.as_round()
    turns_left = rd.turns_left(is_target=is_target)
    target_can_end_round = rd.target_can_end_round()
    condition = rd_orm.condition()

    time_remaining: int | None = None

    if settings.round_time_limit:
        elapsed = datetime.now(timezone.utc) - rd_orm.created_at.replace(
            tzinfo=timezone.utc
        )
        time_remaining = int(
            (timedelta(seconds=settings.round_time_limit) - elapsed).total_seconds()
        )

    # Get the last msg sent to this player or None if it is their turn
    last_msg_received: dict[str, Any] | None = None
    if last_sent_msg is not None and last_sent_msg.is_target != is_target:
        assert not last_sent_msg.flagged
        last_msg_received = {
            "type": "response",
            "text": last_sent_msg.message_content,
            "audio": last_sent_msg.audio,
            "turns_left": turns_left,
            "target_can_end_round": target_can_end_round,
            "sentences": Round.split_into_sentences(
                last_sent_msg.message_content or ""
            ),
        }
    # Whether the participant is the one waiting
    waiting: bool = (
        # if we are the target and there have been no messages sent
        (last_sent_msg is None and is_target)
        # or if we did not send the last message
        or (last_sent_msg and last_sent_msg.is_target == is_target)
    )

    prompt: str | None = None
    prompt_during_round: str | None = None
    if rd.target_initial_belief is not None or is_target:
        prompt = markdown.markdown(
            rd.prompt(is_target=is_target, during_round=False),
            extensions=["md_in_html"],
        )

        prompt_during_round = markdown.markdown(
            rd.prompt(is_target=is_target, during_round=True),
            extensions=["md_in_html"],
        )
    belief_survey_items = rd.belief_survey_items()
    belief_survey_enabled = bool(condition.enable_node_belief_survey and is_target)
    if belief_survey_enabled and not belief_survey_items:
        raise ValueError(
            "Node-belief survey is enabled for this condition, but the selected "
            "proposition has no belief nodes."
        )
    response = {
        "round_id": rd_orm.id,
        "is_target": is_target,
        "prompt": prompt,
        "prompt_during_round": prompt_during_round,
        "use_audio": condition.use_audio,
        "show_transcript": condition.show_transcript,
        "continuous_measure": condition.continuous_measure,
        "on_reflection": condition.on_reflection,
        "time_remaining": time_remaining,
        "target_final_belief": rd.target_final_belief,
        "target_initial_belief": rd.target_initial_belief,
        "last_msg_received": last_msg_received,
        "messages": get_user_messages(session, rd_orm.id, is_target),
        "turns_left": turns_left,
        "target_can_end_round": target_can_end_round,
        "waiting": waiting,
        "belief_survey": {
            "enabled": bool(belief_survey_enabled and belief_survey_items),
            "items": belief_survey_items,
            "initial_node_beliefs": rd.target_initial_node_beliefs,
            "final_node_beliefs": rd.target_final_node_beliefs,
        },
    }

    logger.info("r=%s current_round response: %r", rd_orm.id, response)
    return response


def _final_continuous_measure(
    round_id: int,
    engine: Engine,
    last_mouse_trace: dict[str, Any] | None = None,
    last_serial_question: float | None = None,
    last_serial_question_sentences: list[float] | None = None,
    last_message_highlight: list[dict[str, Any]] | None = None,
):
    serial_sentence_values = normalize_serial_sentence_values(
        last_serial_question_sentences,
        context="_final_continuous_measure",
        round_id=round_id,
    )

    message_highlight_value = normalize_message_highlight(
        last_message_highlight,
        context="_final_continuous_measure",
        round_id=round_id,
    )

    serialized_serial_question: float | None = None
    if last_serial_question is not None:
        try:
            serialized_serial_question = float(last_serial_question)
        except (TypeError, ValueError):
            serialized_serial_question = None

    with Session(engine) as session:
        rd_orm = session.get(RoundORM, round_id)
        assert rd_orm is not None
        session.add(rd_orm)

        if (
            last_mouse_trace is not None
            or serialized_serial_question is not None
            or (serial_sentence_values and len(serial_sentence_values) > 0)
            or (message_highlight_value and len(message_highlight_value) > 0)
        ):
            sent_message = SentMessage(
                is_target=True,
                audio=None,
                original_audio=None,
                transcript=None,
                message_content=None,
                thought_content=None,
                reasoning_trace=None,
                flagged=False,
                round_id=round_id,
                last_mouse_trace=last_mouse_trace,
                last_serial_question=serialized_serial_question,
                last_serial_question_sentences=serial_sentence_values,
                last_message_highlight=message_highlight_value,
            )
            logger.info("Adding final dummy sent message for rd=%d", round_id)
            session.add(sent_message)

        session.commit()


def _round_result(round_id: int) -> dict[str, Any]:
    """
    Get information on the current round
    """
    with Session(global_engine) as session:
        rd_orm = session.get(RoundORM, round_id)

        assert rd_orm.finished(), f"Round {round_id} not completed"

        return {
            "type": "round_result",
            "persuader_message": markdown.markdown(
                rd_orm.round_result(is_target=False),
                extensions=["md_in_html"],
            ),
            "target_message": markdown.markdown(
                rd_orm.round_result(is_target=True),
                extensions=["md_in_html"],
            ),
        }


def _belief_survey_node_ids_for_round(rd_orm: RoundORM) -> list[str]:
    """
    Return ordered belief-node ids for one round's proposition network.

    Args:
        rd_orm: Round ORM row.

    Returns:
        Ordered belief ids (for example ``Belief_1``). Empty list when absent.
    """
    proposition_obj = rd_orm.proposition_obj
    if proposition_obj is None or not isinstance(
        proposition_obj.bayesian_network, dict
    ):
        return []
    belief_nodes = proposition_obj.bayesian_network.get("belief_nodes")
    if not isinstance(belief_nodes, list):
        return []
    return [f"Belief_{idx}" for idx in range(1, len(belief_nodes) + 1)]


def _normalize_node_beliefs_payload(
    raw_payload: object,
    *,
    valid_node_ids: list[str],
) -> dict[str, float]:
    """
    Normalize and validate node-belief survey payload.

    Args:
        raw_payload: Incoming JSON payload.
        valid_node_ids: Allowed node ids for this round.

    Returns:
        Normalized node-belief mapping.
    """
    if not isinstance(raw_payload, dict):
        raise ValueError("node_beliefs must be a mapping of node id -> belief.")
    normalized: dict[str, float] = {}
    valid_id_set = set(valid_node_ids)
    for node_id_raw, belief_raw in raw_payload.items():
        node_id = str(node_id_raw)
        if valid_id_set and node_id not in valid_id_set:
            raise ValueError(f"Unknown node id '{node_id}' in node_beliefs.")
        if not isinstance(belief_raw, (int, float)):
            raise ValueError(f"Belief for '{node_id}' must be numeric.")
        belief = float(belief_raw)
        if not 0 <= belief <= 1:
            raise ValueError(f"Belief for '{node_id}' must be between 0 and 1.")
        normalized[node_id] = belief
    if valid_node_ids and set(normalized) != valid_id_set:
        missing = sorted(valid_id_set - set(normalized))
        extras = sorted(set(normalized) - valid_id_set)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extras:
            details.append(f"extra={extras}")
        detail_text = ", ".join(details) if details else "mismatched keys"
        raise ValueError(f"node_beliefs must cover all survey nodes ({detail_text}).")
    return normalized


def _simulated_target_node_beliefs_from_trace(
    trace_payload: dict[str, Any] | None,
    *,
    initial: bool,
) -> dict[str, float] | None:
    """
    Read node-level beliefs from a simulated-target trace snapshot.

    Args:
        trace_payload: Serialized simulated-target trace payload.
        initial: Whether to read the first (True) or last (False) distribution.

    Returns:
        Node-id to belief mapping, or None when unavailable/invalid.
    """
    return node_beliefs_from_trace_payload(trace_payload, initial=initial)


def _llm_target_self_report_node_beliefs(
    round_obj: Round,
    model: str,
) -> dict[str, float] | None:
    """
    Ask an llm_target to self-report current node-level beliefs.

    Args:
        round_obj: Current round state.
        model: LLM target model name.

    Returns:
        Parsed node-belief mapping, or None when survey/parse/call fails.
    """
    items = round_obj.belief_survey_items()
    if not items:
        return None
    node_ids = [item["id"] for item in items]

    try:
        messages = round_obj.llm_target_node_belief_report_messages()
    except ValueError as exc:
        logger.warning(
            "LLM target node-belief prompt construction failed for model=%s: %s",
            model,
            exc,
        )
        return None

    try:
        response = call_llm(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=256,
        )
    except ValueError as exc:
        logger.warning(
            "LLM target node-belief self-report call failed for model=%s: %s",
            model,
            exc,
        )
        return None

    text = response.get("text")
    if not isinstance(text, str):
        return None
    return Round.parse_llm_target_node_beliefs_response(text, node_ids)


def _make_choice(
    round_id: int,
    initial: bool,
    belief: float,
    node_beliefs: dict[str, float] | None = None,
):
    """
    Register that a choice is made in the web socket.
    """
    with Session(global_engine) as session:
        rd_orm = session.get(RoundORM, round_id)
        try:
            belief_value = float(belief)
        except (TypeError, ValueError) as exc:
            raise ValueError("belief must be numeric.") from exc
        if not 0 <= belief_value <= 1:
            raise ValueError("belief must be between 0 and 1.")

        if initial:
            assert not rd_orm.messages
            # NB: not asserting these as we call make choice
            # repeatedly in cases of a server error
            # assert rd_orm.target_initial_belief is None
            # assert rd_orm.persuader_supports_proposition is None
            rd_orm.target_initial_belief = belief_value
            rd_orm.persuader_supports_proposition = belief_value < 0.5
            if node_beliefs is not None:
                valid_node_ids = _belief_survey_node_ids_for_round(rd_orm)
                rd_orm.target_initial_node_beliefs = _normalize_node_beliefs_payload(
                    node_beliefs,
                    valid_node_ids=valid_node_ids,
                )
        else:
            assert rd_orm.timed_out or not rd_orm.turns_left(is_target=True)

            assert rd_orm.target_final_belief is None
            rd_orm.target_final_belief = belief_value
            if node_beliefs is not None:
                valid_node_ids = _belief_survey_node_ids_for_round(rd_orm)
                rd_orm.target_final_node_beliefs = _normalize_node_beliefs_payload(
                    node_beliefs,
                    valid_node_ids=valid_node_ids,
                )

        session.add(rd_orm)
        session.commit()


def _clear_round(round_id: int):
    """
    Clear both participants out of the round.
    """
    with Session(global_engine) as session:
        rd_orm = session.get(RoundORM, round_id)
        if rd_orm.target_final_belief is None:
            # The round is not over; do not boot the participants yet.
            return
        for pid in (rd_orm.persuader_id, rd_orm.target_id):
            if pid is not None:
                p2 = session.get(Participant, pid)
                if p2:
                    p2.current_round = None
                    p2.entered_waiting_room = None
                    session.add(p2)
        session.commit()


def _get_prompts(round_id: int) -> (str, str):
    """
    Returns the prompts for the round.
    """
    with Session(global_engine) as session:
        rd_orm = session.get(RoundORM, round_id)

        assert rd_orm.target_initial_belief is not None

        persuader_prompt = rd_orm.prompt(is_target=False)

        target_prompt = rd_orm.prompt(is_target=True, during_round=False)

        target_prompt_in_rd = rd_orm.prompt(is_target=True, during_round=True)

        prompts = (persuader_prompt, target_prompt, target_prompt_in_rd)

        return (
            markdown.markdown(prompt, extensions=["md_in_html"]) for prompt in prompts
        )


@app.websocket("/ws/round/{round_id}/participant/{participant_id}")
async def round_ws(
    websocket: WebSocket,
    round_id: int,
    participant_id: int,
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Duplex channel for turn-by-turn exchange. Clients send
      {"type":"message","content": ... }
    Server replies
      {"type":"response","content": ...,"turns_left": ...}
    and when the round ends:
      {"type":"round_over"} then close.
    """

    #######
    # Get info about the round
    #######
    with Session(global_engine) as session:
        rd_orm = session.get(RoundORM, round_id)
        rd = rd_orm.as_round()
        participant = session.get(Participant, participant_id)
        if (
            rd_orm is None
            or participant is None
            or rd_orm.finished()
            or participant.current_round != round_id
            or participant_id not in (rd_orm.persuader_id, rd_orm.target_id)
        ):
            # 1008 — Policy Violation
            # (you use this to reject clients who aren't in the round or have a finished round)
            await websocket.close(code=1008)
            return
        is_target = participant_id == rd_orm.target_id

        other_participant_id = None
        condition: Condition = rd_orm.condition()
        if condition.roles.is_paired_human():
            other_participant_id = (
                rd_orm.persuader_id if is_target else rd_orm.target_id
            )

    # register with the websocket
    await manager.connect(round_id, participant_id, websocket)
    logger.info("WS registered p=%s r=%s", participant_id, round_id)

    # LLM targets need to send a fake initial belief
    if condition.roles.llm_target and rd.target_initial_belief is None:
        # LLM target initial belief is chosen here; keep in sync with runner
        belief: float = rd.initial_belief_per_policy()
        initial_node_beliefs: dict[str, float] | None = None
        if condition.enable_node_belief_survey:
            rd_for_report = rd.model_copy(
                update={
                    "target_initial_belief": belief,
                    "persuader_supports_proposition": belief < 0.5,
                }
            )
            initial_node_beliefs = _llm_target_self_report_node_beliefs(
                rd_for_report,
                condition.roles.llm_target,
            )
            if initial_node_beliefs is None and rd_for_report.belief_survey_items():
                logger.warning(
                    "LLM target pre-survey node-belief self-report failed for round %s.",
                    round_id,
                )
        _make_choice(
            round_id=round_id,
            initial=True,
            belief=belief,
            node_beliefs=initial_node_beliefs,
        )
        persuader_prompt, _, _ = _get_prompts(round_id=round_id)

        logger.info(
            "WS send to p=%s r=%s: round started",
            participant_id,
            round_id,
        )
        await websocket.send_json(
            {
                "type": "round_started",
                "prompt": persuader_prompt,
            }
        )

    #######
    # Loop to send and receive messages
    #######
    sent_round_over = False

    try:
        while True:

            ########
            # If the round just ended for either player or if they have timed out,
            # notify & close
            ########
            turns_left, timed_out = round_over_state(
                global_engine,
                round_id,
                is_target=is_target,
                round_time_limit=settings.round_time_limit,
            )

            if not turns_left or timed_out and not sent_round_over:
                logger.info("WS round_over for p=%s r=%s", participant_id, round_id)
                await websocket.send_json({"type": "round_over"})
                # NB: We don't break here because the target has to
                # make their final choice
                sent_round_over = True

                # LLM target makes a choice and ends the round
                if condition.roles.llm_target:
                    # Prefer using last known serial measure; fall back to random per policy
                    final_node_beliefs: dict[str, float] | None = None
                    with Session(global_engine) as session:
                        rd_orm2 = session.get(RoundORM, round_id)
                        rd2 = rd_orm2.as_round()
                        belief_val = rd2.final_belief_per_policy()
                        if condition.enable_node_belief_survey:
                            final_node_beliefs = _llm_target_self_report_node_beliefs(
                                rd2,
                                condition.roles.llm_target,
                            )
                            if final_node_beliefs is None and rd2.belief_survey_items():
                                logger.warning(
                                    "LLM target post-survey node-belief self-report "
                                    "failed for round %s.",
                                    round_id,
                                )
                    _make_choice(
                        round_id=round_id,
                        initial=False,
                        belief=belief_val,
                        node_beliefs=final_node_beliefs,
                    )

                    round_result = _round_result(round_id=round_id)
                    assert condition.roles.human_persuader

                    logger.info(
                        "WS send to p=%s r=%s: round result",
                        participant_id,
                        round_id,
                    )
                    await websocket.send_json(round_result)

                    break

                if condition.roles.simulated_target:
                    with Session(global_engine) as session:
                        rd_orm2 = session.get(RoundORM, round_id)
                        trace = rd_orm2.simulated_target_trace or {}
                        sim_target = SimulatedTarget(**trace)
                        belief_val = sim_target.get_belief_state(
                            len(sim_target.belief_history) - 1
                        )
                        rd_orm2.target_final_belief = belief_val
                        node_beliefs = _simulated_target_node_beliefs_from_trace(
                            rd_orm2.simulated_target_trace,
                            initial=False,
                        )
                        if node_beliefs is not None:
                            rd_orm2.target_final_node_beliefs = node_beliefs
                        session.add(rd_orm2)
                        session.commit()

                    round_result = _round_result(round_id=round_id)
                    assert condition.roles.human_persuader

                    logger.info(
                        "WS send to p=%s r=%s: round result",
                        participant_id,
                        round_id,
                    )
                    await websocket.send_json(round_result)

                    break

            ########
            # Get messages from the client
            ########
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=settings.participant_conversation_timeout.total_seconds(),
                )
            except asyncio.TimeoutError:
                # no data for a while, assume client dropped
                logger.warning(
                    "WS timeout: no message from p=%s r=%s",
                    participant_id,
                    round_id,
                )
                break
            except ValueError:
                # malformed JSON
                logger.error(
                    "WS received invalid JSON from p=%s r=%s",
                    participant_id,
                    round_id,
                )
                await websocket.send_json({"type": "error", "detail": "invalid JSON"})
                continue

            ######
            # Process the client's message
            ######
            msg_type: str | None = raw.get("type")
            logger.info(
                "WS received %s from p=%s r=%s", msg_type, participant_id, round_id
            )

            content: str | None = None
            last_serial_question: float | None = None
            last_serial_question_sentences: list[float] | None = None
            last_message_highlight: list[dict[str, Any]] | None = None
            last_mouse_trace: list[dict[str, Any]] | None = None

            partner_ws = await manager.get_partner(round_id, participant_id)
            logger.info("WS partner ws, %s", partner_ws)

            if msg_type not in [
                "message",
                "target_ends_round",
                "make_choice",
                "final_continuous_measure",
                "on_reflection_highlights",
            ]:
                # Malformed input
                logger.warning(
                    "WS invalid type %r from p=%s r=%s",
                    msg_type,
                    participant_id,
                    round_id,
                )
                await websocket.send_json({"type": "error", "detail": "invalid type"})
                continue

            if msg_type == "target_ends_round":
                ######
                # The client has ended the round
                ######
                logger.info(
                    "WS p=%s r=%s has ended the round",
                    participant_id,
                    round_id,
                )

                assert is_target, "Only the target can end the round."

                with Session(global_engine) as session:
                    rd_orm = session.get(RoundORM, round_id)
                    rd = rd_orm.as_round()
                    if not rd.target_can_end_round():
                        logger.info(
                            "WS p=%s r=%s: target attempted early end",
                            participant_id,
                            round_id,
                        )
                        await websocket.send_json(
                            {
                                "type": "error",
                                "detail": "round cannot end yet",
                            }
                        )
                        continue
                    rd_orm.target_ended_round = True
                    session.add(rd_orm)
                    session.commit()

                if partner_ws:
                    logger.info(
                        "WS send to p=%s r=%s: end round",
                        other_participant_id,
                        round_id,
                    )
                    await partner_ws.send_json(
                        {
                            "type": "round_over",
                        }
                    )

                # NB: We end the rounds for the target on the next pass
                continue

            if msg_type == "on_reflection_highlights":
                logger.info(
                    "WS p=%s r=%s on reflection highlights",
                    participant_id,
                    round_id,
                )

                # Add to the round the highlights we have received
                with Session(global_engine) as session:
                    rd_orm = session.get(RoundORM, round_id)
                    assert rd_orm is not None
                    rd_orm.on_reflection_highlights = raw.get(
                        "on_reflection_highlights"
                    )
                    session.add(rd_orm)
                    session.commit()

                break

            if msg_type == "final_continuous_measure":
                logger.info(
                    "WS p=%s r=%s final continuous measure",
                    participant_id,
                    round_id,
                )

                _final_continuous_measure(
                    round_id=round_id,
                    last_mouse_trace=raw.get("last_mouse_trace"),
                    last_serial_question=raw.get("last_serial_question"),
                    last_serial_question_sentences=raw.get(
                        "last_serial_question_sentences"
                    ),
                    last_message_highlight=raw.get("last_message_highlight"),
                    engine=global_engine,
                )
                continue

            if msg_type == "make_choice":
                assert is_target

                initial: bool = raw.get("initial")
                belief: float = raw.get("belief")
                node_beliefs_raw = raw.get("node_beliefs")

                try:
                    _make_choice(
                        round_id=round_id,
                        initial=initial,
                        belief=belief,
                        node_beliefs=(
                            node_beliefs_raw
                            if isinstance(node_beliefs_raw, dict)
                            else None
                        ),
                    )
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue

                persuader_prompt, target_prompt, target_prompt_in_rd = _get_prompts(
                    round_id=round_id
                )

                if initial:
                    # If this is the intial belief... AND it is an LLM persuader
                    # We need to query for a starting message by the LLM (below)
                    # Otherwise the persuader is a human and we have no message processing to do yet
                    # But we do need to tell them to start the round
                    if condition.roles.human_persuader:
                        assert partner_ws

                        logger.info(
                            "WS send to p=%s r=%s: round started",
                            other_participant_id,
                            round_id,
                        )
                        await partner_ws.send_json(
                            {
                                "type": "round_started",
                                "prompt": persuader_prompt,
                            }
                        )

                    logger.info(
                        "WS send to p=%s r=%s: round started",
                        participant_id,
                        round_id,
                    )
                    await websocket.send_json(
                        {
                            "type": "round_started",
                            "prompt": target_prompt,
                            "prompt_during_round": target_prompt_in_rd,
                        }
                    )
                else:
                    # Final make choice -> Round is over so send the result
                    round_result = _round_result(round_id=round_id)
                    if partner_ws:
                        assert condition.roles.human_persuader

                        logger.info(
                            "WS send to p=%s r=%s: round result",
                            other_participant_id,
                            round_id,
                        )
                        await partner_ws.send_json(round_result)

                    logger.info(
                        "WS send to p=%s r=%s: round result",
                        participant_id,
                        round_id,
                    )
                    await websocket.send_json(round_result)

                    if not condition.on_reflection:
                        break  # The round is over; end the websocket

                # NB: Not continuing here as we might need to get the first LLM
                # response and send it to the target
            elif msg_type == "message":
                content = raw.get("content")
                last_serial_question = raw.get("last_serial_question")
                last_serial_question_sentences = raw.get(
                    "last_serial_question_sentences"
                )
                last_message_highlight = raw.get("last_message_highlight")
                last_mouse_trace = raw.get("last_mouse_trace")

            _, timed_out_now = round_over_state(
                global_engine,
                round_id,
                is_target=is_target,
                round_time_limit=settings.round_time_limit,
            )

            if msg_type == "message" and (timed_out_now or sent_round_over):
                logger.info(
                    "WS ignore message after round_over p=%s r=%s",
                    participant_id,
                    round_id,
                )
                await websocket.send_json({"type": "round_over"})
                sent_round_over = True
                continue

            ####
            # Process the message and perhaps get an LLM response -- Blocking
            ####
            flagged = False
            flagged_reason = None

            sent_message: SentMessageBase | None = None
            received_message: SentMessageBase | None = None
            our_turns_left: int | bool | None = None
            their_turns_left: int | bool | None = None
            target_can_end_round: bool

            # Either we have received a message or need to query an LLM for the persuader
            # 1st turn
            if content or msg_type == "make_choice":
                # offload all blocking work here
                try:
                    (
                        sent_message,
                        received_message,
                        our_turns_left,
                        their_turns_left,
                        target_can_end_round,
                        round_over,
                    ) = await run_in_threadpool(
                        message_processing.process_message_and_response,
                        content=content,
                        is_target=is_target,
                        round_id=round_id,
                        engine=global_engine,
                        llm_target_end_game_prob=settings.llm_target_end_game_prob,
                        use_audio=condition.use_audio,
                        synthetic_audio=condition.synthetic_audio,
                        max_response_chars=condition.max_message_chars,
                        max_audio_duration_s=settings.max_audio_seconds,
                        round_time_limit=settings.round_time_limit,
                        last_serial_question=last_serial_question,
                        last_mouse_trace=last_mouse_trace,
                        last_serial_question_sentences=last_serial_question_sentences,
                        last_message_highlight=last_message_highlight,
                        llm_persuader_reasoning_effort=settings.llm_persuader_reasoning_effort,
                        delay_sleep=not settings.dev_environment,
                    )
                    if round_over:
                        logger.info(
                            "WS round_over after processing p=%s r=%s",
                            participant_id,
                            round_id,
                        )
                        await websocket.send_json({"type": "round_over"})
                        sent_round_over = True
                        # Do not continue here, so the final message can be sent to the partner.

                    if content:
                        assert sent_message
                        flagged = sent_message.flagged
                        flagged_reason = sent_message.flagged_response
                except ValueError:
                    flagged = True
                    flagged_reason = FlaggedResponse.SERVER_ERROR
                    logger.exception("Unexpected ValueError while processing message")
            else:
                flagged = True
                flagged_reason = FlaggedResponse.NO_WORDS

            # Fetching the ws again as it might have changed
            # Error if they have disconnected
            partner_ws = await manager.get_partner(round_id, participant_id)
            if partner_ws and websocket.client_state != WebSocketState.CONNECTED:
                flagged = True
                flagged_reason = FlaggedResponse.DISCONNECTED

            if flagged:
                # Moderate out the flagged message from the ppt.
                # and do nothing else
                logger.info("WS send p=%s r=%s: flagged msg", participant_id, round_id)
                await websocket.send_json(
                    {
                        "type": "flagged",
                        "reason": flagged_reason,
                    }
                )
                sent_message = None
                continue

            if sent_message and sent_message.message_content is not None:
                ########
                # Send back our own transcript
                ########
                logger.info(
                    "WS send to p=%s r=%s: reply=%r",
                    participant_id,
                    round_id,
                    sent_message.message_content,
                )
                await websocket.send_json(
                    {
                        "type": "echo",
                        "text": sent_message.message_content,
                        "sentences": Round.split_into_sentences(
                            sent_message.message_content or ""
                        ),
                    }
                )

                ########
                # Send our message to the partner's ws
                ########
                if partner_ws:
                    assert condition.roles.human_persuader
                    logger.info(
                        "WS send to p=%s r=%s: sent=%r turns=%r",
                        other_participant_id,
                        round_id,
                        sent_message.message_content,
                        their_turns_left,
                    )
                    transcript = token_time_totals_verbose(sent_message.transcript)
                    await partner_ws.send_json(
                        {
                            "type": "response",
                            "text": sent_message.message_content,
                            "audio": sent_message.audio,
                            "transcript": transcript,
                            "turns_left": their_turns_left,
                            "target_can_end_round": target_can_end_round,
                            "sentences": Round.split_into_sentences(
                                sent_message.message_content or ""
                            ),
                        }
                    )

            ########
            # Send back the partner's turn (or the LLM reply)
            ########
            if received_message and received_message.message_content is not None:
                logger.info(
                    "WS send to p=%s r=%s: reply=%r turns=%r",
                    participant_id,
                    round_id,
                    received_message.message_content,
                    our_turns_left,
                )
                transcript = token_time_totals_verbose(received_message.transcript)
                await websocket.send_json(
                    {
                        "type": "response",
                        "text": received_message.message_content,
                        "audio": received_message.audio,
                        "transcript": transcript,
                        "turns_left": our_turns_left,
                        "target_can_end_round": target_can_end_round,
                        "sentences": Round.split_into_sentences(
                            received_message.message_content or ""
                        ),
                    }
                )

            logger.info(
                "WS p=%s r=%s, our turns %r, their turns %r",
                participant_id,
                round_id,
                our_turns_left,
                their_turns_left,
            )
    except WebSocketDisconnect:
        logger.info("WS client disconnected p=%s r=%s", participant_id, round_id)
    finally:
        # ALWAYS clean up the manager entry and close the socket
        await manager.disconnect(round_id, participant_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                pass
            except RuntimeError as exc:
                msg = str(exc)
                # String match on the specific RuntimeError message
                if 'Cannot call "send" once a close message has been sent.' not in msg:
                    raise

        #######
        # Clear both participants out of the round
        #######
        _clear_round(round_id)


@app.post("/send_feedback/")
def send_feedback(
    request: FeedbackRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Accepts the feedback from the participant.

    Raises an exception if

    - the participant does not exist
    - the participant is not done with their rounds
    - the participant has already answered the feedback
    """
    logger.info(request)

    participant = get_participant(session, request.participant_id)

    rounds = get_participant_rounds(participant, session)

    rounds_left = (settings.rounds_per_participant - len(rounds)) > 0

    if rounds_left and not participant.waited_too_long:
        message = "Participant has rounds left."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    if participant.feedback is not None:
        message = "Participant has given feedback."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    participant.feedback = request.feedback
    session.add(participant)
    session.commit()

    logger.info(
        "Feedback from participant %s stored: %s",
        request.participant_id,
        request.feedback,
    )


@app.post("/participant_rounds/")
def participant_rounds(
    request: ParticipantRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Get information on the rounds the participant has played in for use in disclosing
    to that participant in which conversations they interacted with other humans vs.
    LLMs.

    Returns a dict with keys

    - 'human_conversations': a list of bools indicating whether each of the conversaitons
        that the participant was were with another human
    - 'rounds_remaining': a bool indicating whether or not the participant has more rounds
        left

    Raises an exception if

    - the participant does not exist
    """
    logger.info(request)

    participant = get_participant(session, request.id)

    rounds = get_participant_rounds(participant, session)

    human_conversations = list(
        filter(
            lambda rd: (participant.is_target() and rd.persuader_id)
            or (not participant.is_target() and rd.target_id),
            rounds,
        )
    )

    return {
        "num_human_conversations": len(human_conversations),
        "rounds_remaining": settings.rounds_per_participant - len(rounds),
        "num_rounds": len(rounds),
        "total_rounds": settings.rounds_per_participant,
        "completion_code": settings.completion_code,
    }


@app.get("/server_config.js")
def server_config(settings: Annotated[ServerSettings, Depends(get_settings)]):
    """
    Serves a javascript page for the server settings.
    """
    config_obj = {
        "development_mode": settings.dev_environment,
        "may_use_audio": settings.may_use_audio,
        "max_audio_seconds": settings.max_audio_seconds,
        "max_message_chars": settings.max_message_chars,
        "post_play_delay": settings.post_play_delay,
        "participant_propositions_required": settings_require_participant_propositions(
            settings
        ),
    }
    body = "window.SERVER_CONFIG = " + json.dumps(config_obj) + ";"
    return Response(body, media_type="application/javascript")
