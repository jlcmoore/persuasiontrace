"""
src/api/message_processing.py

Author: Jared Moore
Date: July, 2025

Utilities for processing messages.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Tuple

import openai
from fastapi import WebSocket
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

# Import both real and dummy implementations
import experiment.dummy_endpoints as _dummy
from experiment.condition import DEFAULT_MAX_MESSAGE_CHARS
from experiment.condition import ContinuousMeasure as _CM
from experiment.condition import Roles
from experiment.endpoints import is_refusal as _prod_is_refusal
from experiment.endpoints import moderate_content as _prod_moderate
from experiment.endpoints import synthesize_audio as _prod_synthesize
from experiment.endpoints import transcribe_audio as _prod_transcribe
from experiment.llm_utils import (
    call_llm,
    model_supports_reasoning,
    split_thought_from_response,
)
from experiment.persuader_policies import (
    is_naive_persuader_model,
    naive_persuader_action_for_round,
)
from experiment.round import LLM_HUMAN_LIKE_PROMPT_TEMPLATE, Round
from experiment.utils import (
    limit_text_to_char_and_audio_budget,
    make_text_transcript,
    normalize_message_highlight,
    normalize_serial_sentence_values,
)
from simulation.target import SimulatedTarget

from .sql_model import FlaggedResponse, RoundORM, SentMessage, SentMessageBase
from .utils import round_over_state

# Set up logginga
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_GENERATION_ATTEMPTS = 5

_DUMMY_MODE: bool = False
#
# Module-level callables (can be swapped out in tests)
#
transcribe_audio_fn: Callable[[str], Tuple[dict[str, Any], str]] = _prod_transcribe
synthesize_audio_fn: Callable[[str], bytes] = _prod_synthesize
moderate_content_fn: Callable[[str], bool] = _prod_moderate
is_refusal_fn: Callable[[str], bool] = _prod_is_refusal


def _limit_llm_response_content(
    response_content: str,
    *,
    use_audio: bool,
    max_response_chars: int,
    max_audio_duration_s: int | None,
) -> str:
    """
    Limit LLM responses by character count and, for audio, by duration.

    Parameters:
    - response_content: raw LLM response text
    - use_audio: whether the response will be synthesized to audio
    - max_response_chars: maximum allowed characters
    - max_audio_duration_s: maximum audio duration in seconds (optional)

    Returns:
    - str: a possibly truncated response
    """
    return limit_text_to_char_and_audio_budget(
        response_content,
        use_audio=use_audio,
        max_response_chars=max_response_chars,
        max_audio_duration_s=max_audio_duration_s,
    )


def _clear_processing_flag(engine: Engine, round_id: int) -> None:
    """Clear the processing_msg flag for the round."""
    with Session(engine) as session:
        rd_orm = session.get(RoundORM, round_id)
        rd_orm.processing_msg = False
        session.add(rd_orm)
        session.commit()


class ConnectionManager:
    """
    Tracks active WebSocket connections by (round_id, participant_id).
    """

    def __init__(self) -> "Websocket":
        """
        Initializes the connection manager.
        """
        # round_id -> { participant_id -> WebSocket }
        self.active: dict[int, dict[int, WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, round_id: int, participant_id: int, ws: WebSocket):
        """
        Adds the round and participant's websocket
        """
        await ws.accept()
        async with self._lock:
            self.active.setdefault(round_id, {})[participant_id] = ws

    async def disconnect(self, round_id: int, participant_id: int):
        """
        Takes the round and participant's web socket out of management.
        """
        async with self._lock:
            part = self.active.get(round_id)
            if part and participant_id in part:
                part.pop(participant_id)
                if not part:
                    self.active.pop(round_id)

    async def get_partner(self, round_id: int, participant_id: int) -> "Websocket":
        """
        Return the other side’s WebSocket (if any).
        """
        async with self._lock:
            conns = self.active.get(round_id, {})
            for pid, sock in conns.items():
                if pid != participant_id:
                    return sock
            return None


def use_dummy_endpoints() -> None:
    """
    Switch all audio/moderation calls to the dummy implementations.
    """
    global transcribe_audio_fn, synthesize_audio_fn, moderate_content_fn, is_refusal_fn, _DUMMY_MODE
    transcribe_audio_fn = _dummy.transcribe_audio
    synthesize_audio_fn = _dummy.synthesize_audio
    moderate_content_fn = _dummy.moderate_content
    is_refusal_fn = _dummy.is_refusal
    _DUMMY_MODE = True


def _receive_simulated_target_response(
    rd_orm: RoundORM, session: Session
) -> Tuple[str | None, str | None, str | None]:
    """Helper to process a turn for a SimulatedTarget."""
    logger.info("Calling SimulatedTarget for response")

    # Load the target's cognitive trace from the DB
    trace = rd_orm.simulated_target_trace
    if not trace:
        raise ValueError("Round is missing simulated_target_trace")

    sim_target = SimulatedTarget(**trace)
    if not sim_target.output_constraints:
        condition = rd_orm.condition()
        sim_target.output_constraints = LLM_HUMAN_LIKE_PROMPT_TEMPLATE.format(
            max_audio_seconds=condition.max_audio_seconds,
            max_message_chars=condition.max_message_chars,
        )

    # We need to construct the conversation history it expects
    conversation_history = [
        {
            "role": "persuader" if not msg.is_target else "target",
            "content": msg.message_content,
        }
        for msg in rd_orm.non_flagged_messages()
    ]

    # Generate the response and update the internal state
    response_content = sim_target.take_turn(conversation_history)
    logger.info(
        "SimulatedTarget belief: initial=%.4f current=%.4f turns=%d",
        sim_target.get_belief_state(0),
        sim_target.get_belief_state(len(sim_target.belief_history) - 1),
        len(sim_target.belief_history) - 1,
    )

    # Save the mutated state back to the database object
    rd_orm.simulated_target_trace = sim_target.model_dump()
    session.add(rd_orm)

    # Return the response (no explicit thought/reasoning trace for now)
    return response_content, None, None


def _llm_target_self_report_belief(round_obj: Round, model: str) -> float | None:
    """
    Ask an llm_target to self-report current proposition belief.

    Args:
        round_obj: Current round state before the next target message is stored.
        model: LLM target model name.

    Returns:
        Parsed belief in [0,1], or None when parsing/call fails.
    """
    messages = round_obj.llm_target_belief_report_messages()
    try:
        response = call_llm(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=16,
        )
    except ValueError as exc:
        logger.warning(
            "LLM target self-report call failed for model=%s: %s", model, exc
        )
        return None
    text = response.get("text")
    if not isinstance(text, str):
        return None
    return Round.parse_llm_target_belief_response(text)


def receive_response(
    is_target: bool,
    role: Roles,
    rd_orm: RoundORM,
    session: Session,
    llm_persuader_reasoning_effort: str | None = None,
) -> (str | None, str | None, str | None):
    """
    A helper function to get the response an LLM or simulated target.

    Parameters:
    - is_target (bool): whether *this player* the llm or rational target is the target
    - role (Roles): the condition of the round
    - rd_orm (RoundORM): the Round being played, so we can update traces
    - session (Session): the database session
    - llm_persuader_reasoning_effort (str | None): optional explicit
      reasoning effort for LLM persuader calls. When None, the provider default
      reasoning behavior is used.

    Returns:
    - str: The response, or None if no more turns
    - str: the thought content
    - str: the reasoning trace
    """
    logger.info("in receive_response")

    assert not role.is_paired_human()

    # Handle Simulated Target
    if is_target and role.simulated_target:
        return _receive_simulated_target_response(rd_orm, session)

    rd = rd_orm.as_round()

    # Handle standard LLM
    model = role.llm_target if is_target else role.llm_persuader
    assert model, "No LLM specified in Roles"
    if not is_target and is_naive_persuader_model(model):
        response_content, thought_content, reasoning_trace = (
            naive_persuader_action_for_round(rd)
        )
        logger.info("Generated naive persuader response: %r", response_content)
        return response_content, thought_content, reasoning_trace

    messages = rd.messages_for_llms(
        is_target=is_target,
        include_intermediate_beliefs=bool(is_target and role.llm_target),
    )

    logger.info(f"Calling {model} for a response to {messages}")

    # Call the relevant LLM, passing in the entire round history
    # TODO: should the temperature really be one here?
    # TODO: Need to have a fail mechanism
    llm_call_kwargs: dict[str, Any] = {"temperature": 1}
    if (
        not is_target
        and llm_persuader_reasoning_effort is not None
        and model_supports_reasoning(model) is True
    ):
        llm_call_kwargs["reasoning_effort"] = str(llm_persuader_reasoning_effort)
    response = call_llm(model=model, messages=messages, **llm_call_kwargs)
    text = response.get("text")

    if not text:
        raise ValueError("Empty response from LLM")

    if is_refusal_fn(text):
        logger.info("Refusal: %s", text)
        raise ValueError("Model refused to answer")

    thought_content, response_content = split_thought_from_response(text)

    # TODO: figure out how to get the reasoning trace from the api
    reasoning_trace = None

    # NB: We could moderate the LLM's response but why?
    # if moderate_content_fn(response_content):
    #     raise ValueError("Response content was flagged by moderation")

    logger.info(f"Got response from {model}: {response_content!r}")

    return response_content, thought_content, reasoning_trace


# TODO: refactor this so that it accepts message_content and also a flag for audio or not
# and only if the flag is set processes it like it is audio
def process_message_and_response(
    content: str | None,
    is_target: bool,
    round_id: int,
    engine: Engine,
    llm_target_end_game_prob: float = 0.5,
    use_audio: bool = True,
    synthetic_audio: bool = False,
    max_response_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    max_audio_duration_s: int | None = None,
    round_time_limit: int | None = None,
    last_serial_question: float | None = None,
    last_serial_question_sentences: list[float] | None = None,
    last_message_highlight: list[dict[str, Any]] | None = None,
    last_mouse_trace: list[dict[str, Any]] | None = None,
    llm_persuader_reasoning_effort: str | None = None,
    delay_sleep: bool = False,
) -> (
    SentMessageBase | None,
    SentMessageBase | None,
    int | bool | None,
    int | bool | None,
    bool,
    bool | bool | None,
):
    """
    Moderates/transcribes the incoming message, possibly calls an LLM to
    generate a reply, and updates the DB.

    By default uses the real endpoints; in tests you can call `use_dummy_endpoints()`
    at the top of your test to switch to the dummy implementations.

    Parameters:
    - content (str): the message sent by the participant (always human).
    - llm_target_end_game_prob (float): the prob. that the LLM target ends the game
    - use_audio (bool): whether to accept and return audio or not
    - synthetic_audio (bool): whether to regenerate participants' audio
    - max_response_chars (int): maximum LLM response characters
    - max_audio_duration_s (int | None): maximum audio duration in seconds
    - round_time_limit (int | None): maximum round duration in seconds
    - is_target (str): whehter or not the participant who sent this message is the target
    - round_id (int): the id of the Round
    - delay_sleep (bool): whether to wait to send a response to make it appear sent by a human
    - llm_persuader_reasoning_effort (str | None): optional explicit
      reasoning effort for LLM persuader calls. When None, the provider default
      reasoning behavior is used.

    Returns:
    - (SentMessageBase | None): the message sent
    - (SentMessageBase | None): the message received
    - (int | bool | None): the number of turns the participant has left, or True if any
    - (int | bool | None): the number of turns the other player has left, or True if any
    - (bool): Whether the target can yet end the round
    - (bool | None): whether the game is over

    Should take place on a different thread.
    """
    logger.info(
        "Processing message for round %s",
        round_id,
    )
    processing_start: datetime = datetime.now(timezone.utc)

    # In case the WS has to restart store whether we are already processing the messages.
    with Session(engine) as session:
        statement = select(RoundORM).filter_by(id=round_id).with_for_update()
        rd_orm = session.exec(statement).one()
        rd_orm.processing_msg = True
        session.add(rd_orm)
        session.commit()

    # Audio content may not be sent in cases in which we just want the model to respond
    sent_message: SentMessage | None = None
    sent_message_base: SentMessageBase | None = None
    other_is_target: bool = not is_target

    if content:
        # 1) Transcribe or accept text directly
        audio_content: str | None = None
        original_audio_content: str | None = None

        if use_audio:
            audio_content = content
            original_audio_content = content
            try:
                full_transcript, message_content = transcribe_audio_fn(content)
            except openai.BadRequestError as err:
                logger.warning(
                    "Audio transcription failed for round %s: %s",
                    round_id,
                    err,
                )
                full_transcript = None
                message_content = None
            else:
                logger.info("Transcribed audio to %r", message_content)

        else:
            message_content = content
            full_transcript = make_text_transcript(message_content)
            logger.info("Received text message: %r", message_content)

        # 2) Moderate the message
        flagged = moderate_content_fn(message_content) if message_content else False
        flagged_response = None
        if flagged:
            flagged_response = FlaggedResponse.INAPPROPRIATE
        elif not message_content or not full_transcript:
            # This should only happen if there is no transcript available
            flagged = True
            flagged_response = FlaggedResponse.NO_WORDS
        elif synthetic_audio:
            # If the message is not flagged has content and we should sythesize a new one.
            audio_content = synthesize_audio_fn(message_content)

        serial_sentence_values = normalize_serial_sentence_values(
            last_serial_question_sentences,
            context="process_message_and_response",
            round_id=round_id,
        )

        serial_question_value: float | None = None
        if last_serial_question is not None:
            try:
                serial_question_value = float(last_serial_question)
            except (TypeError, ValueError):
                serial_question_value = None

        message_highlight_value = normalize_message_highlight(
            last_message_highlight,
            context="process_message_and_response",
            round_id=round_id,
        )

        _, timed_out = round_over_state(
            engine,
            round_id,
            is_target=is_target,
            round_time_limit=round_time_limit,
        )
        if timed_out:
            _clear_processing_flag(engine, round_id)
            return (
                None,
                None,
                None,
                None,
                False,
                True,
            )

        sent_message = SentMessage(
            is_target=is_target,
            audio=audio_content,
            original_audio=original_audio_content,
            transcript=full_transcript,
            message_content=message_content,
            thought_content=None,
            reasoning_trace=None,
            flagged=flagged,
            round_id=round_id,
            last_mouse_trace=last_mouse_trace,
            last_serial_question=serial_question_value,
            last_serial_question_sentences=serial_sentence_values,
            last_message_highlight=message_highlight_value,
        )

        with Session(engine) as session:
            statement = select(RoundORM).filter_by(id=round_id).with_for_update()
            rd_orm = session.exec(statement).one()

            if sent_message.flagged:
                sent_message.flagged_response = flagged_response
                logger.warning("Incoming message flagged; skipping LLM reply")

            logger.info("Adding sent message for round %s", round_id)
            # Fail early if we introduce duplicate messages
            assert rd_orm.is_roles_turn(sent_message.is_target)
            session.add(sent_message)
            session.commit()

            # So that we can return these obects
            sent_message_base = SentMessageBase.model_validate(sent_message)

    # 3) Possibly generate an LLM reply if this is a non-paired-human condition
    received_message: SentMessage | None = None
    received_message_base: SentMessageBase | None = None

    # Get the round information. This should have the message we just added to it.
    with Session(engine) as session:
        statement = select(RoundORM).filter_by(id=round_id).with_for_update()
        rd_orm: RoundORM = session.exec(statement).one()
        # NB: this next line has to happen in a session
        rd: Round = rd_orm.as_round()
    role: Roles = rd.condition.roles

    # With some probability, end the game if the game is one that does not
    # have a fixed turn limit
    llm_target_ends_game = False

    # If we need to calculate a response now
    logger.debug("Sent message: %s", sent_message)
    logger.info("Role: %s", role)
    if (not sent_message or not sent_message.flagged) and not role.is_paired_human():
        _, timed_out = round_over_state(
            engine,
            round_id,
            is_target=is_target,
            round_time_limit=round_time_limit,
        )
        if timed_out:
            _clear_processing_flag(engine, round_id)
            return (
                sent_message_base,
                None,
                None,
                None,
                False,
                True,
            )
        if other_is_target:
            if (
                rd.condition.turn_limit is None
                and rd.target_can_end_round()
                and (role.llm_target or role.simulated_target)
            ):
                llm_target_ends_game = random.random() <= llm_target_end_game_prob

            # If this is the start of a round, we should have picked
            # a belief at the start of the WS
            # and should not have set a final
            assert (
                rd.target_initial_belief is not None and rd.target_final_belief is None
            )

            # TODO: also create a fake mouse / belief trace! add this into SentMessage

        # 3a) get the LLM response
        if rd.turns_left(is_target=other_is_target) and not llm_target_ends_game:
            generation_attempts = 0
            reply_transcript: dict[str, Any] | None = None
            response_content: str | None = None
            while generation_attempts < MAX_GENERATION_ATTEMPTS and (
                not response_content
                or not reply_transcript
                or "words"
                not in reply_transcript  # pylint: disable=unsupported-membership-test
            ):
                try:
                    if _DUMMY_MODE:
                        # echo back whatever we transcribed from the human
                        response_content = (
                            sent_message.message_content if sent_message else ""
                        )
                        thought_content: str | None = None
                        reasoning_trace: str | None = None
                    else:
                        response_content, thought_content, reasoning_trace = (
                            receive_response(
                                other_is_target,
                                role,
                                rd_orm,
                                session,
                                llm_persuader_reasoning_effort=llm_persuader_reasoning_effort,
                            )
                        )

                    response_content = _limit_llm_response_content(
                        response_content,
                        use_audio=use_audio,
                        max_response_chars=max_response_chars,
                        max_audio_duration_s=max_audio_duration_s,
                    )

                    # 3c) Regardless, synthesize and transcribe the audio reply
                    if use_audio:
                        audio_reply = synthesize_audio_fn(response_content)
                        reply_transcript, _ = transcribe_audio_fn(audio_reply)

                    else:
                        audio_reply = None
                        reply_transcript = make_text_transcript(response_content)

                except (openai.BadRequestError, ValueError) as err:
                    logger.debug(err)
                finally:
                    generation_attempts += 1

            # If we have used all our generation attempts, give up.
            if (
                response_content is None
                or not reply_transcript
                or "words" not in reply_transcript
            ):
                with Session(engine) as session:
                    if sent_message:
                        session.delete(sent_message)

                    rd_orm = session.get(RoundORM, round_id)
                    rd_orm.processing_msg = False
                    session.add(rd_orm)
                    session.commit()
                raise ValueError("Could not generate response, speech, or transcript.")

            _, timed_out = round_over_state(
                engine,
                round_id,
                is_target=is_target,
                round_time_limit=round_time_limit,
            )
            if timed_out:
                _clear_processing_flag(engine, round_id)
                return (
                    sent_message_base,
                    None,
                    None,
                    None,
                    False,
                    True,
                )

            # sleep for either the duration of the audio message or the
            # duration of the fake transcript time (minus the time spent in this func)
            processing_duration_s: float = (
                datetime.now(timezone.utc) - processing_start
            ).total_seconds()
            delay_s: float = max(
                reply_transcript["duration"] - processing_duration_s, 0
            )
            assert delay_s < 60
            if delay_sleep:
                logger.info(f"Sleeping for {delay_s:.2f} seconds.")
                time.sleep(delay_s)

            received_message = SentMessage(
                is_target=other_is_target,
                audio=audio_reply,
                transcript=reply_transcript,
                message_content=response_content,
                thought_content=thought_content,
                reasoning_trace=reasoning_trace,
                flagged=False,
                round_id=round_id,
            )

            # Autofill continuous measures for LLM targets to mirror runner
            cond = rd.condition
            if (
                other_is_target
                and cond
                and cond.continuous_measure
                and (cond.llm_target_fill_serial is not False)
            ):
                persuader_text = sent_message.message_content if sent_message else ""
                if cond.continuous_measure == _CM.SERIAL_QUESTIONS:
                    if role.simulated_target and other_is_target:
                        trace = rd_orm.simulated_target_trace or {}
                        sim_target = SimulatedTarget(**trace)
                        belief = sim_target.get_belief_state(
                            len(sim_target.belief_history) - 1
                        )
                        received_message.last_serial_question = float(belief)
                    elif role.llm_target and other_is_target:
                        belief = _llm_target_self_report_belief(
                            rd,
                            role.llm_target,
                        )
                        if belief is not None:
                            received_message.last_serial_question = float(belief)
                    else:
                        next_serial = rd.compute_next_serial_question(persuader_text)
                        if next_serial is not None:
                            received_message.last_serial_question = float(next_serial)
                elif cond.continuous_measure == _CM.SERIAL_QUESTIONS_SENTENCE:
                    next_serial_sent = rd.compute_next_serial_question_sentence(
                        persuader_text
                    )
                    if next_serial_sent is not None:
                        received_message.last_serial_question_sentences = (
                            next_serial_sent
                        )

            received_message_base = SentMessageBase.model_validate(received_message)
            logger.debug("Received message: %s", received_message)

    logger.info("Updating database for round %s", round_id)

    our_turns_left: bool | int | None = None
    their_turns_left: bool | int | None = None
    target_can_end_round: bool
    round_over: bool | None = None

    with Session(engine) as session:
        statement = select(RoundORM).filter_by(id=round_id).with_for_update()
        rd_orm = session.exec(statement).one()

        if received_message is not None:
            assert not sent_message or not sent_message.flagged
            # Fail early if we introduce duplicate messages
            assert rd_orm.is_roles_turn(received_message.is_target)
            session.add(received_message)

        # The other player has chosen and they cannot be a persuader (we are the persuader)
        if other_is_target and llm_target_ends_game:
            rd_orm.target_ended_round = llm_target_ends_game

        rd_orm.processing_msg = False

        session.add(rd_orm)
        session.commit()
        session.refresh(rd_orm)

        rd = rd_orm.as_round()
        # Compute final turn-counts & end-of-round
        our_turns_left = rd.turns_left(is_target=is_target)
        their_turns_left = rd.turns_left(is_target=other_is_target)
        target_can_end_round = rd.target_can_end_round()
        round_over = rd.neither_turns_left()

    return (
        sent_message_base,
        received_message_base,
        our_turns_left,
        their_turns_left,
        target_can_end_round,
        round_over,
    )
