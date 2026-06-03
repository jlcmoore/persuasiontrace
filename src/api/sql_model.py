"""
src/api/sql_model.py

Author: Jared Moore
Date: July, 2025

Utilities to serialize objects into SQL.
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import ConfigDict, ValidationError
from sqlalchemy import String, func, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import JSON, Column, DateTime, Field, ForeignKey, Relationship, SQLModel
from sqlmodel._compat import SQLModelConfig

from experiment.condition import Condition, ContinuousMeasure, PropositionSource, Roles
from experiment.round import Round, RoundBase, RoundMixin
from experiment.utils import (
    normalize_message_highlight,
    normalize_serial_sentence_values,
)

SQLITE_FILE_NAME = "database.db"
SQLITE_URL_FMT = "sqlite:///{filename}"
SQLITE_URL = SQLITE_URL_FMT.format(filename=SQLITE_FILE_NAME)

CONNECT_ARGS = {"check_same_thread": False}

logger = logging.getLogger(__name__)


def _group_message_highlight_entries(
    payloads: list[list[dict[str, Any]] | None],
    *,
    message_roles: list[str],
    round_id: int | None,
) -> list[list[dict[str, Any]]]:
    """
    Normalize and bucket highlight payloads by persuader message index.
    """

    grouped: list[list[dict[str, Any]]] = [[] for _ in message_roles]
    seen_ranges: list[set[tuple[int, int]]] = [set() for _ in message_roles]

    for raw_entries in payloads:
        highlights = normalize_message_highlight(
            raw_entries,
            context="RoundORM.as_round",
            round_id=round_id,
        )
        if not highlights:
            continue

        for entry in highlights:
            idx = entry.get("message_index")
            if not isinstance(idx, int) or not 0 <= idx < len(message_roles):
                continue
            if message_roles[idx] != "persuader":
                continue

            start = entry.get("start")
            end = entry.get("end")
            try:
                start_idx = int(start)
                end_idx = int(end)
            except (TypeError, ValueError):
                continue
            if end_idx <= start_idx:
                continue

            key = (start_idx, end_idx)
            if key in seen_ranges[idx]:
                continue
            seen_ranges[idx].add(key)

            normalized_entry = dict(entry)
            normalized_entry["start"] = start_idx
            normalized_entry["end"] = end_idx
            grouped[idx].append(normalized_entry)

    return grouped


class Proposition(SQLModel, table=True):
    """
    A Sql class for storing propositions.
    """

    id: str = Field(primary_key=True)
    original_text: str | None = Field(default=None)
    factual_domain: bool = Field(default=True)
    proposition_is_correct: bool | None = Field(default=None)
    control_dialogue: bool = Field(default=False)
    participant_id: int | None = Field(default=None, foreign_key="participant.id")
    proposition_source: PropositionSource | None = Field(default=None)

    # Optional BN structure, used when condition.roles.simulated_target is True
    bayesian_network: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON(none_as_null=True))
    )

    # The below so that the class calls the validator
    model_config = SQLModelConfig(validate_assignment=True)

    # pylint: disable=arguments-differ
    def model_post_init(self, __context):
        if self.factual_domain and self.proposition_is_correct is None:
            raise ValueError(
                "You must pass a truth value for the prop when a factual domain."
            )

        return self


class RoundORM(RoundBase, RoundMixin, SQLModel, table=True):
    """
    A table to store all of the rounds -- the games human participants
    are playing or have played
    """

    # NB: we only make rounds once that are ready or are already playing
    id: int | None = Field(default=None, primary_key=True)

    persuader_id: int | None = Field(default=None, foreign_key="participant.id")
    target_id: int | None = Field(default=None, foreign_key="participant.id")

    # Whether the persuader is an LLM and its name
    llm_persuader: str | None = Field(default=None)
    llm_target: str | None = Field(default=None)
    simulated_target: str | None = Field(default=None)

    # NB: redefining proposition from RoundBase
    proposition: str = Field(
        sa_column=Column(
            String,
            ForeignKey("proposition.id"),
            nullable=False,
        )
    )

    proposition_obj: Proposition = Relationship(
        sa_relationship_kwargs={
            "lazy": "selectin",
            "foreign_keys": "[RoundORM.proposition]",
        },
    )

    proposition_during_round: str | None = Field(
        sa_column=Column(
            String,
            ForeignKey("proposition.id"),
            nullable=True,
        ),
        default=None,
    )

    # When neither of these is none, the round is complete
    target_initial_belief: float | None = Field(default=None)
    target_final_belief: float | None = Field(default=None)
    target_initial_node_beliefs: dict[str, float] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )
    target_final_node_beliefs: dict[str, float] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )

    persuader_supports_proposition: bool | None = Field(default=None)

    continuous_measure: ContinuousMeasure | None = Field(default=None)

    on_reflection: bool = Field(default=False)

    synthetic_audio: bool = Field(default=False)

    use_audio: bool = Field(default=False)

    show_transcript: bool = Field(default=False)

    turn_limit: int | None = Field(default=None)

    minimum_turns: int | None = Field(default=None)

    no_early_end: bool = Field(default=False)

    processing_msg: bool = Field(default=False)

    control_dialogue: bool = Field(default=False)

    participant_proposition: bool = Field(default=False)
    enable_node_belief_survey: bool = Field(default=False)

    simulated_target_trace: dict[str, Any] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )

    # TODO: make sure this works
    # make messages a real relationship so you can lazy-load it
    messages: list["SentMessage"] = Relationship()

    # If the continuous measure is on-reflection
    on_reflection_highlights: list[dict[str, Any]] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )

    # The below so that the class calls the validator
    model_config = SQLModelConfig(validate_assignment=True)

    # pylint: disable=arguments-differ
    def model_post_init(self, __context):

        # Generic "re-apply whatever was passed in" so that
        # no user-supplied field ever gets clobbered by the ORM
        # this is only necessary because of the oddities of SqlModel
        # and validation
        if __context:
            raw: dict = __context.get("data") or {}
            for field_name, field_value in raw.items():
                # only set it if we actually have that attribute
                if hasattr(self, field_name):
                    # bypass any validators or frozen-flags
                    object.__setattr__(self, field_name, field_value)

        if not self.persuader_id and not self.llm_persuader:
            raise ValueError("No persuader specified.")
        if not self.persuader_id and not self.target_id:
            raise ValueError("One of the participants must be human")
        if self.continuous_measure:
            self.continuous_measure = ContinuousMeasure(self.continuous_measure)

        # This should not error

        if self.proposition_obj:
            assert self.condition().roles

        return self

    def condition(self) -> Condition | None:
        """Returns the Condition for this Round."""
        if not self.proposition_obj:
            raise ValueError("No proposition object associated.")

        try:
            continuous_measure = (
                ContinuousMeasure(self.continuous_measure)
                if self.continuous_measure
                else None
            )

            condition = Condition(
                roles=Roles(
                    llm_persuader=self.llm_persuader,
                    llm_target=self.llm_target,
                    human_persuader=self.persuader_id is not None,
                    human_target=self.target_id is not None,
                    simulated_target=self.simulated_target,
                    # The persona is not explicitly saved as a column to avoid schema bloat.
                    # Analysts can determine it directly from
                    # `self.simulated_target_trace["susceptibilities"]`.
                    simulated_target_persona=None,
                ),
                participant_proposition=self.participant_proposition,
                on_reflection=self.on_reflection,
                factual_domain=self.proposition_obj.factual_domain,  # pylint: disable=no-member
                proposition_is_correct=self.proposition_obj.proposition_is_correct,  # pylint: disable=no-member
                continuous_measure=continuous_measure,
                synthetic_audio=self.synthetic_audio,
                use_audio=self.use_audio,
                control_dialogue=self.control_dialogue,
                show_transcript=self.show_transcript,
                turn_limit=self.turn_limit,
                minimum_turns=self.minimum_turns,
                no_early_end=self.no_early_end,
                proposition_source=self.proposition_obj.proposition_source,
                enable_node_belief_survey=self.enable_node_belief_survey,
                llm_target_use_bayes_structure=bool(
                    self.llm_target and self.proposition_obj.bayesian_network
                ),
            )
            return condition
        except ValidationError as exc:
            logger.error(
                "Could not validate round id=%s while reconstructing condition: %s",
                self.id,
                exc,
            )
            return None

    def is_roles_turn(self, is_target: bool) -> bool:
        """
        Returns true if there are no messages or if the last message does not match
        the role of the new turn.
        """
        messages = self.non_flagged_messages()
        return not messages or messages[-1].is_target != is_target

    def non_flagged_messages(self) -> list["SentMessage"]:
        """
        Returns a sorted list of all of the non flagged messages in the round.
        """
        # 1) Grab only non-flagged messages with content
        non_flagged: list[SentMessage] = [
            m for m in self.messages if not m.flagged and m.message_content
        ]

        # 2) Sort by timestamp and message id.
        # SQLite defaults can have second-level timestamp precision, so two
        # back-to-back messages (target then LLM) may share the same created_at.
        # In that case message id preserves insertion order.
        non_flagged.sort(
            key=lambda m: (
                m.created_at or datetime.min,
                m.id if isinstance(m.id, int) else -1,
            )
        )

        # 3) Assert strict alternation of roles
        for prev, curr in zip(non_flagged, non_flagged[1:]):
            if prev.is_target == curr.is_target:
                message = (
                    f"Round {self.id!r}: messages at "
                    f"{prev.created_at!r}/{curr.created_at!r} "
                    f"did not alternate (both is_target={curr.is_target})"
                )
                logger.error(message)

        return non_flagged

    def as_round(self) -> Round:
        """
        Returns this ORM object as a normal Round.
        """

        condition = self.condition()
        if condition is None:
            raise ValueError(f"Could not reconstruct condition for round id={self.id}.")

        non_flagged: list[SentMessage] = self.non_flagged_messages()

        # 4) Build up your parallel lists
        transcripts: list[dict] = []
        messages: list[dict] = []
        chains_of_thought: list[dict] = []
        reasoning_traces: list[dict] = []

        mouse_traces: list[list[dict[str, Any]]] | None = None
        serial_questions: list[float] | None = None
        serial_questions_sentence: list[list[float]] | None = None
        message_highlights: list[list[dict[str, Any]]] | None = None
        raw_highlight_payloads: list[list[dict[str, Any]] | None] = []
        message_roles: list[str] = []

        if condition.continuous_measure == ContinuousMeasure.MOUSE_TRACE:
            mouse_traces = []
        elif condition.continuous_measure == ContinuousMeasure.SERIAL_QUESTIONS:
            serial_questions = []
        elif (
            condition.continuous_measure == ContinuousMeasure.SERIAL_QUESTIONS_SENTENCE
        ):
            serial_questions_sentence = []
        for i, m in enumerate(non_flagged):
            transcripts.append(m.transcript)

            role = "target" if m.is_target else "persuader"

            if serial_questions is not None and m.last_serial_question is not None:
                assert m.is_target
                serial_questions.append(m.last_serial_question)
            if (
                serial_questions_sentence is not None
                and m.last_serial_question_sentences is not None
            ):
                assert m.is_target
                values = normalize_serial_sentence_values(
                    m.last_serial_question_sentences,
                    context="RoundORM.as_round",
                    round_id=self.id,
                )
                if values:
                    serial_questions_sentence.append(values)
            if (
                condition.continuous_measure == ContinuousMeasure.MESSAGE_HIGHLIGHTS
                and m.last_message_highlight
            ):
                assert m.is_target
                raw_highlight_payloads.append(m.last_message_highlight)

            if mouse_traces is not None and m.last_mouse_trace is not None:
                assert m.is_target
                mouse_traces.append(m.last_mouse_trace)

            if not m.message_content:
                # In the mouse trace we allow an ultimate message
                # that is a "dummy" meant just to give us the final measure
                assert i == len(non_flagged) - 1
                continue

            messages.append(
                {
                    "role": role,
                    "content": m.message_content,
                }
            )

            chains_of_thought.append(
                {
                    "role": role,
                    "content": m.thought_content,
                }
            )

            reasoning_traces.append(
                {
                    "role": role,
                    "content": m.reasoning_trace,
                }
            )
            message_roles.append(role)

        if condition.continuous_measure == ContinuousMeasure.MESSAGE_HIGHLIGHTS:
            message_highlights = _group_message_highlight_entries(
                raw_highlight_payloads,
                message_roles=message_roles,
                round_id=self.id,
            )

        # 5) Return the in-memory Round
        return Round(
            condition=condition,
            proposition=self.proposition,
            proposition_during_round=self.proposition_during_round,
            bayesian_network=self.proposition_obj.bayesian_network,
            persuader_id=(
                self.llm_persuader if self.llm_persuader else self.persuader_id
            ),
            target_id=self.llm_target if self.llm_target else self.target_id,
            human_persuader_id=self.persuader_id,
            human_target_id=self.target_id,
            target_initial_belief=self.target_initial_belief,
            target_final_belief=self.target_final_belief,
            target_initial_node_beliefs=self.target_initial_node_beliefs,
            target_final_node_beliefs=self.target_final_node_beliefs,
            transcripts=transcripts,
            messages=messages,
            chains_of_thought=chains_of_thought,
            reasoning_traces=reasoning_traces,
            persuader_supports_proposition=self.persuader_supports_proposition,
            target_ended_round=self.target_ended_round,
            timed_out=self.timed_out,
            serial_questions=serial_questions,
            serial_questions_sentence=serial_questions_sentence,
            message_highlights=message_highlights,
            mouse_traces=mouse_traces,
            on_reflection_highlights=self.on_reflection_highlights,
            simulated_target_trace=self.simulated_target_trace,
        )


class FlaggedResponse(str, Enum):
    """
    The responses to give to flagged messages.
    """

    SERVER_ERROR = "Server error. Please try again."
    NO_WORDS = "We could not detect any words in your your message."
    DISCONNECTED = "Error. Please send your message again."
    INAPPROPRIATE = "Your last message was flagged as inappropriate."


class SentMessageBase(SQLModel, table=False):
    """
    Base data for any sent message.
    """

    # The Base64 encoded audio
    audio: str | None = Field(default=None)

    # The participant's original audio (if `audio` is sythesized)
    original_audio: str | None = Field(default=None)

    # The dict of the transcript of the audio
    transcript: dict[str, Any] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )

    message_content: str | None = Field()

    thought_content: str | None = Field(default=None)

    reasoning_trace: str | None = Field(default=None)

    flagged: bool = Field()

    flagged_response: FlaggedResponse | None = Field(default=None)

    is_target: bool = Field()

    # If the message is the target and the continuous measure is serial questions
    # then this is the target's degree of belief after the last sent message
    last_serial_question: float | None = Field(default=None)

    # If the message is the target and the continuous measure is serial-questions-sentence
    # this stores the beliefs recorded after each sentence in the persuader's message
    last_serial_question_sentences: list[float] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )

    # If the message is the target and the continuous measure captures message highlights
    # store the highlighted snippets for the persuader message just read.
    last_message_highlight: list[dict[str, Any]] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )

    # If the message is the target and the continuous measure is mouse traces
    last_mouse_trace: list[dict[str, Any]] | None = Field(
        sa_column=Column(JSON(none_as_null=True)), default=None
    )

    model_config = ConfigDict(
        from_attributes=True,  # equivalent to old orm_mode=True
    )


class SentMessage(SentMessageBase, table=True):
    """
    A table to store all of the messages sent by the persuader or target in a round
    """

    id: int | None = Field(default=None, primary_key=True)

    round_id: int = Field(foreign_key="roundorm.id")

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )


def ensure_schema_compatibility(engine: Engine) -> None:
    """
    Ensure older SQLite databases have newly added columns.

    This performs lightweight, idempotent ALTER TABLE operations for any
    missing columns that were introduced after initial deployments. New
    columns are added as nullable with implicit NULL defaults to match the
    SQLModel field defaults.

    Currently handled columns (table -> columns):
    - sentmessage:
        - last_serial_question_sentences (JSON)
        - last_message_highlight (JSON)
        - last_mouse_trace (JSON)
        - reasoning_trace (TEXT)
        - original_audio (TEXT)
    - roundorm:
        - proposition_during_round (TEXT, NULL default)
        - on_reflection (BOOLEAN, DEFAULT 0)
        - timed_out (BOOLEAN, DEFAULT 0)
        - on_reflection_highlights (JSON, NULL default)
        - minimum_turns (INTEGER, NULL default)
        - no_early_end (BOOLEAN, DEFAULT 0)
        - control_dialogue (BOOLEAN, DEFAULT 0)
        - participant_proposition (BOOLEAN, DEFAULT 0)
    - proposition:
        - control_dialogue (BOOLEAN, DEFAULT 0)
        - original_text (TEXT, NULL default)
        - participant_id (INTEGER, NULL default)
    """

    # Use a transactional connection for DDL (compatible with SA 1.4/2.0)
    try:
        with engine.begin() as conn:
            insp = inspect(conn)
            tables = set(insp.get_table_names())

            # Generic helpers to avoid redundancy across tables
            existing: dict[str, set[str]] = {}

            def _exec(sql: str) -> None:
                # Use portable execution that works across SA versions
                conn.execute(text(sql))

            def _existing_cols(table: str) -> set[str]:
                if table in existing:
                    return existing[table]
                try:
                    cols = {c["name"] for c in insp.get_columns(table)}
                except SQLAlchemyError:
                    logger.exception("Failed to inspect columns for %r", table)
                    cols = set()
                existing[table] = cols
                return cols

            def _add_col(table: str, name: str, ddl_type: str) -> None:
                if table not in tables:
                    return
                cols = _existing_cols(table)
                if name in cols:
                    return
                try:
                    _exec(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}")
                    logger.info("Added missing column %s.%s", table, name)
                    cols.add(name)
                except SQLAlchemyError as err:
                    logger.warning(
                        "Failed to add column %s.%s of type %s: %s",
                        table,
                        name,
                        ddl_type,
                        err,
                    )

            # sentmessage columns (NULL defaults to match model)
            _add_col("sentmessage", "last_serial_question_sentences", "JSON")
            _add_col("sentmessage", "last_message_highlight", "JSON")
            _add_col("sentmessage", "last_mouse_trace", "JSON")
            _add_col("sentmessage", "reasoning_trace", "TEXT")
            _add_col("sentmessage", "original_audio", "TEXT")

            # roundorm columns
            _add_col("roundorm", "proposition_during_round", "TEXT")
            _add_col("roundorm", "on_reflection", "BOOLEAN NOT NULL DEFAULT 0")
            _add_col("roundorm", "timed_out", "BOOLEAN NOT NULL DEFAULT 0")
            _add_col("roundorm", "on_reflection_highlights", "JSON")
            _add_col("roundorm", "minimum_turns", "INTEGER")
            _add_col("roundorm", "no_early_end", "BOOLEAN NOT NULL DEFAULT 0")
            _add_col("roundorm", "control_dialogue", "BOOLEAN NOT NULL DEFAULT 0")
            _add_col(
                "roundorm", "participant_proposition", "BOOLEAN NOT NULL DEFAULT 0"
            )
            _add_col(
                "roundorm", "enable_node_belief_survey", "BOOLEAN NOT NULL DEFAULT 0"
            )
            _add_col("roundorm", "simulated_target_trace", "JSON")
            _add_col("roundorm", "target_initial_node_beliefs", "JSON")
            _add_col("roundorm", "target_final_node_beliefs", "JSON")

            # proposition columns
            _add_col("proposition", "control_dialogue", "BOOLEAN NOT NULL DEFAULT 0")
            _add_col("proposition", "original_text", "TEXT")
            _add_col("proposition", "participant_id", "INTEGER")
            _add_col("proposition", "bayesian_network", "JSON")
            _add_col("proposition", "proposition_source", "TEXT")

            if "roundorm" in tables and "proposition" in tables:
                _exec(
                    "UPDATE roundorm "
                    "SET participant_proposition = 1 "
                    "WHERE proposition IN ("
                    "SELECT id FROM proposition WHERE participant_id IS NOT NULL"
                    ")"
                )
    except SQLAlchemyError:
        logger.exception("Schema compatibility check failed")


class ExternalUser(SQLModel, table=True):
    """
    A table to store the ids of external users (e.g. from Prolific)
    mapping to our internal IDs.
    """

    # NB: It is a bit hacky here to use a uuid and store it as a string not as
    # a uuid but much of the code already uses a string. It is possible there is a
    # collision. I think that just means `participant_init` would fail and would have
    # to be called again, which is fine.
    id: int = Field(primary_key=True)
    external_id: str = Field()  # The Mturk or Prolific Id


class Participant(SQLModel, table=True):
    """A table to store information about our participants"""

    id: int = Field(primary_key=True)

    # Whether the particpant is always the 'target' or always the 'persuader'
    # or not yet initialized, None
    role: str | None = Field(default=None)

    # Whether the participant has waited too long in the waiting room
    # and should be forced to end the experiment early.
    # (This should only be set on human-human conditions.)
    waited_too_long: bool = Field(default=False)

    condition: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON(none_as_null=True))
    )

    ## The below fields we update over the course of the experiment

    entered_waiting_room: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )

    # Whether the participant has been approved, paid, and paid a bonus (if relevant)
    # None means the value is not set
    # False if we may want to deny some work (unlikely).
    work_approved: bool | None = Field(default=None)

    # Any feedback the participant gives at the end of the session
    feedback: str | None = Field(default=None)

    # NB: we have to set `use_alter` so that the SQL database knows which tables to create when
    current_round: int | None = Field(
        default=None,
        sa_column=Column(
            ForeignKey(
                "roundorm.id",
                use_alter=True,
            )
        ),
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )

    # The below so that the class calls the validator
    model_config = SQLModelConfig(validate_assignment=True)

    # pylint: disable=arguments-differ
    def model_post_init(self, __context: Any) -> "Participant":
        if (
            self.entered_waiting_room is not None
            and self.entered_waiting_room.tzinfo is None
        ):
            # assume UTC for naive timestamps
            self.entered_waiting_room = self.entered_waiting_room.replace(
                tzinfo=timezone.utc
            )
        return self

    def conditions_assigned(self):
        """Returns whether or not this particpant has been assigned to conditions yet"""
        return self.role is not None

    def waiting_time(self) -> timedelta | None:
        """Returns the time the participant has been waiting in the lobby"""
        if not self.entered_waiting_room:
            return None
        return datetime.now(timezone.utc) - (
            self.entered_waiting_room
            if self.entered_waiting_room.tzinfo  # pylint: disable=no-member
            else self.entered_waiting_room.replace(  # pylint: disable=no-member
                tzinfo=timezone.utc
            )
        )

    def is_target(self) -> bool:
        """Returns whether the participant is the target"""
        return self.role and self.role == "target"
