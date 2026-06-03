"""
src/api/utils.py

Author: Jared Moore
Date: July, 2025

Contains utility functions and constants for the api.

"""

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any
from typing import Counter as TypeCounter
from typing import Tuple, Type

from pydantic import AliasChoices, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from sqlmodel import Session

from experiment.condition import (
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_MAX_MESSAGE_CHARS,
    Condition,
    Roles,
)
from experiment.utils import EXAMPLE_PROPOSITIONS_FILE, get_data_file_path

from .sql_model import RoundORM

DEFAULT_WAITING_ROOM_TIMEOUT = timedelta(seconds=60)

MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER = 5


def min_positive_timedelta_diff(td1: timedelta, td2: timedelta) -> timedelta:
    """
    Computes the differences between the two time deltas and returns the minimum
    """
    diff1 = abs(td1 - td2)
    diff2 = abs(td2 - td1)

    min_diff = min(diff1, diff2)

    return min_diff


def round_over_state(
    engine,
    round_id: int,
    *,
    is_target: bool,
    round_time_limit: int | None,
) -> tuple[int | bool, bool]:
    """
    Return remaining turns and whether the round has timed out.

    If a timeout is detected, update the round's timed_out flag in the DB.
    """
    with Session(engine) as session:
        rd_orm = session.get(RoundORM, round_id)
        turns_left = rd_orm.turns_left(is_target=is_target)
        timed_out = False
        if round_time_limit is not None:
            elapsed_time = datetime.now(timezone.utc) - rd_orm.created_at.replace(
                tzinfo=timezone.utc
            )
            timed_out = elapsed_time >= timedelta(seconds=round_time_limit)
            if timed_out and not rd_orm.timed_out:
                rd_orm.timed_out = True
                session.add(rd_orm)
                session.commit()
        return turns_left, timed_out


class ServerSettings(BaseSettings):
    """
    Variables potentially to change when running differently-styled experiments.
    """

    dev_environment: bool = True

    # The likelihood an llm target ends the game on each turn
    llm_target_end_game_prob: float = 0.5

    # Optional explicit reasoning effort for llm persuader calls.
    # When None, the provider/model default reasoning behavior is used.
    llm_persuader_reasoning_effort: str | None = None

    # If True, players can only play one kind of `Condition`
    enforce_player_condition: bool = True

    # If there are no participants in the waiting room and there are only paired experiments
    # left to run, whether to 'overstuff' the non-paired conditions
    overassign_non_paired_conditions: bool = True

    # How long should a participant wait until timing out of a round and
    # starting a new one.
    participant_conversation_timeout: timedelta = timedelta(minutes=3)

    # NB: The total number of participants is `condition_num_rounds.total()`
    # possibly with a few extra, as we overassign at times
    condition_num_rounds: TypeCounter[Condition] = Counter(
        {
            # Human - human condition -- 10 persuaders, 10 targets; n = 20
            # Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False): 10,
            # Human - llm condition -- 10 persuaders, 10 instances of gpt-4o as a target; n = 10
            # Coundition(roles=Roles(human_persuader=True, llm_target='gpt-4o')) : 10,
        }
    )

    # NB: This is used to initialize `condition_num_rounds` from a file
    conditions: list[dict[str, Any]] | None = None

    # The completion code for the Prolific study
    completion_code: str = "TEST"

    waiting_room_timeout: timedelta = DEFAULT_WAITING_ROOM_TIMEOUT

    # Whether to run any server tasks in the background
    background_tasks: bool = True

    # When running the server, loads in the settings from the file
    # TODO: verify that the path is correct for this file.
    model_config = SettingsConfigDict(yaml_file="configs/server_settings.yml")

    # The max number of rounds we let a participant play
    rounds_per_participant: int = 5

    # Whether *any* condition requires that messages should be sent as audio or raw text
    may_use_audio: bool = False

    # In seconds
    max_audio_seconds: int = Field(
        default=DEFAULT_MAX_AUDIO_SECONDS,
        validation_alias=AliasChoices("max_audio_seconds", "max_recording_duration"),
    )

    # The maximum number of characters allowed in a single message.
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS

    # The number of seconds to allow for each round before forcing it to end
    round_time_limit: int | None = None

    # The number of seconds to pause before recording the participant
    post_play_delay: int | None = 3

    # list of base names (without .jsonl)
    propositions_filenames: list[str] | None = None

    # Full paths (resolved from data package) corresponding to filenames
    propositions_full_filenames: list[str] = []

    # pylint: disable=arguments-differ
    def model_post_init(self, __context):
        # Initialize the condition counter
        if isinstance(self.conditions, list):
            self.condition_num_rounds = Counter()
            for condition_data in self.conditions:  # pylint: disable=not-an-iterable
                roles = Roles(
                    **condition_data["roles"],
                )
                condition_kwargs = dict(condition_data["condition"])
                if "max_message_chars" in condition_kwargs:
                    if condition_kwargs["max_message_chars"] != self.max_message_chars:
                        raise ValueError(
                            "Condition max_message_chars must match server max_message_chars."
                        )
                else:
                    condition_kwargs["max_message_chars"] = self.max_message_chars
                if "max_audio_seconds" in condition_kwargs:
                    if condition_kwargs["max_audio_seconds"] != self.max_audio_seconds:
                        raise ValueError(
                            "Condition max_audio_seconds must match server max_audio_seconds."
                        )
                else:
                    condition_kwargs["max_audio_seconds"] = self.max_audio_seconds
                condition = Condition(roles=roles, **condition_kwargs)
                self.condition_num_rounds[condition] += condition_data["count"]
                if condition.use_audio:
                    self.may_use_audio = True
        if self.llm_target_end_game_prob > 1 or self.llm_target_end_game_prob <= 0:
            raise ValueError("llm_target_end_game_prob must be [0, 1)")

        # Resolve proposition filenames (support str or list in settings)
        filenames: list[str]
        if (
            isinstance(self.propositions_filenames, list)
            and self.propositions_filenames
        ):
            filenames = list(self.propositions_filenames)
        else:
            # Fall back to default example file
            filenames = [EXAMPLE_PROPOSITIONS_FILE]

        # Convert to full paths and store list
        self.propositions_full_filenames = [
            get_data_file_path(f"{name}.jsonl") for name in filenames
        ]

    # NB: We redefine this method so we can use a yaml settings file
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # Allow programmatic initialization
            YamlConfigSettingsSource(settings_cls),  # Allow YAML configuration
        )
