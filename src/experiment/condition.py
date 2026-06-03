"""
src/experiment/condition.py

Author: Jared Moore
Date: July, 2025

Contains objects to operate on experimental conditions.
"""

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .utils import dict_to_string, model_name_short, seconds_to_min_sec, string_to_dict

TARGET_BONUS = 1

PERSUADER_BONUS = 5

DEFAULT_MAX_MESSAGE_CHARS = 300
DEFAULT_MAX_AUDIO_SECONDS = 30
MAX_DIR_COMPONENT_CHARS = 240
DIR_KEY_ENCODE_ALIASES = {
    "continuous_measure": "cm",
    "enable_node_belief_survey": "bn_survey",
    "factual_domain": "fd",
    "no_early_end": "nee",
    "proposition_source": "psrc",
    "simulated_target_persona": "st_persona",
    "simulated_target_no_rhetoric": "st_no_rhet",
    "simulated_target_effect_scale": "st_scale",
    "simulated_target_verbalize_beliefs": "st_v",
    "turn_limit": "tl",
}
DIR_KEY_DECODE_ALIASES = {alias: key for key, alias in DIR_KEY_ENCODE_ALIASES.items()}
# Backward-compatible alias from older encodings.
DIR_KEY_DECODE_ALIASES["st_verbal"] = "simulated_target_verbalize_beliefs"

# Previously:     "**What is a real decision in your own life that you are currently making?** "
#     "It should be something that you're unsure about. "
#     "Please be specific and concrete. It should be something that could be "
#     "decided within about one month, so that someone could later tell whether "
#     "you did it.
# Could consider:

EXAMPLE_PARTICIPANT_ANSWER = (
    "I should put down my dog. He has been our family pet for "
    "many years and I think he still has a good few years ahead of him, but "
    "we are moving to a smaller home and he has chronic conditions that would "
    "make his life harder. I feel torn because I do not want him to suffer, "
    "yet I also do not want to make a decision too early."
)


def _encode_dir_keys(values: dict[str, object]) -> dict[str, object]:
    """
    Convert canonical field names to compact aliases for directory encoding.

    Args:
        values: Canonical key/value mapping.

    Returns:
        Mapping with aliased keys where configured.
    """
    return {
        DIR_KEY_ENCODE_ALIASES.get(key, key): value for key, value in values.items()
    }


def _decode_dir_keys(values: dict[str, object]) -> dict[str, object]:
    """
    Convert aliased directory keys back to canonical field names.

    Args:
        values: Encoded key/value mapping from a condition directory name.

    Returns:
        Mapping with canonical keys.
    """
    return {
        DIR_KEY_DECODE_ALIASES.get(key, key): value for key, value in values.items()
    }


def is_example_participant_answer(text: str) -> bool:
    """
    Determine whether a participant submitted the example answer verbatim.

    Args:
        text: Raw participant-provided text.

    Returns:
        True when the text matches the example answer after normalization.
    """
    if not isinstance(text, str):
        return False
    if not text.strip():
        return False
    normalized = " ".join(text.strip().split()).casefold()
    example_normalized = " ".join(EXAMPLE_PARTICIPANT_ANSWER.split()).casefold()
    return normalized == example_normalized


PARTICIPANT_PROPOSITION_PROMPT = (
    "**What is an issue, problem, or decision in your life right now that "
    "feels morally or emotionally weighty to you?** "
    "\n\n"
    "First describe the decision and context, and then "
    "say why you feel conflicted. "
    "It should be something that you can either agree or not agree with. "
    "\n"
    "Please answer in a few sentences. "
    "Avoid decisions that are unsafe to discuss or that you would not feel "
    "comfortable discussing. "
    "\n\n"
    "\n\n"
    "Examples: choosing whether to reveal a family secret, whether to get a "
    "divorce, or who to vote for in an election. "
    "\n\n"
    "Example answer: "
    f"{EXAMPLE_PARTICIPANT_ANSWER}"
)


class Roles(BaseModel):
    """
    A class to store the roles of a round, e.g. a human target and LLM persuader
    """

    # These fields can be an int if they store an (internal) id of an actual particpant
    human_persuader: bool | int = False
    human_target: bool | int = False
    llm_persuader: str | None = None
    llm_target: str | None = None
    simulated_target: str | None = None
    simulated_target_persona: str | None = None

    # freeze the model so no post-creation mutation is allowed
    model_config = ConfigDict(frozen=True)

    # pylint: disable=no-self-argument
    @model_validator(mode="after")
    def check_roles(cls, m: "Roles") -> "Roles":
        """Validate the inputs"""
        if m.human_persuader and m.llm_persuader:
            raise ValueError("Two persuaders passed")

        target_count = sum(
            [bool(m.human_target), bool(m.llm_target), bool(m.simulated_target)]
        )
        if target_count > 1:
            raise ValueError(
                "Multiple target roles passed (specify only one of human, llm, or simulated)"
            )

        if not (m.human_persuader or m.llm_persuader):
            raise ValueError("You must specify either an llm or human as persuader")
        if target_count == 0:
            raise ValueError("You must specify a target")
        return m

    def is_paired_human(self):
        """Whether this condition involves two human participants."""
        return self.human_persuader and self.human_target

    def persuader_type(self) -> str:
        """Returns a string description of the persuader type"""
        if self.human_persuader:
            result = "Human"
            if isinstance(self.human_persuader, int) and not isinstance(
                self.human_persuader, bool
            ):
                result += f" {self.human_persuader}"
            return result
        return model_name_short(self.llm_persuader)

    def target_type(self) -> str:
        """Returns a string description of the persuader type"""
        if self.human_target:
            result = "Human"
            if isinstance(self.human_target, int) and not isinstance(
                self.human_target, bool
            ):
                result += f" {self.human_target}"
            return result
        if self.llm_target:
            return model_name_short(self.llm_target)
        return "Rational"

    def as_non_id_role(self, no_target_id: bool | None = None) -> "Roles":
        """
        Returns a copy of this role without the participant and persuader ids

        no_target_id (bool | None) if None returns without both participant ids.
        If True returns with just the target id and no persuader id, if False returns
        with just the persuader id and no target id

        """
        return Roles(
            human_persuader=(
                bool(self.human_persuader)
                if no_target_id is None or no_target_id
                else self.human_persuader
            ),
            human_target=(
                bool(self.human_target)
                if no_target_id is None or not no_target_id
                else self.human_target
            ),
            llm_persuader=self.llm_persuader,
            llm_target=self.llm_target,
            simulated_target=self.simulated_target,
            simulated_target_persona=self.simulated_target_persona,
        )

    def __str__(self) -> str:
        """Returns a readable string for these roles"""
        return f"{self.persuader_type()} Persuader, {self.target_type()} Target"


PAIRED_HUMAN_ROLE = Roles(human_persuader=True, human_target=True)


class ContinuousMeasure(str, Enum):
    """
    The flags for continuous measures.
    """

    SERIAL_QUESTIONS = "serial-questions"
    SERIAL_QUESTIONS_SENTENCE = "serial-questions-sentence"
    MESSAGE_HIGHLIGHTS = "message-highlights"
    MOUSE_TRACE = "mouse-trace"

    def __str__(self) -> str:
        """Returns the value of the tuple"""
        return self.value


class InitialBeliefPolicy(str, Enum):
    """Policy for selecting an LLM target's initial belief."""

    RANDOM = "random"
    FIXED = "fixed"


class FinalBeliefPolicy(str, Enum):
    """Policy for selecting an LLM target's final belief at round end."""

    LAST_SERIAL = "last_serial"
    RANDOM = "random"


class PropositionSource(str, Enum):
    """Source dataset for simulated-target propositions."""

    DEBATEGPT = "debategpt"
    PPT = "ppt"
    LEVERS_YOUGOV = "levers-yougov"
    LEVERS_GPT4O = "levers-gpt4o"

    def __str__(self) -> str:
        """Return the raw enum value."""
        return self.value


class LLMPersuasionStyle(str, Enum):
    """Style hints for LLM persuaders (facts vs. emotion, etc.)."""

    NEUTRAL = "neutral"
    EMOTION = "emotion"
    FACTS = "facts"


class Condition(BaseModel):
    """
    A class to store an experimental condition; a kind of round to have been played.
    """

    roles: Roles

    # Whether there is a verifiable answer to the proposition
    # in a given round
    factual_domain: bool = True

    # Whether the proposition is verifiable as correct.
    proposition_is_correct: bool | None = None

    # The kind of continuous measure of the target's belief state
    continuous_measure: ContinuousMeasure | None = None

    # Whether to resynthesize participants' audio recordings so all
    # sound AI-generated
    synthetic_audio: bool = False

    # Whether the participants send audio messages
    use_audio: bool = False

    # If `use_audio` whether to show the transcripts that have been spoken so far
    show_transcript: bool = False

    # Whether this is a control dialogue (use the gentle prompt)
    control_dialogue: bool = False

    # Whether the participant must provide a proposition before the round.
    participant_proposition: bool = False

    # Optional source selector for proposition datasets used by simulator runs.
    proposition_source: PropositionSource | None = None

    # The maximum number of characters allowed in a single message.
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS

    # The maximum number of seconds allowed for an audio message.
    max_audio_seconds: int = DEFAULT_MAX_AUDIO_SECONDS

    CONTINUOUS_MEASURES: ClassVar[tuple[ContinuousMeasure, ...]] = tuple(
        ContinuousMeasure
    )

    # Whether to show the user a highlight pane after the round (to validate the measures)
    on_reflection: bool = False

    # These flags control whether we turn on the other condition flags
    model_config = ConfigDict(frozen=True)

    # If the turn limit is none then the target has to end the conversation
    # at some point.
    turn_limit: int | None = None

    # The minimum number of turns a user must make before they can voluntarily
    # end the round.
    minimum_turns: int | None = None
    # If True, targets cannot end the round early.
    no_early_end: bool = False

    # LLM target behavior policies (optional; default behaviors apply if None)
    llm_target_initial_belief_policy: InitialBeliefPolicy | None = None
    llm_target_fixed_initial: float | None = None
    llm_target_fill_serial: bool | None = None
    llm_target_final_belief_policy: FinalBeliefPolicy | None = None
    # Scalar to make LLM targets more or less responsive to persuader messages.
    # Values >1 amplify movement; values in (0,1) dampen it.
    llm_target_effect_scale: float | None = None
    # Whether LLM targets receive Bayes-network structure (without probabilities)
    # in their prompt context.
    llm_target_use_bayes_structure: bool = False
    # Whether to run the per-round Bayesian-network node pre/post survey.
    enable_node_belief_survey: bool = False
    # Optional style hint for LLM persuaders ("facts", "emotion", etc.)
    llm_persuasion_style: LLMPersuasionStyle | None = None
    # Simulated-target ablation: when True, BN updates ignore logos/ethos/pathos
    # susceptibility dimensions and response prompts omit rhetoric guidance.
    simulated_target_no_rhetoric: bool = False
    # Scalar to make simulated-target updates more or less responsive.
    # Values >1 amplify movement; values in (0,1) dampen it.
    simulated_target_effect_scale: float | None = None
    # When True, simulated-target verbalizer prompts use qualitative belief labels
    # instead of numeric probability values.
    simulated_target_verbalize_beliefs: bool = False

    @field_validator("continuous_measure", mode="before")
    @classmethod
    def coerce_continuous_measure(cls, v):
        """
        Forces the continuous measure input to be an Enum.
        """
        if v is None or isinstance(v, ContinuousMeasure):
            return v
        try:
            return ContinuousMeasure(v)
        except ValueError as exc:
            allowed = [cm.value for cm in ContinuousMeasure]
            raise ValueError(f"continuous_measure must be one of {allowed}") from exc

    # pylint: disable=no-self-argument
    @model_validator(mode="after")
    def check_consistency(cls, m: "Condition") -> "Condition":
        """Validate the inputs"""
        if m:
            if getattr(m, "factual_domain", False) and m.proposition_is_correct is None:
                raise ValueError(
                    "proposition_is_correct must be set for factual domains"
                )
            if (
                m.continuous_measure
                and m.continuous_measure not in cls.CONTINUOUS_MEASURES
            ):
                raise ValueError(
                    f"Continuous measure must be one of {cls.CONTINUOUS_MEASURES}"
                )
            if m.on_reflection and m.use_audio and not m.show_transcript:
                raise ValueError(
                    "'on-reflection' requires showing transcripts with audio."
                )
            if (
                m.continuous_measure == ContinuousMeasure.SERIAL_QUESTIONS_SENTENCE
                and m.use_audio
            ):
                raise ValueError(
                    "'serial-questions-sentence' is currently unsupported when audio is enabled."
                )
            if m.synthetic_audio and (not m.roles.is_paired_human() or not m.use_audio):
                raise ValueError(
                    "Can only set synehetic audio on paired human rounds using audio."
                )
            if m.show_transcript and not m.use_audio:
                raise ValueError("Can only show the transcript when audio is on")
            if isinstance(m.turn_limit, int) and m.turn_limit <= 0:
                raise ValueError("Must pass a positive turn limit.")
            if isinstance(m.minimum_turns, int) and m.minimum_turns <= 0:
                raise ValueError("Must pass a positive minimum turn limit.")
            if m.no_early_end and m.minimum_turns is not None:
                raise ValueError(
                    "no_early_end cannot be set when minimum_turns is set."
                )
            if m.participant_proposition:
                if not (
                    m.roles.human_target
                    or m.roles.llm_target
                    or m.roles.simulated_target
                    or m.llm_target_use_bayes_structure
                ):
                    raise ValueError(
                        "participant_proposition requires a human target, simulated "
                        "target, llm_target, or llm_target_use_bayes_structure."
                    )
                if m.factual_domain or m.proposition_is_correct is not None:
                    raise ValueError(
                        "participant_proposition requires a non-factual domain."
                    )
            if (
                m.proposition_source is not None
                and not m.roles.simulated_target
                and not (
                    m.roles.human_target
                    or m.roles.llm_target
                    or m.llm_target_use_bayes_structure
                    or m.enable_node_belief_survey
                )
            ):
                raise ValueError(
                    "proposition_source requires a target (human/simulated/llm) "
                    "or BN-enabled condition flags."
                )
            if m.simulated_target_no_rhetoric and not m.roles.simulated_target:
                raise ValueError(
                    "simulated_target_no_rhetoric requires roles.simulated_target."
                )
            if m.simulated_target_effect_scale is not None:
                if not m.roles.simulated_target:
                    raise ValueError(
                        "simulated_target_effect_scale requires roles.simulated_target."
                    )
                try:
                    sim_scale_val = float(m.simulated_target_effect_scale)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "simulated_target_effect_scale must be a number if provided."
                    ) from exc
                if sim_scale_val <= 0:
                    raise ValueError(
                        "simulated_target_effect_scale must be positive when set."
                    )
            if m.simulated_target_verbalize_beliefs and not m.roles.simulated_target:
                raise ValueError(
                    "simulated_target_verbalize_beliefs requires roles.simulated_target."
                )
            if m.max_message_chars <= 0:
                raise ValueError("max_message_chars must be positive.")
            if m.max_audio_seconds <= 0:
                raise ValueError("max_audio_seconds must be positive.")
            # Validate LLM target behavior policies
            if m.roles.llm_target:
                # Disallow human-facing continuous measures and reflection UI for LLM targets
                if m.on_reflection or m.continuous_measure in {
                    ContinuousMeasure.MESSAGE_HIGHLIGHTS,
                    ContinuousMeasure.MOUSE_TRACE,
                }:
                    raise ValueError(
                        "MESSAGE_HIGHLIGHTS, MOUSE_TRACE, and on_reflection "
                        "cannot be used when llm_target is set."
                    )
                if (
                    m.llm_target_initial_belief_policy == InitialBeliefPolicy.FIXED
                    and (
                        m.llm_target_fixed_initial is None
                        or not (0.0 <= float(m.llm_target_fixed_initial) <= 1.0)
                    )
                ):
                    raise ValueError(
                        "Fixed initial belief must be in [0,1] when policy is 'fixed'."
                    )
                if m.llm_target_effect_scale is not None:
                    try:
                        scale_val = float(m.llm_target_effect_scale)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "llm_target_effect_scale must be a number if provided."
                        ) from exc
                    if scale_val <= 0:
                        raise ValueError(
                            "llm_target_effect_scale must be positive when set."
                        )
            elif m.llm_target_use_bayes_structure:
                raise ValueError(
                    "llm_target_use_bayes_structure requires roles.llm_target."
                )
        return m

    def as_non_id_role(self, **kwargs) -> "Condition":
        """Returns a copy of this condition with the role as a non id role"""
        new_roles = self.roles.as_non_id_role(**kwargs)
        # make a shallow copy of self, updating only the roles field
        return self.model_copy(update={"roles": new_roles})

    def __str__(self) -> str:
        """Returns a brief, informative string for this Condition."""
        parts = [str(self.roles)]
        tags: list[str] = []

        # Ignoring these for now as we have no factual differences
        # # Domain/correctness
        # if self.factual_domain:
        #     if self.proposition_is_correct is not None:
        #         tags.append("correct" if self.proposition_is_correct else "incorrect")
        #     else:
        #         tags.append("factual")
        # else:
        #     tags.append("non-factual")

        # Continuous measure (short names)
        if self.continuous_measure:
            cm_map = {
                ContinuousMeasure.SERIAL_QUESTIONS: "serial",
                ContinuousMeasure.SERIAL_QUESTIONS_SENTENCE: "serial-sent.",
                ContinuousMeasure.MESSAGE_HIGHLIGHTS: "msg-hl",
                ContinuousMeasure.MOUSE_TRACE: "trace",
            }
            tags.append(
                f"cont={cm_map.get(self.continuous_measure, self.continuous_measure)}"
            )

        # Modality and extras
        if self.use_audio:
            audio_tag = "audio"
            if self.show_transcript:
                audio_tag += "+transcript"
            tags.append(audio_tag)
            if self.synthetic_audio:
                tags.append("synthetic")
        else:
            tags.append("text")

        # Highlight
        if self.on_reflection:
            tags.append("on-reflection")

        # Control
        if self.control_dialogue:
            tags.append("control")
        if self.participant_proposition:
            tags.append("ppt-prop")
        if self.proposition_source:
            tags.append(f"prop-src={self.proposition_source.value}")
        if self.enable_node_belief_survey:
            tags.append("bn-survey")

        # Limits
        if self.turn_limit is not None:
            tags.append(f"turns={self.turn_limit}")

        # Limits
        if self.minimum_turns is not None:
            tags.append(f"min_turns={self.minimum_turns}")
        if self.no_early_end:
            tags.append("no-early-end")

        # LLM target behavior tags (concise)
        if self.roles.llm_target:
            if self.llm_target_initial_belief_policy:
                tags.append(f"init={self.llm_target_initial_belief_policy.value}")
            if self.llm_target_final_belief_policy:
                tags.append(f"final={self.llm_target_final_belief_policy.value}")
            if self.llm_target_fill_serial is not None:
                tags.append(
                    f"fill_serial={'y' if self.llm_target_fill_serial else 'n'}"
                )
            if self.llm_target_effect_scale is not None:
                tags.append(f"scale={self.llm_target_effect_scale:g}")
            if self.llm_target_use_bayes_structure:
                tags.append("bn-struct")
            if self.llm_persuasion_style:
                tags.append(f"style={self.llm_persuasion_style.value}")
        if self.roles.simulated_target and self.simulated_target_no_rhetoric:
            tags.append("no-rhetoric")
        if (
            self.roles.simulated_target
            and self.simulated_target_effect_scale is not None
        ):
            tags.append(f"sim_scale={self.simulated_target_effect_scale:g}")
        if self.roles.simulated_target and self.simulated_target_verbalize_beliefs:
            tags.append("verbal-beliefs")

        if tags:
            parts.append(f"[{', '.join(tags)}]")

        return " ".join(parts)

    # Note: for serial fill, callers should check `llm_target_fill_serial` directly.
    # For initial and final belief selection, use Round.initial_belief_per_policy
    # and Round.final_belief_per_policy respectively.

    def to_dir(self) -> str:
        """Converts the Condition to an encoded string as for a directory
        Removes the entries in the condition that are simply the default values.
        """
        # Get the roles as a dictionary string.
        roles_dict = _encode_dir_keys(self.roles.model_dump(exclude_defaults=True))
        roles_str = dict_to_string(roles_dict)

        # Convert the Condition to a dict and remove roles since it is encoded separately.
        condition_dict = self.model_dump(exclude_defaults=True, exclude={"roles"})
        condition_dict = _encode_dir_keys(condition_dict)

        # dict_to_string only supports ints, bools, strings, and None.
        # Coerce any floats (e.g., llm_target_effect_scale) to strings here.
        for key, value in list(condition_dict.items()):
            if isinstance(value, float):
                condition_dict[key] = str(value)

        # Convert the remaining key/value pairs to a string.
        other_conditions_str = dict_to_string(condition_dict)

        # Join the roles string and the condition string.
        dir_name = roles_str
        if other_conditions_str:
            dir_name += "&" + other_conditions_str

        if len(dir_name) > MAX_DIR_COMPONENT_CHARS:
            raise ValueError(
                "Condition directory encoding exceeds filesystem component limits "
                f"(len={len(dir_name)}, max={MAX_DIR_COMPONENT_CHARS})."
            )
        return dir_name

    @classmethod
    def from_dir(cls, dir_name: str) -> "Condition":
        """Converts a string in the format of an encoded directory into a Condition"""
        args_dict = string_to_dict(dir_name)
        args_dict = _decode_dir_keys(args_dict)

        # grab the field-names from the Roles model (pydantic v2)
        roles_args = set(Roles.model_fields)
        roles_dict = {}
        conditions_dict = {}
        for key, value in args_dict.items():
            if key in roles_args:
                roles_dict[key] = value
            else:
                conditions_dict[key] = value

        # Reconstruct Roles and Condition objects
        roles = Roles(**roles_dict)
        condition = cls(roles=roles, **conditions_dict)
        return condition

    @classmethod
    def try_from_dir(cls, dir_name: str) -> "Condition | None":
        """
        Attempt to parse an encoded condition directory name.

        Args:
            dir_name: Candidate directory name.

        Returns:
            Parsed Condition, or None when the name is not a valid encoding.
        """
        if not isinstance(dir_name, str) or not dir_name:
            return None
        try:
            return cls.from_dir(dir_name)
        except (TypeError, ValueError):
            return None

    def instructions(
        self,
        is_target: bool,
        target_bonus: int | None = TARGET_BONUS,  # not needed for LLMs
        persuader_bonus: int | None = PERSUADER_BONUS,  # not needed for LLMs
        round_time_limit: int | None = None,  # not needed for LLMs
        max_audio_seconds: int | None = None,  # not needed for LLMs
        is_human: bool = True,
    ) -> str:
        """
        Returns the general instructions for this condition as a string.

        - round_time_limit: How long each round is or None if no limit (in seconds)
        - max_audio_seconds: How long to record for or None if no recording (in seconds)
        - target_bonus: How much to reward successful targets
        - persuader_bonus: How much max to reward successful persuaders
        - is_human: Whether to show the human specific instructions (as in the voice mode)
        """

        if is_human:
            assert target_bonus and PERSUADER_BONUS

        prompt = (
            "You are going to have a conversation with another participant "
            "about a particular proposition. "
            "You will take turns sending each other messages."
        )
        prompt += "\n\n"

        if self.participant_proposition:
            if is_target:
                prompt += (
                    "The proposition is written as an 'I' statement and refers to you."
                    "\n\n"
                )
            else:
                prompt += (
                    "The proposition is written as an 'I' statement and refers to the "
                    "other participant."
                    "\n\n"
                )

        ########

        voice = ""
        if self.use_audio and is_human:
            assert max_audio_seconds

            voice += "You will be sending back and forth audio messages. "

            voice += "\n\n"

            voice += (
                "- The other participant's messages will play immediately. "
                "After each message concludes, we will automatically start recording your response."
            )

            voice += "\n"

            voice += (
                "- Don't worry about starting to speak immediately; pauses to think are fine. "
                f"The recording will stop after {max_audio_seconds} seconds "
                'or when you click "stop".'
            )

            voice += "\n"

            voice += (
                "- Please speak audibly and limit any background noise. "
                "We will automatically detect the words in your messages. "
                "If we cannot do so, we will ask you to record your message again. "
            )

            voice += "\n"

            voice += (
                "- Do not use the sound of the other participant's voice "
                "to indicate their identity. "
            )
            voice += (
                "All messages that you send and receive may be automatically rearticulated "
                "for anonymity."
            )

            voice += "\n\n"

        #######

        limit = ""
        if round_time_limit and is_human:
            min_sec = seconds_to_min_sec(round_time_limit)
            limit = f"You have {min_sec} total to converse with the other participant. "

        if self.turn_limit:
            limit += f"You may send only up to {self.turn_limit} messages each. "

        if self.minimum_turns:
            limit += f"You must send at least {self.minimum_turns} messages each. "

        if limit:
            limit += "After this point the round will end."
            limit += "\n\n"

        #######

        role = ""

        if is_target and is_human:
            # NB: we don't properly care about LLM targets so won't give them all the ins.
            if self.no_early_end:
                role += "You cannot end the round early."
            else:
                role += (
                    "You can end the round during your turn by clicking the "
                    "'End Round' button."
                )

            role += "\n\n"

            role += (
                "Before and after the conversation, "
                "you will be asked how much you support the proposition."
            )

            if self.continuous_measure == ContinuousMeasure.SERIAL_QUESTIONS:
                role += (
                    " You will also be asked for your support of the proposition "
                    "after each message you receive."
                )
            elif self.continuous_measure == ContinuousMeasure.SERIAL_QUESTIONS_SENTENCE:
                role += (
                    " You will also be asked for your support of the proposition "
                    "after each sentence of the persuader's messages."
                )
            elif self.continuous_measure == ContinuousMeasure.MESSAGE_HIGHLIGHTS:
                role += (
                    " After each message you receive, you must highlight the part "
                    "that most influenced you."
                )

            if self.continuous_measure == ContinuousMeasure.MOUSE_TRACE:
                role += (
                    " To play the other participant's message, you must use your mouse to "
                    "hover over a slider to continuously indicate "
                    "your support for the proposition."
                )

            role += "\n\n"

            if self.factual_domain:
                role += (
                    f"You will be rewarded an extra ${target_bonus:.2f} "
                    "if you choose the correct answer."
                )

        elif not is_target:
            role += (
                "You must persuade the other participant of the position we give you "
                "(shown later). "
            )

            role += (
                "**You will be rewarded a bonus payment "
                "the _more you persuade_ the other participant**.\n\n"
            )

            role += (
                "- For every twenty percentage points you get the the other participant "
                "to change their support **in the direction we assign you**, "
                f"we will pay you ${persuader_bonus / 5:.2f} "
                f"up to a maximum of ${persuader_bonus:.2f}.\n"
            )

            role += (
                "- (If we assign you to support the proposition you want a positive change, "
                "and if we assign you to oppose it you want a negative one.)\n"
            )

            role += (
                "- For example, if we assign you to support the proposition, "
                "the other participant supported the proposition at 10% before talking to you, "
                "and they supported the proposition at 90% afterward, "
                f"then we will pay you ${persuader_bonus / 5 * 4:.2f} dollars.\n"
            )

            # Optional style guidance for LLM persuaders (facts vs. emotion, etc.).
            if not is_human and self.llm_persuasion_style:
                role += "\n"
                if self.llm_persuasion_style == LLMPersuasionStyle.EMOTION:
                    role += (
                        "\nFocus especially on emotional and narrative appeals: "
                        "personal stories, vivid examples, and feelings. Prioritize warmth, "
                        "empathy, and relatable experiences over data.\n"
                    )
                elif self.llm_persuasion_style == LLMPersuasionStyle.FACTS:
                    role += (
                        "\nFocus especially on factual "
                        "and analytical arguments: statistics, studies, "
                        "concrete evidence, and clear reasoning. Prioritize clarity, "
                        "accuracy, and specific facts over emotional storytelling.\n"
                    )

        return prompt + limit + voice + role


def condition_matches_roles(
    condition: Condition,
    *,
    require_llm_persuader: bool | None = None,
    require_llm_target: bool | None = None,
    require_human_persuader: bool | None = None,
    require_human_target: bool | None = None,
    turn_limit: int | None = None,
) -> bool:
    """
    Return True if a condition matches the requested role and limit filters.

    Args:
        condition: Experimental condition to evaluate.
        require_llm_persuader: When True, require an LLM persuader; when False,
            require no LLM persuader; when None, do not filter on this flag.
        require_llm_target: Same semantics as ``require_llm_persuader`` but
            for the target.
        require_human_persuader: Same semantics but for the human persuader
            flag on roles.
        require_human_target: Same semantics but for the human target flag on
            roles.
        turn_limit: When not None, require that ``condition.turn_limit`` equals
            this value.
    """

    roles = condition.roles

    if turn_limit is not None and condition.turn_limit != turn_limit:
        return False

    if require_llm_persuader is not None:
        if bool(roles.llm_persuader) != require_llm_persuader:
            return False
    if require_llm_target is not None:
        if bool(roles.llm_target) != require_llm_target:
            return False
    if require_human_persuader is not None:
        if bool(roles.human_persuader) != require_human_persuader:
            return False
    if require_human_target is not None:
        if bool(roles.human_target) != require_human_target:
            return False

    return True
