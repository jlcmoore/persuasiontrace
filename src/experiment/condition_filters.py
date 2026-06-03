"""
src/experiment/condition_filters.py

Helpers for filtering Conditions from CLI inputs.
"""

from __future__ import annotations

import argparse
from typing import Any

from .condition import Condition

NONE_SENTINEL = object()

FilterSpec = dict[str, Any]


def parse_optional_bool(value: str) -> bool | object:
    """
    Parse a boolean CLI value, supporting "none"/"null" to filter for nulls.
    """
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return NONE_SENTINEL
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_optional_int(value: str) -> int | object:
    """
    Parse an integer CLI value, supporting "none"/"null" to filter for nulls.
    """
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return NONE_SENTINEL
    try:
        return int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}") from exc


def parse_optional_float(value: str) -> float | object:
    """
    Parse a float CLI value, supporting "none"/"null" to filter for nulls.
    """
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return NONE_SENTINEL
    try:
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value!r}") from exc


def parse_optional_str(value: str) -> str | object:
    """
    Parse a string CLI value, supporting "none"/"null" to filter for nulls.
    """
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return NONE_SENTINEL
    return value


FILTER_SPECS: list[FilterSpec] = [
    {
        "name": "human_persuader",
        "parser": parse_optional_bool,
        "getter": lambda condition: bool(condition.roles.human_persuader),
    },
    {
        "name": "human_target",
        "parser": parse_optional_bool,
        "getter": lambda condition: bool(condition.roles.human_target),
    },
    {
        "name": "llm_persuader",
        "parser": parse_optional_str,
        "getter": lambda condition: condition.roles.llm_persuader,
    },
    {
        "name": "llm_target",
        "parser": parse_optional_str,
        "getter": lambda condition: condition.roles.llm_target,
    },
    {
        "name": "factual_domain",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.factual_domain,
    },
    {
        "name": "proposition_is_correct",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.proposition_is_correct,
    },
    {
        "name": "continuous_measure",
        "parser": parse_optional_str,
        "getter": lambda condition: (
            condition.continuous_measure.value if condition.continuous_measure else None
        ),
    },
    {
        "name": "synthetic_audio",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.synthetic_audio,
    },
    {
        "name": "use_audio",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.use_audio,
    },
    {
        "name": "show_transcript",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.show_transcript,
    },
    {
        "name": "control_dialogue",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.control_dialogue,
    },
    {
        "name": "participant_proposition",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.participant_proposition,
    },
    {
        "name": "enable_node_belief_survey",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.enable_node_belief_survey,
    },
    {
        "name": "proposition_source",
        "parser": parse_optional_str,
        "getter": lambda condition: (
            condition.proposition_source.value if condition.proposition_source else None
        ),
    },
    {
        "name": "on_reflection",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.on_reflection,
    },
    {
        "name": "turn_limit",
        "parser": parse_optional_int,
        "getter": lambda condition: condition.turn_limit,
    },
    {
        "name": "minimum_turns",
        "parser": parse_optional_int,
        "getter": lambda condition: condition.minimum_turns,
    },
    {
        "name": "no_early_end",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.no_early_end,
    },
    {
        "name": "llm_target_initial_belief_policy",
        "parser": parse_optional_str,
        "getter": lambda condition: (
            condition.llm_target_initial_belief_policy.value
            if condition.llm_target_initial_belief_policy
            else None
        ),
    },
    {
        "name": "llm_target_fixed_initial",
        "parser": parse_optional_float,
        "getter": lambda condition: condition.llm_target_fixed_initial,
    },
    {
        "name": "llm_target_fill_serial",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.llm_target_fill_serial,
    },
    {
        "name": "llm_target_final_belief_policy",
        "parser": parse_optional_str,
        "getter": lambda condition: (
            condition.llm_target_final_belief_policy.value
            if condition.llm_target_final_belief_policy
            else None
        ),
    },
    {
        "name": "llm_target_effect_scale",
        "parser": parse_optional_float,
        "getter": lambda condition: condition.llm_target_effect_scale,
    },
    {
        "name": "llm_persuasion_style",
        "parser": parse_optional_str,
        "getter": lambda condition: (
            condition.llm_persuasion_style.value
            if condition.llm_persuasion_style
            else None
        ),
    },
    {
        "name": "simulated_target_effect_scale",
        "parser": parse_optional_float,
        "getter": lambda condition: condition.simulated_target_effect_scale,
    },
    {
        "name": "simulated_target_verbalize_beliefs",
        "parser": parse_optional_bool,
        "getter": lambda condition: condition.simulated_target_verbalize_beliefs,
    },
]

FILTER_KEYS = {spec["name"] for spec in FILTER_SPECS}


def _matches_optional(actual: object, desired: object | None) -> bool:
    """
    Return True if the actual value matches the desired filter.
    """
    if desired is None:
        return True
    if desired is NONE_SENTINEL:
        return actual is None
    return actual == desired


def condition_matches_filters(condition: Condition, filters: dict[str, object]) -> bool:
    """
    Return True if a condition matches all provided filter values.
    """
    for filter_spec in FILTER_SPECS:
        name = filter_spec["name"]
        desired = filters.get(name)
        actual = filter_spec["getter"](condition)
        if not _matches_optional(actual, desired):
            return False

    return True


def add_condition_filter_args(parser: argparse.ArgumentParser) -> None:
    """
    Attach condition filter CLI args using the Condition schema.
    """
    for filter_spec in FILTER_SPECS:
        flag = f"--{filter_spec['name'].replace('_', '-')}"
        parser.add_argument(flag, type=filter_spec["parser"])


def filters_from_args(args: argparse.Namespace) -> dict[str, object]:
    """
    Build a filter dict from parsed args.
    """
    args_dict = vars(args)
    return {key: args_dict.get(key) for key in FILTER_KEYS}
