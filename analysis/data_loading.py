"""Shared round-loading helpers for analysis scripts."""

from __future__ import annotations

import itertools

import pandas as pd

from experiment import Condition, Round, load_round_results
from experiment.condition_filters import condition_matches_filters


def persuader_relative_name(name: str, persuader_relative: bool = True) -> str:
    """Return a persuader-relative column name when requested.

    Args:
        name: Base column name.
        persuader_relative: Whether to append the persuader-relative suffix.

    Returns:
        Column name.
    """
    if persuader_relative:
        name += "_persuader_relative"
    return name


def round_to_record(cond: Condition, rd: Round) -> dict:
    """Flatten one round/condition pair into a tidy row dictionary.

    Args:
        cond: Round condition metadata.
        rd: Parsed round object.

    Returns:
        Tidy row dictionary for analysis.
    """
    record: dict[str, object] = {
        "condition": str(cond.as_non_id_role()),
        "condition_obj": cond,
        "proposition": rd.proposition,
        "initial": rd.target_initial_belief,
        "final": rd.target_final_belief,
        "persuader_supports_proposition": rd.persuader_supports_proposition,
        persuader_relative_name("initial"): rd.persuader_relative_belief(
            rd.target_initial_belief
        ),
        persuader_relative_name("final"): rd.persuader_relative_belief(
            rd.target_final_belief
        ),
        "delta_dir": rd.target_belief_change(),
        "delta_raw": rd.target_final_belief - rd.target_initial_belief,
        "turns": len(rd.messages),
        "continuous": cond.continuous_measure,
    }

    if rd.serial_questions:
        record["serial"] = rd.get_serial_questions(persuader_relative=False)
        record[persuader_relative_name("serial")] = rd.get_serial_questions(
            persuader_relative=True
        )

    if rd.mouse_traces:
        record["mouse_trace"] = rd.get_mouse_traces(persuader_relative=False)
        record[persuader_relative_name("mouse_trace")] = rd.get_mouse_traces(
            persuader_relative=True
        )

    return record


def load_dataframe(
    min_date: str | None = None,
    filters: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Load round results into a dataframe, optionally filtered by condition.

    Args:
        min_date: Optional minimum date filter.
        filters: Optional condition filter mapping.

    Returns:
        Analysis dataframe.
    """
    condition_to_rounds = load_round_results(min_date)
    rows: list[dict] = []
    for condition, round_lists in condition_to_rounds.items():
        if filters and not condition_matches_filters(condition, filters):
            continue
        for round_obj in itertools.chain.from_iterable(round_lists):
            rows.append(round_to_record(condition, round_obj))

    df = pd.DataFrame(rows)
    if "serial" not in df.columns:
        df["serial"] = None
        df[persuader_relative_name("serial")] = None
    if "mouse_trace" not in df.columns:
        df["mouse_trace"] = None
        df[persuader_relative_name("mouse_trace")] = None
    return df
