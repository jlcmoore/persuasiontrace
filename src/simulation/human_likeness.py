"""Shared utilities for simulator-vs-human trajectory analysis."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from experiment import Condition, ContinuousMeasure
from experiment.round import Round
from simulation.human_trajectory_clusters import extract_round_updates


@dataclass(frozen=True)
class RoundTrajectory:
    """
    Round-level belief trajectory distilled to serial per-message updates.

    Attributes:
        proposition: Proposition identifier.
        updates: Per-message belief deltas for serial questions.
        condition: Round condition metadata.
        round_obj: Full round object used for trace rendering.
        source_path: Source JSONL path.
        source_line_index: 0-based line index in the source file.
        source_round_index: Optional 0-based index when one line stores a list.
    """

    proposition: str
    updates: tuple[float, ...]
    condition: Condition
    round_obj: Round
    source_path: Path
    source_line_index: int
    source_round_index: int | None


def parse_min_date(min_date: str | None) -> dt.date | None:
    """
    Parse an optional minimum date string.

    Args:
        min_date: Date string in YYYY-MM-DD format or None.

    Returns:
        Parsed date or None.
    """
    if min_date is None:
        return None
    return dt.date.fromisoformat(min_date)


def _file_date(path: Path) -> dt.date | None:
    """
    Parse a date from a JSONL filename stem.

    Args:
        path: JSONL file path.

    Returns:
        Parsed date when stem is YYYY-MM-DD, else None.
    """
    try:
        return dt.date.fromisoformat(path.stem)
    except ValueError:
        return None


def load_serial_trajectories(
    results_dir: Path,
    *,
    min_date: dt.date | None,
) -> list[RoundTrajectory]:
    """
    Load serial-question round trajectories from results.

    Args:
        results_dir: Root results directory.
        min_date: Optional minimum file date.

    Returns:
        List of parsed round trajectories.
    """
    trajectories: list[RoundTrajectory] = []
    if not results_dir.exists():
        return trajectories

    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue

        for jsonl_path in sorted(condition_dir.glob("*.jsonl")):
            file_date = _file_date(jsonl_path)
            if min_date is not None and file_date is not None and file_date < min_date:
                continue

            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line_index, raw_line in enumerate(handle):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed_line = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue

                    payloads: list[tuple[dict[str, object], int | None]] = []
                    if isinstance(parsed_line, dict):
                        payloads.append((parsed_line, None))
                    elif isinstance(parsed_line, list):
                        for round_index, payload in enumerate(parsed_line):
                            if isinstance(payload, dict):
                                payloads.append((payload, round_index))
                    else:
                        continue

                    for payload, round_index in payloads:
                        try:
                            round_obj = Round(**payload)
                        except (TypeError, ValueError, ValidationError):
                            continue
                        condition = round_obj.get_condition()
                        if (
                            condition.continuous_measure
                            != ContinuousMeasure.SERIAL_QUESTIONS
                        ):
                            continue
                        updates = extract_round_updates(round_obj)
                        if updates is None:
                            continue
                        trajectories.append(
                            RoundTrajectory(
                                proposition=round_obj.proposition,
                                updates=updates,
                                condition=condition,
                                round_obj=round_obj,
                                source_path=jsonl_path,
                                source_line_index=int(line_index),
                                source_round_index=(
                                    int(round_index)
                                    if round_index is not None
                                    else None
                                ),
                            )
                        )
    return trajectories


def _matches_common_filters(
    row: RoundTrajectory,
    *,
    include_control: bool,
    include_audio: bool,
    persuader_model: str | None,
    turn_limit: int | None,
    participant_proposition: str,
    exclude_bn_survey: bool = False,
) -> bool:
    """
    Check condition-level filters shared across corpora.

    Args:
        row: Round trajectory row.
        include_control: Whether to keep control rounds.
        include_audio: Whether to keep audio rounds.
        persuader_model: Optional exact llm_persuader filter.
        turn_limit: Optional exact turn_limit filter.
        participant_proposition: Tri-state participant_proposition filter.
        exclude_bn_survey: Exclude BN-survey rounds when True.

    Returns:
        True when row passes all shared filters.
    """
    condition = row.condition
    if not include_control and condition.control_dialogue:
        return False
    if not include_audio and condition.use_audio:
        return False
    if persuader_model is not None and condition.roles.llm_persuader != persuader_model:
        return False
    if turn_limit is not None and condition.turn_limit != turn_limit:
        return False
    if participant_proposition == "true" and not condition.participant_proposition:
        return False
    if participant_proposition == "false" and condition.participant_proposition:
        return False
    if exclude_bn_survey and bool(condition.enable_node_belief_survey):
        return False
    return True


def _iter_rows_matching_common_filters(
    rows: list[RoundTrajectory],
    *,
    include_control: bool,
    include_audio: bool,
    persuader_model: str | None,
    turn_limit: int | None,
    participant_proposition: str,
    exclude_bn_survey: bool = False,
) -> list[RoundTrajectory]:
    """
    Apply shared corpus filters and return passing rows.

    Args:
        rows: Candidate round trajectories.
        include_control: Whether to keep control rounds.
        include_audio: Whether to keep audio rounds.
        persuader_model: Optional exact llm_persuader filter.
        turn_limit: Optional exact turn_limit filter.
        participant_proposition: Tri-state participant_proposition filter.
        exclude_bn_survey: Exclude BN-survey rounds when True.

    Returns:
        List of rows that pass shared condition-level filters.
    """
    selected: list[RoundTrajectory] = []
    for row in rows:
        if not _matches_common_filters(
            row,
            include_control=include_control,
            include_audio=include_audio,
            persuader_model=persuader_model,
            turn_limit=turn_limit,
            participant_proposition=participant_proposition,
            exclude_bn_survey=exclude_bn_survey,
        ):
            continue
        selected.append(row)
    return selected


def select_human_reference(
    rows: list[RoundTrajectory],
    *,
    human_source: str,
    include_control: bool,
    include_audio: bool,
    persuader_model: str | None,
    turn_limit: int | None,
    participant_proposition: str,
    exclude_bn_survey: bool = False,
) -> list[RoundTrajectory]:
    """
    Select the human reference corpus.

    Args:
        rows: Candidate rows.
        human_source: Human source selector.
        include_control: Keep control rounds when True.
        include_audio: Keep audio rounds when True.
        persuader_model: Optional llm_persuader filter.
        turn_limit: Optional turn limit filter.
        participant_proposition: Tri-state participant proposition filter.
        exclude_bn_survey: Exclude BN-survey rounds when True.

    Returns:
        Filtered human-reference rows.
    """
    selected: list[RoundTrajectory] = []
    filtered_rows = _iter_rows_matching_common_filters(
        rows,
        include_control=include_control,
        include_audio=include_audio,
        persuader_model=persuader_model,
        turn_limit=turn_limit,
        participant_proposition=participant_proposition,
        exclude_bn_survey=exclude_bn_survey,
    )
    for row in filtered_rows:
        condition = row.condition
        roles = condition.roles
        if human_source == "llm-human-target":
            if not (roles.human_target and bool(roles.llm_persuader)):
                continue
        elif human_source == "human-human":
            if not (roles.human_target and bool(roles.human_persuader)):
                continue
        elif human_source == "all-human-target":
            if not roles.human_target:
                continue
        else:
            raise ValueError(f"Unknown human source: {human_source}")
        selected.append(row)
    return selected


def select_simulator(
    rows: list[RoundTrajectory],
    *,
    simulator_type: Literal["structure", "full", "full_no_rhetoric", "vanilla"],
    include_control: bool,
    include_audio: bool,
    persuader_model: str | None,
    turn_limit: int | None,
    participant_proposition: str,
    exclude_bn_survey: bool = False,
) -> list[RoundTrajectory]:
    """
    Select simulator rows by type using shared condition filters.

    Args:
        rows: Candidate rows.
        simulator_type: Simulator type selector.
        include_control: Keep control rounds when True.
        include_audio: Keep audio rounds when True.
        persuader_model: Optional llm_persuader filter.
        turn_limit: Optional turn limit filter.
        participant_proposition: Tri-state participant proposition filter.
        exclude_bn_survey: Exclude BN-survey rounds when True.

    Returns:
        Filtered simulator rows.
    """
    selected: list[RoundTrajectory] = []
    filtered_rows = _iter_rows_matching_common_filters(
        rows,
        include_control=include_control,
        include_audio=include_audio,
        persuader_model=persuader_model,
        turn_limit=turn_limit,
        participant_proposition=participant_proposition,
        exclude_bn_survey=exclude_bn_survey,
    )
    for row in filtered_rows:
        condition = row.condition
        roles = condition.roles
        if simulator_type == "structure":
            if not roles.llm_target or not condition.llm_target_use_bayes_structure:
                continue
        elif simulator_type == "full":
            if not roles.simulated_target or condition.simulated_target_no_rhetoric:
                continue
        elif simulator_type == "full_no_rhetoric":
            if not roles.simulated_target or not condition.simulated_target_no_rhetoric:
                continue
        elif simulator_type == "vanilla":
            if not roles.llm_target or condition.llm_target_use_bayes_structure:
                continue
        else:
            raise ValueError(f"Unknown simulator type: {simulator_type}")
        selected.append(row)
    return selected


def apply_proposition_matching(
    human_rows: list[RoundTrajectory],
    structure_rows: list[RoundTrajectory],
    full_rows: list[RoundTrajectory],
    vanilla_rows: list[RoundTrajectory],
    *,
    mode: str,
) -> tuple[
    list[RoundTrajectory],
    list[RoundTrajectory],
    list[RoundTrajectory],
    list[RoundTrajectory],
]:
    """
    Align corpora by proposition coverage.

    Args:
        human_rows: Human reference rows.
        structure_rows: Structure-conditioned simulator rows.
        full_rows: Full simulator rows.
        vanilla_rows: Vanilla simulator rows.
        mode: Matching mode.

    Returns:
        Tuple of filtered (human, structure, full, vanilla) rows.
    """
    if mode == "none":
        return human_rows, structure_rows, full_rows, vanilla_rows

    human_props = {row.proposition for row in human_rows}
    structure_props = {row.proposition for row in structure_rows}
    full_props = {row.proposition for row in full_rows}

    if mode == "human-overlap":
        structure_rows = [
            row for row in structure_rows if row.proposition in human_props
        ]
        full_rows = [row for row in full_rows if row.proposition in human_props]
        vanilla_rows = [row for row in vanilla_rows if row.proposition in human_props]
        return human_rows, structure_rows, full_rows, vanilla_rows

    if mode == "three-way-intersection":
        common = human_props & structure_props & full_props
        if vanilla_rows:
            vanilla_props = {row.proposition for row in vanilla_rows}
            common = common & vanilla_props
        human_rows = [row for row in human_rows if row.proposition in common]
        structure_rows = [row for row in structure_rows if row.proposition in common]
        full_rows = [row for row in full_rows if row.proposition in common]
        vanilla_rows = [row for row in vanilla_rows if row.proposition in common]
        return human_rows, structure_rows, full_rows, vanilla_rows

    raise ValueError(f"Unknown proposition match mode: {mode}")
