"""Shared CLI and corpus-selection helpers for simulator analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiment.cli_utils import add_min_date_arg
from simulation.human_likeness import (
    RoundTrajectory,
    apply_proposition_matching,
    select_human_reference,
    select_simulator,
)


def add_common_human_simulator_filter_args(
    parser: argparse.ArgumentParser,
    *,
    include_results_dir: bool,
    default_results_dir: Path | None,
    include_proposition_match: bool,
) -> None:
    """Attach shared human/simulator filtering arguments to a parser.

    Args:
        parser: Target parser to mutate.
        include_results_dir: Whether to add ``--results-dir``.
        default_results_dir: Default value used for ``--results-dir``.
        include_proposition_match: Whether to add ``--proposition-match``.

    Returns:
        None.
    """
    if include_results_dir:
        if default_results_dir is None:
            raise ValueError(
                "default_results_dir must be provided when include_results_dir is True."
            )
        parser.add_argument(
            "--results-dir",
            type=Path,
            default=default_results_dir,
            help="Root directory containing condition subdirectories with JSONL files.",
        )
    add_min_date_arg(parser)
    parser.add_argument(
        "--human-source",
        choices=["llm-human-target", "human-human", "all-human-target"],
        default="llm-human-target",
        help="Human reference corpus selector.",
    )
    parser.add_argument(
        "--persuader-model",
        type=str,
        default=None,
        help="Optional exact filter on roles.llm_persuader.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=None,
        help="Optional exact turn-limit filter.",
    )
    parser.add_argument(
        "--participant-proposition",
        choices=["any", "true", "false"],
        default="any",
        help="Filter by participant_proposition flag.",
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        help="Include control-dialogue rounds. Default excludes them.",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Include audio rounds. Default uses text-only rounds.",
    )
    parser.add_argument(
        "--exclude-bn-survey",
        action="store_true",
        help="Exclude BN-survey rounds (enable_node_belief_survey=true).",
    )
    if include_proposition_match:
        parser.add_argument(
            "--proposition-match",
            choices=["none", "human-overlap", "three-way-intersection"],
            default="human-overlap",
            help="How to align proposition coverage across corpora.",
        )


def selector_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build shared ``select_*`` filter kwargs from parsed CLI arguments.

    Args:
        args: Parsed CLI namespace with common filter fields.

    Returns:
        Mapping suitable for ``select_human_reference`` and ``select_simulator``.
    """
    return {
        "include_control": bool(args.include_control),
        "include_audio": bool(args.include_audio),
        "persuader_model": args.persuader_model,
        "turn_limit": args.turn_limit,
        "participant_proposition": str(args.participant_proposition),
        "exclude_bn_survey": bool(args.exclude_bn_survey),
    }


def add_include_vanilla_llm_target_arg(
    parser: argparse.ArgumentParser,
    *,
    default: bool,
    help_text: str,
) -> None:
    """Attach a shared ``--include-vanilla-llm-target`` boolean option.

    Args:
        parser: Target parser to mutate.
        default: Default boolean value.
        help_text: Help text shown in ``--help`` output.

    Returns:
        None.
    """
    parser.add_argument(
        "--include-vanilla-llm-target",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def select_human_rows(
    rows: list[RoundTrajectory],
    *,
    human_source: str,
    selector_kwargs: dict[str, Any],
) -> list[RoundTrajectory]:
    """Select a human-reference corpus using shared selector kwargs.

    Args:
        rows: Candidate trajectory rows.
        human_source: Human source selector.
        selector_kwargs: Shared filter kwargs.

    Returns:
        Filtered human-reference rows.
    """
    return select_human_reference(
        rows,
        human_source=human_source,
        **selector_kwargs,
    )


def select_matched_human_structure_full_vanilla(
    rows: list[RoundTrajectory],
    *,
    human_source: str,
    proposition_match: str,
    include_vanilla: bool,
    selector_kwargs: dict[str, Any],
) -> tuple[
    list[RoundTrajectory],
    list[RoundTrajectory],
    list[RoundTrajectory],
    list[RoundTrajectory],
]:
    """Select and proposition-match core comparison corpora.

    Args:
        rows: Candidate trajectory rows.
        human_source: Human source selector.
        proposition_match: Proposition matching mode.
        include_vanilla: Whether to select vanilla rows before matching.
        selector_kwargs: Shared filter kwargs.

    Returns:
        Tuple ``(human_rows, structure_rows, full_rows, vanilla_rows)``.
    """
    human_rows = select_human_rows(
        rows,
        human_source=human_source,
        selector_kwargs=selector_kwargs,
    )
    structure_rows = select_simulator(
        rows,
        simulator_type="structure",
        **selector_kwargs,
    )
    full_rows = select_simulator(
        rows,
        simulator_type="full",
        **selector_kwargs,
    )
    vanilla_rows = (
        select_simulator(
            rows,
            simulator_type="vanilla",
            **selector_kwargs,
        )
        if include_vanilla
        else []
    )

    return apply_proposition_matching(
        human_rows,
        structure_rows,
        full_rows,
        vanilla_rows,
        mode=proposition_match,
    )


def select_matched_human_structure_full_vanilla_from_args(
    rows: list[RoundTrajectory],
    *,
    args: argparse.Namespace,
    include_vanilla: bool,
) -> tuple[
    list[RoundTrajectory],
    list[RoundTrajectory],
    list[RoundTrajectory],
    list[RoundTrajectory],
]:
    """Select matched core corpora using parsed CLI arguments.

    Args:
        rows: Candidate trajectory rows.
        args: Parsed CLI namespace with shared corpus fields.
        include_vanilla: Whether to include vanilla rows before matching.

    Returns:
        Tuple ``(human_rows, structure_rows, full_rows, vanilla_rows)``.
    """
    return select_matched_human_structure_full_vanilla(
        rows,
        human_source=str(args.human_source),
        proposition_match=str(args.proposition_match),
        include_vanilla=include_vanilla,
        selector_kwargs=selector_kwargs_from_args(args),
    )
