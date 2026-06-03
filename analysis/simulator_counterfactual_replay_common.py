"""Shared CLI helpers for counterfactual replay post-processing scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_round_errors_io_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared replay round-error input/output prefix arguments.

    Args:
        parser: Parser that should receive the shared arguments.

    Returns:
        None.
    """
    parser.add_argument(
        "--round-errors-csv",
        type=Path,
        required=True,
        help=(
            "Path to <prefix>_round_errors.csv or <prefix>_round_errors.jsonl "
            "from simulator_counterfactual_replay."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix for output files. Defaults to the input prefix inferred from "
            "--round-errors-csv."
        ),
    )


def infer_output_prefix(round_errors_csv: Path) -> Path:
    """Infer an output prefix from a replay round-error CSV path.

    Args:
        round_errors_csv: Path to a replay round-error CSV.

    Returns:
        Output prefix with ``_round_errors`` suffix removed when present.
    """
    csv_suffix = "_round_errors.csv"
    jsonl_suffix = "_round_errors.jsonl"
    if round_errors_csv.name.endswith(csv_suffix):
        base_name = round_errors_csv.name[: -len(csv_suffix)]
        return round_errors_csv.with_name(base_name)
    if round_errors_csv.name.endswith(jsonl_suffix):
        base_name = round_errors_csv.name[: -len(jsonl_suffix)]
        return round_errors_csv.with_name(base_name)
    if round_errors_csv.suffix in {".csv", ".jsonl"}:
        return round_errors_csv.with_suffix("")
    return round_errors_csv
