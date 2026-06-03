"""Post-process counterfactual replay round errors without model calls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from simulation.counterfactual_replay.aggregation import (
    build_summary_rows,
    load_round_error_rows,
    summary_table_rows,
    write_csv_rows,
)

from .simulator_counterfactual_replay_common import (
    add_round_errors_io_args,
    infer_output_prefix,
)
from .tables import print_table


def _load_round_rows_any(path: Path) -> list[dict[str, Any]]:
    """Load replay round-error rows from CSV or JSONL."""

    if path.suffix != ".jsonl":
        return load_round_error_rows(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue
            row = dict(payload)
            for key in (
                "final_target_abs_error",
                "serial_trajectory_mae",
                "final_node_mae",
                "node_delta_mae",
            ):
                value = row.get(key)
                if value is None or value == "":
                    row[key] = math.nan
                    continue
                row[key] = float(value)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for replay post-processing."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate counterfactual replay round errors into corpus summaries "
            "without running any replay API calls."
        )
    )
    add_round_errors_io_args(parser)
    parser.add_argument(
        "--include-no-rhetoric-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include corpora with simulated_target_no_rhetoric=true. "
            "Default excludes them."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Read replay round errors, aggregate summary rows, and write outputs."""
    args = parse_args()
    round_rows = _load_round_rows_any(args.round_errors_csv)
    if not bool(args.include_no_rhetoric_target):
        round_rows = [
            row
            for row in round_rows
            if not str(row.get("corpus", "")).startswith("full_no_rhetoric_target__")
        ]
    if not round_rows:
        raise ValueError(
            f"No rows found in round-error input file: {args.round_errors_csv}"
        )

    summary_rows = build_summary_rows(round_rows)
    if not summary_rows:
        raise ValueError("No summary rows could be computed from round-error rows.")

    print_table(
        summary_table_rows(summary_rows),
        columns=["corpus", "n", "target_err", "node_err", "score"],
        title="Counterfactual Replay Human-Likeness",
        aligns={"n": "right", "target_err": "right", "node_err": "right"},
    )

    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix is not None
        else infer_output_prefix(args.round_errors_csv)
    )
    summary_csv_path = Path(f"{output_prefix}_summary.csv")
    write_csv_rows(summary_csv_path, summary_rows, list(summary_rows[0].keys()))
    print("Wrote output:", summary_csv_path)


if __name__ == "__main__":
    main()
