"""
Aggregate annotation outputs and fit regression models for persuasion rounds.

This script joins annotation JSONL outputs to round results, aggregates
message-level scores to the dialogue level, exports a tidy CSV, and optionally
fits mixed-effects models via R (lme4).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path
from typing import Sequence

from analysis.annotation_regression_utils import add_summary_csv_arg
from annotation.records import (
    ANNOTATION_CODES,
    add_annotation_args,
    compute_dialogue_scores,
    extract_message_text,
    load_annotation_rows,
    resolve_annotation_paths_from_args,
)
from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import add_condition_filter_args, filters_from_args
from experiment.round import IndexedRound, Round
from experiment.round_lookup import RoundKey, build_round_lookup

DEFAULT_R_SCRIPT = "analysis/annotation_regression.R"
DEFAULT_RSCRIPT_BIN = "Rscript"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for annotation regression."""
    parser = argparse.ArgumentParser(
        description="Build regression data from annotation outputs."
    )
    add_min_date_arg(parser)
    add_condition_filter_args(parser)
    add_annotation_args(parser)
    parser.add_argument(
        "--output-csv",
        default="analysis/data/annotation_regression.csv",
        help="Output CSV path.",
    )
    add_summary_csv_arg(parser)
    parser.add_argument(
        "--run-r",
        action="store_true",
        help="Run the R mixed-effects model after writing the CSV.",
    )
    parser.add_argument(
        "--no-per-condition",
        action="store_true",
        help="Skip per-condition model fits.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help=(
            "Load all matching JSONL files per condition directory instead of "
            "only the newest matching file."
        ),
    )
    return parser.parse_args()


def count_persuader_messages(round_obj: Round) -> int:
    """Return the number of persuader messages in a round."""
    return sum(
        1
        for message in round_obj.messages
        if str(message.get("role", "")).lower() == "persuader"
    )


def total_persuader_message_chars(round_obj: Round) -> int:
    """Return the total character count for persuader messages."""
    lengths: list[int] = []
    for message in round_obj.messages:
        if str(message.get("role", "")).lower() != "persuader":
            continue
        text = extract_message_text(message)
        if text:
            lengths.append(len(text))
    return sum(lengths)


def average_persuader_message_chars(round_obj: Round) -> float:
    """Return the average character count per persuader message."""
    message_count = count_persuader_messages(round_obj)
    if message_count == 0:
        return 0.0
    total_chars = total_persuader_message_chars(round_obj)
    return float(total_chars) / message_count


def proposition_hash(proposition: str) -> str:
    """Return a stable hash for a proposition string."""
    digest = hashlib.sha1(proposition.encode("utf-8")).hexdigest()
    return digest


def condition_factors(condition: object) -> dict[str, object]:
    """Extract factor columns from a Condition-like object."""
    roles = getattr(condition, "roles", None)
    participant_proposition = getattr(condition, "participant_proposition", None)
    persuader_type = None
    if roles is not None:
        if getattr(roles, "human_persuader", False):
            persuader_type = "human"
        else:
            persuader_type = "llm"
    return {
        "participant_proposition": participant_proposition,
        "persuader_type": persuader_type,
    }


def build_regression_rows(
    *,
    round_lookup: dict[RoundKey, IndexedRound],
    score_map: dict[RoundKey, dict[str, float]],
) -> list[dict[str, object]]:
    """Build regression-ready rows from rounds and annotation aggregates."""
    rows: list[dict[str, object]] = []
    for key, scores in score_map.items():
        indexed_round = round_lookup.get(key)
        if indexed_round is None:
            continue
        round_obj = indexed_round.round
        condition = indexed_round.condition
        if (
            round_obj.target_initial_belief is None
            or round_obj.target_final_belief is None
        ):
            continue

        row: dict[str, object] = {
            "source_path": key.source_path,
            "line_index": key.line_index,
            "round_index": key.round_index,
            "condition": str(condition.as_non_id_role()),
            "delta_changed": round_obj.target_belief_change(),
            "baseline_belief": round_obj.target_initial_belief,
            "msg_count": count_persuader_messages(round_obj),
            "avg_chars": average_persuader_message_chars(round_obj),
            "proposition_id": proposition_hash(str(round_obj.proposition)),
            "target_id": getattr(round_obj, "target_id", None),
            "persuader_id": getattr(round_obj, "persuader_id", None),
            "human_target_id": getattr(round_obj, "human_target_id", None),
            "human_persuader_id": getattr(round_obj, "human_persuader_id", None),
        }
        if not row["persuader_id"]:
            row["persuader_id"] = row["human_persuader_id"]
        if not row["target_id"]:
            row["target_id"] = row["human_target_id"]
        roles = getattr(condition, "roles", None)
        row.update(condition_factors(condition))
        if not row["persuader_id"] and roles is not None:
            llm_persuader = getattr(roles, "llm_persuader", None)
            if llm_persuader:
                row["persuader_id"] = llm_persuader
        for code in ANNOTATION_CODES:
            row[f"mean_{code}"] = scores.get(code)
        rows.append(row)
    return rows


def write_csv(path: str, rows: Sequence[dict[str, object]]) -> None:
    """Write rows to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def zscore_columns(rows: list[dict[str, object]], columns: Sequence[str]) -> None:
    """Add z-scored columns for the provided column names."""
    if not rows:
        return
    for col in columns:
        values = [
            float(row[col])
            for row in rows
            if row.get(col) is not None
            and row.get(col) == row.get(col)
            and row.get(col) != ""
        ]
        if not values:
            continue
        mean_val = sum(values) / len(values)
        variance = sum((val - mean_val) ** 2 for val in values) / len(values)
        std_val = variance**0.5
        z_col = f"{col}_z"
        for row in rows:
            val = row.get(col)
            if val is None or val == "":
                row[z_col] = None
            else:
                if std_val == 0:
                    row[z_col] = 0.0
                else:
                    row[z_col] = (float(val) - mean_val) / std_val


def run_r_lmer(
    *,
    data_csv: str,
    summary_csv: str,
    per_condition: bool,
) -> None:
    """Run the R lmer script with the prepared CSV."""
    cmd = [
        DEFAULT_RSCRIPT_BIN,
        DEFAULT_R_SCRIPT,
        "--data",
        data_csv,
        "--summary",
        summary_csv,
    ]
    if not per_condition:
        cmd.append("--no-per-condition")
    subprocess.run(cmd, check=True)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    condition_filters = filters_from_args(args)
    annotation_paths = resolve_annotation_paths_from_args(args)

    annotation_rows = load_annotation_rows(annotation_paths)
    round_lookup = build_round_lookup(
        min_date=args.min_date,
        condition_filters=condition_filters,
        include_all_files=bool(args.all_files),
    )
    score_map = compute_dialogue_scores(annotation_rows)
    rows = build_regression_rows(
        round_lookup=round_lookup,
        score_map=score_map,
    )
    zscore_columns(
        rows,
        [
            "baseline_belief",
            "msg_count",
            "avg_chars",
            *[f"mean_{code}" for code in ANNOTATION_CODES],
        ],
    )
    write_csv(args.output_csv, rows)

    if args.run_r:
        run_r_lmer(
            data_csv=args.output_csv,
            summary_csv=args.summary_csv,
            per_condition=not args.no_per_condition,
        )


if __name__ == "__main__":
    main()
