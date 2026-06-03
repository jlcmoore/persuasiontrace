"""Run planned Welch tests comparing non-control persuasiveness buckets to control.

This module computes three planned comparisons used in the paper-facing
persuasiveness analysis and writes a tidy CSV with raw and Holm-adjusted
p-values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import condition_matches_filters

from .data_loading import load_dataframe

DEFAULT_OUTPUT_CSV = Path("analysis/data/persuasiveness_welch_vs_control.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute planned Welch tests for persuasiveness buckets versus "
            "control and write a paper-facing CSV."
        )
    )
    add_min_date_arg(parser)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for Welch test results.",
    )
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    """Resolve a path relative to repository root.

    Args:
        path: Raw path argument.

    Returns:
        Absolute path anchored at repository root when input is relative.
    """
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).resolve()


def _bucket_filters() -> dict[str, dict[str, object]]:
    """Return canonical condition filters for persuasiveness buckets.

    Returns:
        Mapping from bucket name to condition filter dictionary.
    """
    return {
        "control": {
            "continuous_measure": "serial-questions",
            "control_dialogue": True,
            "human_persuader": False,
            "human_target": True,
            "llm_persuader": "gpt-5-2025-08-07",
            "use_audio": False,
            "participant_proposition": False,
            "enable_node_belief_survey": False,
            "proposition_source": "debategpt",
            "minimum_turns": 4,
            "turn_limit": 4,
            "factual_domain": False,
        },
        "standard_text": {
            "continuous_measure": "serial-questions",
            "control_dialogue": False,
            "human_target": True,
            "use_audio": False,
            "participant_proposition": False,
            "enable_node_belief_survey": False,
            "llm_persuader": "gpt-5-2025-08-07",
            "proposition_source": "debategpt",
            "minimum_turns": 4,
            "turn_limit": 4,
            "factual_domain": False,
        },
        "personal_text": {
            "continuous_measure": "serial-questions",
            "control_dialogue": False,
            "human_target": True,
            "use_audio": False,
            "participant_proposition": True,
        },
        "audio": {
            "continuous_measure": "serial-questions",
            "control_dialogue": False,
            "human_persuader": False,
            "human_target": True,
            "llm_persuader": "gpt-5-2025-08-07",
            "use_audio": True,
            "show_transcript": True,
            "participant_proposition": False,
            "enable_node_belief_survey": False,
            "proposition_source": "debategpt",
            "minimum_turns": 4,
            "turn_limit": 4,
            "factual_domain": False,
        },
    }


def _subset_deltas(
    dataframe: pd.DataFrame,
    filters: dict[str, object],
) -> np.ndarray:
    """Extract persuader-relative deltas for one condition filter.

    Args:
        dataframe: Analysis dataframe from ``load_dataframe``.
        filters: Condition filter mapping.

    Returns:
        One-dimensional array of ``delta_dir`` values for matching rounds.
    """
    mask = dataframe["condition_obj"].apply(
        lambda condition: condition_matches_filters(condition, filters)
    )
    return dataframe.loc[mask, "delta_dir"].to_numpy(dtype=float)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Compute Holm-adjusted p-values.

    Args:
        p_values: Unadjusted p-values.

    Returns:
        Holm-adjusted p-values in original order.
    """
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    n_tests = len(ordered)
    adjusted = [1.0] * n_tests
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (n_tests - rank) * float(p_value))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _build_result_row(
    bucket: str,
    bucket_values: np.ndarray,
    control_values: np.ndarray,
) -> dict[str, float | str | int]:
    """Build one test result row for a non-control bucket.

    Args:
        bucket: Non-control bucket name.
        bucket_values: Persuader-relative deltas for the bucket.
        control_values: Persuader-relative deltas for control.

    Returns:
        Mapping containing Welch statistics and p-values.
    """
    test_two = stats.ttest_ind(
        bucket_values,
        control_values,
        equal_var=False,
        alternative="two-sided",
    )
    test_greater = stats.ttest_ind(
        bucket_values,
        control_values,
        equal_var=False,
        alternative="greater",
    )
    return {
        "comparison": f"{bucket}_minus_control",
        "bucket": bucket,
        "n_bucket": int(bucket_values.size),
        "n_control": int(control_values.size),
        "mean_bucket": float(np.mean(bucket_values)),
        "mean_control": float(np.mean(control_values)),
        "mean_diff": float(np.mean(bucket_values) - np.mean(control_values)),
        "welch_t": float(test_two.statistic),
        "welch_df": float(test_two.df),
        "p_two_sided": float(test_two.pvalue),
        "p_two_sided_holm": 1.0,
        "p_one_sided_greater": float(test_greater.pvalue),
        "p_one_sided_greater_holm": 1.0,
    }


def _compute_rows(min_date: str | None) -> list[dict[str, float | str | int]]:
    """Run planned Welch tests and return output rows.

    Args:
        min_date: Optional minimum date filter passed to round loading.

    Returns:
        Ordered list of CSV row dictionaries.
    """
    frame = load_dataframe(min_date=min_date, filters=None)
    if frame.empty:
        raise ValueError("No rounds found for the provided min-date filter.")

    filters = _bucket_filters()
    control_values = _subset_deltas(frame, filters["control"])
    if control_values.size < 2:
        raise ValueError(
            "Control bucket has fewer than 2 rows; Welch test is undefined."
        )

    rows: list[dict[str, float | str | int]] = []
    for bucket in ["standard_text", "personal_text", "audio"]:
        bucket_values = _subset_deltas(frame, filters[bucket])
        if bucket_values.size < 2:
            raise ValueError(
                f"Bucket '{bucket}' has fewer than 2 rows; Welch test is undefined."
            )
        rows.append(_build_result_row(bucket, bucket_values, control_values))

    two_sided = [float(row["p_two_sided"]) for row in rows]
    one_sided = [float(row["p_one_sided_greater"]) for row in rows]
    for index, adjusted in enumerate(_holm_adjust(two_sided)):
        rows[index]["p_two_sided_holm"] = adjusted
    for index, adjusted in enumerate(_holm_adjust(one_sided)):
        rows[index]["p_one_sided_greater_holm"] = adjusted
    return rows


def _write_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    """Write Welch test rows to CSV.

    Args:
        path: Output CSV path.
        rows: Result rows to serialize.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison",
        "bucket",
        "n_bucket",
        "n_control",
        "mean_bucket",
        "mean_control",
        "mean_diff",
        "welch_t",
        "welch_df",
        "p_two_sided",
        "p_two_sided_holm",
        "p_one_sided_greater",
        "p_one_sided_greater_holm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run planned Welch tests and write paper-facing CSV output.

    Returns:
        None.
    """
    args = parse_args()
    output_csv = _resolve_repo_path(args.output_csv)
    rows = _compute_rows(min_date=args.min_date)
    _write_rows(output_csv, rows)
    print(f"Wrote Welch comparison CSV: {output_csv}")


if __name__ == "__main__":
    main()
