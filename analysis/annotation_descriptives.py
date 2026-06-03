"""
Print descriptive counts for annotation scores.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from annotation.records import (
    ANNOTATION_CODES,
    add_annotation_args,
    add_max_rows_arg,
    condition_from_source_path,
    extract_scores,
    iter_annotation_records,
    resolve_annotation_paths_from_args,
)
from experiment.condition_filters import (
    add_condition_filter_args,
    condition_matches_filters,
    filters_from_args,
)
from experiment.round_lookup import normalize_source_path

from .formatting import condition_color_map, split_condition_label
from .stats import bootstrap_mean_ci
from .tables import print_table


def parse_args() -> argparse.Namespace:
    """Parse CLI args for annotation descriptives."""
    parser = argparse.ArgumentParser(
        description="Summarize annotation score distributions."
    )
    add_condition_filter_args(parser)
    add_annotation_args(parser)
    add_max_rows_arg(parser)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a grouped bar chart of mean scores by condition.",
    )
    parser.add_argument(
        "--plot-path",
        default="analysis/figures/annotation_descriptives.pdf",
        help="Output path for the plot.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    return parser.parse_args()


def load_annotation_records(
    paths: Sequence[Path], *, max_rows: int | None
) -> list[dict[str, object]]:
    """Load parsed annotation records from JSONL."""
    rows = iter_annotation_records(paths, max_rows=max_rows)
    cleaned: list[dict[str, object]] = []
    for record in rows:
        parsed = record.get("parsed")
        if not isinstance(parsed, dict):
            continue
        scores = extract_scores(parsed)
        record["parsed_scores"] = {k: int(round(v)) for k, v in scores.items()}
        cleaned.append(record)
    return cleaned


def update_counts(counts: dict[str, Counter], scores: dict[str, int]) -> None:
    """Update bin counts for the given score dictionary."""
    for code, value in scores.items():
        counts[code][value] += 1


def counts_to_rows(counts: dict[str, Counter]) -> list[dict[str, int]]:
    """Convert score counters into table rows."""
    rows: list[dict[str, int]] = []
    for code in ANNOTATION_CODES:
        counter = counts.get(code, Counter())
        row: dict[str, int] = {"code": code}
        for score in range(0, 6):
            row[str(score)] = counter.get(score, 0)
        rows.append(row)
    return rows


def extract_condition_label(record: dict[str, object]) -> str:
    """Extract a condition label from an annotation record."""
    condition = record.get("condition")
    if isinstance(condition, str) and condition.strip():
        return condition.strip()
    return "unknown"


def print_summary(title: str, counts: dict[str, Counter], total: int) -> None:
    """Print a summary table for a given set of counts."""
    print(title)
    print(f"n={total}")
    rows = counts_to_rows(counts)
    columns = ["code", "0", "1", "2", "3", "4", "5"]
    aligns = {col: "right" for col in columns if col != "code"}
    print_table(rows, columns=columns, aligns=aligns)
    print("---")


def _record_scores_and_condition_label(
    record: dict[str, object],
    *,
    condition_filters: dict[str, object],
) -> tuple[dict[str, int], str] | None:
    """
    Extract score payload and condition label if record passes condition filters.

    Args:
        record: Parsed annotation record.
        condition_filters: Optional condition filter mapping.

    Returns:
        Tuple of parsed score mapping and condition label, or None when the
        record has no scores or is filtered out.
    """
    parsed_scores = record.get("parsed_scores") or {}
    if not parsed_scores:
        return None
    condition_label = extract_condition_label(record)
    if condition_filters:
        target = record.get("target") or {}
        source_path = normalize_source_path(str(target.get("source_path", "")))
        condition_obj = condition_from_source_path(source_path)
        if condition_obj is not None and not condition_matches_filters(
            condition_obj, condition_filters
        ):
            return None
    return parsed_scores, condition_label


def compute_descriptive_counts(
    records: Sequence[dict[str, object]],
    *,
    condition_filters: dict[str, object],
) -> tuple[dict[str, Counter], dict[str, dict[str, Counter]], Counter]:
    """Return overall and per-condition counts."""
    overall_counts: dict[str, Counter] = {code: Counter() for code in ANNOTATION_CODES}
    per_condition: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {code: Counter() for code in ANNOTATION_CODES}
    )
    per_condition_totals: Counter = Counter()

    for record in records:
        parsed_record = _record_scores_and_condition_label(
            record,
            condition_filters=condition_filters,
        )
        if parsed_record is None:
            continue
        scores, condition_label = parsed_record
        update_counts(overall_counts, scores)
        update_counts(per_condition[condition_label], scores)
        per_condition_totals[condition_label] += 1
    return overall_counts, per_condition, per_condition_totals


def collect_score_rows(
    records: Sequence[dict[str, object]],
    *,
    condition_filters: dict[str, object],
) -> list[dict[str, object]]:
    """Collect message-level scores with condition labels."""
    rows: list[dict[str, object]] = []
    for record in records:
        parsed_record = _record_scores_and_condition_label(
            record,
            condition_filters=condition_filters,
        )
        if parsed_record is None:
            continue
        parsed_scores, condition_label = parsed_record
        for code, score in parsed_scores.items():
            rows.append(
                {
                    "condition": condition_label,
                    "code": code,
                    "score": float(score),
                }
            )
    return rows


def plot_grouped_bars(
    rows: Sequence[dict[str, object]],
    *,
    output_path: str,
    n_boot: int,
) -> None:
    """Plot grouped bars with bootstrap CIs by condition."""
    if not rows:
        return
    conditions = sorted({str(row["condition"]) for row in rows})
    summary = summarize_bootstrap(rows, conditions=conditions, n_boot=n_boot)
    render_grouped_bar_plot(
        summary,
        conditions=conditions,
        output_path=output_path,
    )


def summarize_bootstrap(
    rows: Sequence[dict[str, object]],
    *,
    conditions: Sequence[str],
    n_boot: int,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Return mean and CI per condition and code."""
    summary: dict[str, dict[str, tuple[float, float, float]]] = {}
    for condition in conditions:
        per_code: dict[str, tuple[float, float, float]] = {}
        for code in ANNOTATION_CODES:
            values = [
                float(row["score"])
                for row in rows
                if row["condition"] == condition and row["code"] == code
            ]
            mean_val, lo, hi = bootstrap_mean_ci(
                np.array(values, dtype=float),
                n_boot=n_boot,
            )
            per_code[code] = (mean_val, lo, hi)
        summary[condition] = per_code
    return summary


def render_grouped_bar_plot(
    summary: dict[str, dict[str, tuple[float, float, float]]],
    *,
    conditions: Sequence[str],
    output_path: str,
) -> None:
    """Render a grouped bar plot with CI error bars."""
    codes = ANNOTATION_CODES
    color_map = condition_color_map(conditions)
    width = 0.8 / max(1, len(conditions))
    x_positions = list(range(len(codes)))
    _, ax = plt.subplots(figsize=(8, 4.5))
    for idx, condition in enumerate(conditions):
        means, err_low, err_high = compute_error_bars(
            summary, condition=condition, codes=codes
        )
        offsets = compute_offsets(
            x_positions,
            idx=idx,
            n_conditions=len(conditions),
            width=width,
        )
        ax.bar(
            offsets,
            means,
            width=width,
            color=color_map.get(condition, "#000000"),
            label=split_condition_label(condition),
            yerr=[err_low, err_high],
            capsize=2,
            linewidth=0.5,
            edgecolor="black",
        )

    finalize_plot(ax, x_positions=x_positions, codes=codes, output_path=output_path)


def compute_error_bars(
    summary: dict[str, dict[str, tuple[float, float, float]]],
    *,
    condition: str,
    codes: Sequence[str],
) -> tuple[list[float], list[float], list[float]]:
    """Return means and error bars for a condition."""
    means: list[float] = []
    err_low: list[float] = []
    err_high: list[float] = []
    for code in codes:
        mean_val, lo, hi = summary[condition][code]
        means.append(mean_val)
        err_low.append(mean_val - lo)
        err_high.append(hi - mean_val)
    return means, err_low, err_high


def compute_offsets(
    x_positions: Sequence[int],
    *,
    idx: int,
    n_conditions: int,
    width: float,
) -> list[float]:
    """Return bar offsets for a condition index."""
    center = (n_conditions - 1) / 2
    return [x + (idx - center) * width for x in x_positions]


def finalize_plot(
    ax,
    *,
    x_positions: Sequence[int],
    codes: Sequence[str],
    output_path: str,
) -> None:
    """Finalize and save the plot."""
    ax.set_xticks(x_positions)
    ax.set_xticklabels(codes, rotation=25, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_title("Annotation scores by condition (mean ± 95% CI)")
    ax.legend(fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    condition_filters = filters_from_args(args)
    annotation_paths = resolve_annotation_paths_from_args(args)

    records = load_annotation_records(annotation_paths, max_rows=args.max_rows)
    overall_counts, per_condition, per_condition_totals = compute_descriptive_counts(
        records,
        condition_filters=condition_filters,
    )
    score_rows = collect_score_rows(
        records,
        condition_filters=condition_filters,
    )

    print_summary("Overall", overall_counts, sum(per_condition_totals.values()))
    for condition, counts in sorted(per_condition.items()):
        total = per_condition_totals[condition]
        print_summary(f"Condition: {condition}", counts, total)

    if args.plot:
        plot_grouped_bars(
            score_rows,
            output_path=args.plot_path,
            n_boot=args.n_boot,
        )


if __name__ == "__main__":
    main()
