"""Render the paper base/SFT/GRPO policy-optimization comparison figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import resolve_repo_path, safe_float_or_nan

DEFAULT_SUMMARY_CSV = Path(
    "analysis/data/results_policy_optimization_base_sft_grpo/run_metric_summary.csv"
)
DEFAULT_OUTPUT_PDF = Path(
    "analysis/figures/results_policy_optimization_base_sft_grpo.pdf"
)
DEFAULT_RUN_ORDER = "base_qwen,sft_qbase_nomins,grpo_qbase_sparse_recheck"
DEFAULT_LABELS = "Base Qwen,SFT Qwen,GRPO Qwen"
DEFAULT_COLORS = "#4C78A8,#72B7B2,#54A24B"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the policy optimization figure.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot the paper-facing base/SFT/GRPO comparison from "
            "rl.compare_baseline_runs run_metric_summary.csv output."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Input run_metric_summary.csv path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path for the paper figure.",
    )
    parser.add_argument(
        "--run-order",
        type=str,
        default=DEFAULT_RUN_ORDER,
        help="Comma-separated run_name order from summary CSV.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=DEFAULT_LABELS,
        help="Comma-separated x-axis labels aligned with --run-order.",
    )
    parser.add_argument(
        "--colors",
        type=str,
        default=DEFAULT_COLORS,
        help="Comma-separated bar colors aligned with --run-order.",
    )
    return parser.parse_args()


def _parse_csv_list(raw_value: str) -> list[str]:
    """Parse and normalize a comma-separated list.

    Args:
        raw_value: Comma-delimited raw string.

    Returns:
        Parsed non-empty values in order.
    """
    values = [token.strip() for token in raw_value.split(",")]
    return [value for value in values if value]


def _load_summary_rows(path: Path) -> dict[str, dict[str, str]]:
    """Load run summary rows keyed by run name.

    Args:
        path: Input summary CSV path.

    Returns:
        Mapping from run_name to raw row dictionary.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("run_name") or "").strip(): row
            for row in reader
            if str(row.get("run_name") or "").strip()
        }


def _validate_plot_inputs(
    *,
    run_order: list[str],
    labels: list[str],
    colors: list[str],
    rows_by_run: dict[str, dict[str, str]],
) -> None:
    """Validate plotting inputs before rendering.

    Args:
        run_order: Ordered run identifiers.
        labels: Ordered display labels.
        colors: Ordered bar colors.
        rows_by_run: Summary rows keyed by run name.
    """
    if not run_order:
        raise ValueError("--run-order must contain at least one run.")
    if len(labels) != len(run_order):
        raise ValueError("--labels count must match --run-order count.")
    if len(colors) != len(run_order):
        raise ValueError("--colors count must match --run-order count.")

    missing_runs = [run_name for run_name in run_order if run_name not in rows_by_run]
    if missing_runs:
        joined = ", ".join(missing_runs)
        raise ValueError(f"Missing run rows in summary CSV: {joined}")


def _plot_policy_comparison(
    *,
    output_pdf: Path,
    run_order: list[str],
    labels: list[str],
    colors: list[str],
    rows_by_run: dict[str, dict[str, str]],
) -> None:
    """Render the base/SFT/GRPO comparison bar chart.

    Args:
        output_pdf: Output figure path.
        run_order: Ordered run identifiers.
        labels: Ordered display labels.
        colors: Ordered bar colors.
        rows_by_run: Summary rows keyed by run name.
    """
    means = np.asarray(
        [
            safe_float_or_nan(rows_by_run[run_name].get("metric_mean"))
            for run_name in run_order
        ],
        dtype=float,
    )
    ci_low = np.asarray(
        [
            safe_float_or_nan(rows_by_run[run_name].get("metric_ci_low"))
            for run_name in run_order
        ],
        dtype=float,
    )
    ci_high = np.asarray(
        [
            safe_float_or_nan(rows_by_run[run_name].get("metric_ci_high"))
            for run_name in run_order
        ],
        dtype=float,
    )

    if (
        not np.all(np.isfinite(means))
        or not np.all(np.isfinite(ci_low))
        or not np.all(np.isfinite(ci_high))
    ):
        raise ValueError("Summary CSV contains non-finite metric mean/CI values.")

    x_pos = np.arange(len(run_order), dtype=float)
    lower_errors = means - ci_low
    upper_errors = ci_high - means

    fig, axis = plt.subplots(figsize=(5.8, 4.0))
    axis.bar(x_pos, means, color=colors, alpha=0.92)
    axis.errorbar(
        x_pos,
        means,
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="#222222",
        capsize=4,
        elinewidth=1.1,
    )
    axis.set_xticks(x_pos)
    axis.set_xticklabels(labels, rotation=0)
    axis.set_ylabel("Mean terminal_delta_mean")
    axis.grid(axis="y", linestyle=":", alpha=0.25)
    axis.set_ylim(
        min(0.0, float(np.min(ci_low)) - 0.02),
        float(np.max(ci_high)) + 0.02,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(output_pdf, dpi=220)
    plt.close(fig)


def main() -> None:
    """Render the policy optimization paper figure from run summary CSV."""
    args = parse_args()
    reference_file = Path(__file__).resolve()
    summary_csv = resolve_repo_path(args.summary_csv, reference_file=reference_file)
    output_pdf = resolve_repo_path(args.output_pdf, reference_file=reference_file)

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    run_order = _parse_csv_list(str(args.run_order))
    labels = _parse_csv_list(str(args.labels))
    colors = _parse_csv_list(str(args.colors))
    rows_by_run = _load_summary_rows(summary_csv)
    _validate_plot_inputs(
        run_order=run_order,
        labels=labels,
        colors=colors,
        rows_by_run=rows_by_run,
    )
    _plot_policy_comparison(
        output_pdf=output_pdf,
        run_order=run_order,
        labels=labels,
        colors=colors,
        rows_by_run=rows_by_run,
    )
    print(f"Wrote policy optimization figure: {output_pdf}")


if __name__ == "__main__":
    main()
