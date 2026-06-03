"""Render a paper figure for simulator stance-bias comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .simulator_plot_style import (
    COMPARISON_CORPUS_COLOR_MAP,
    CORE_COMPARISON_CORPUS_ORDER,
    PAPER_RESULTS_FIGURE_SIZE_INCHES,
    comparison_corpus_tick_label,
)
from .utils import resolve_repo_path, safe_float_or_nan, safe_int_or_none

DEFAULT_INPUT_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_stance_bias_summary.csv"
)
DEFAULT_OUTPUT_PDF = Path("analysis/figures/results_stance_bias.pdf")

# Use a non-interactive backend for CLI stability in headless environments.
plt.switch_backend("Agg")


@dataclass(frozen=True)
class StanceBiasRow:
    """Store one corpus summary row for plotting.

    Attributes:
        corpus: Canonical simulator corpus key.
        n_pairs: Number of mirrored for-vs-against pairs.
        stance_bias: Mean absolute for-vs-against gap.
        ci_low: Lower confidence bound for mean stance bias.
        ci_high: Upper confidence bound for mean stance bias.
    """

    corpus: str
    n_pairs: int
    stance_bias: float
    ci_low: float
    ci_high: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stance-bias plotting.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot simulator stance-bias bars from a summary CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input *_stance_bias_summary.csv file.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path for the stance-bias figure.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[StanceBiasRow]:
    """Load stance-bias rows from CSV.

    Args:
        path: Input summary CSV path.

    Returns:
        Filtered stance-bias rows.
    """
    rows: list[StanceBiasRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            corpus = str(row.get("corpus") or "").strip()
            if corpus not in CORE_COMPARISON_CORPUS_ORDER:
                continue
            n_pairs = safe_int_or_none(row.get("n_pairs"))
            stance_bias = safe_float_or_nan(row.get("mean_abs_for_minus_against"))
            ci_low = safe_float_or_nan(row.get("stance_bias_ci_low"))
            ci_high = safe_float_or_nan(row.get("stance_bias_ci_high"))
            if (
                n_pairs is None
                or n_pairs <= 0
                or not np.isfinite(stance_bias)
                or not np.isfinite(ci_low)
                or not np.isfinite(ci_high)
            ):
                continue
            rows.append(
                StanceBiasRow(
                    corpus=corpus,
                    n_pairs=int(n_pairs),
                    stance_bias=float(stance_bias),
                    ci_low=float(ci_low),
                    ci_high=float(ci_high),
                )
            )
    return rows


def _render_plot(rows: list[StanceBiasRow], output_pdf: Path) -> None:
    """Render stance-bias bar chart.

    Args:
        rows: Loaded rows for plotting.
        output_pdf: Destination PDF path.

    Returns:
        None.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    corpus_rank = {
        corpus: index for index, corpus in enumerate(CORE_COMPARISON_CORPUS_ORDER)
    }
    ranked_rows = sorted(
        rows,
        key=lambda item: (
            float(item.stance_bias),
            corpus_rank.get(item.corpus, len(corpus_rank)),
        ),
    )

    x_positions = np.arange(len(ranked_rows), dtype=float)
    bias_values = np.asarray(
        [float(row.stance_bias) for row in ranked_rows],
        dtype=float,
    )
    ci_lows = np.asarray([float(row.ci_low) for row in ranked_rows], dtype=float)
    ci_highs = np.asarray([float(row.ci_high) for row in ranked_rows], dtype=float)
    colors = [
        COMPARISON_CORPUS_COLOR_MAP.get(row.corpus, "#888888") for row in ranked_rows
    ]

    fig, axis = plt.subplots(figsize=PAPER_RESULTS_FIGURE_SIZE_INCHES)
    axis.bar(
        x_positions,
        bias_values,
        color=colors,
        width=0.62,
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.92,
        zorder=2,
    )

    lower_err = np.maximum(0.0, bias_values - ci_lows)
    upper_err = np.maximum(0.0, ci_highs - bias_values)
    axis.errorbar(
        x_positions,
        bias_values,
        yerr=np.vstack([lower_err, upper_err]),
        fmt="none",
        ecolor="#333333",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=3,
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [comparison_corpus_tick_label(row.corpus) for row in ranked_rows]
    )
    axis.tick_params(axis="x", labelsize=6.8, pad=1.2)
    axis.grid(axis="y", linestyle=":", alpha=0.24, zorder=0)
    axis.set_ylabel(r"Stance Bias ($\leftarrow$)", fontsize=10)
    axis.margins(x=0.06)
    axis.set_ylim(
        0.0, float(max(0.08, np.max(np.maximum(bias_values, ci_highs)) * 1.25))
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#444444")
    fig.subplots_adjust(left=0.26, bottom=0.18, right=0.98, top=0.96)
    fig.savefig(output_pdf, dpi=220)
    plt.close(fig)


def main() -> None:
    """Run stance-bias plotting from summary CSV.

    Returns:
        None.
    """
    args = parse_args()
    reference_file = Path(__file__).resolve()
    input_csv = resolve_repo_path(args.input_csv, reference_file=reference_file)
    output_pdf = resolve_repo_path(args.output_pdf, reference_file=reference_file)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    rows = _load_rows(input_csv)
    if not rows:
        raise ValueError("No stance-bias rows passed CSV filters.")
    _render_plot(rows, output_pdf)
    print(f"Wrote stance-bias figure: {output_pdf}")


if __name__ == "__main__":
    main()
