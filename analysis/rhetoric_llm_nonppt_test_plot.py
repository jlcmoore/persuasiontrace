"""Build a test coefficient plot for the LLM non-ppt rhetoric regression.

This script fits a slim OLS model on a pre-filtered CSV and writes:
1) a coefficient summary CSV, and
2) a horizontal coefficient plot with confidence intervals.

Optionally, it overlays coefficients from an external summary CSV
(for example, Salvi ordinal-fit outputs) for shared rhetoric terms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .simulator_plot_style import PAPER_SQUARE_FIGURE_SIZE_INCHES

matplotlib.use("Agg")

FORMULA = (
    "delta_changed ~ mean_logos_z + mean_pathos_z + mean_ethos_z + baseline_belief_z"
)
DEFAULT_INPUT_CSV = Path(
    "analysis/data/annotation_regression_llm_human_nocontrol_noaudio_ppt_false.csv"
)
DEFAULT_OUTPUT_CSV = Path("analysis/data/test_rhetoric_llm_nonppt_coefficients.csv")
DEFAULT_OUTPUT_PDF = Path("analysis/figures/test_rhetoric_llm_nonppt_coefficients.pdf")
TERM_ORDER = [
    "mean_logos_z",
    "mean_pathos_z",
    "mean_ethos_z",
]
TERM_LABELS = {
    "mean_logos_z": "Logos (z)",
    "mean_pathos_z": "Pathos (z)",
    "mean_ethos_z": "Ethos (z)",
}
OVERLAY_DEFAULT_LABEL = "Salvi"
OVERLAY_RHETORIC_TERMS = ["mean_logos_z", "mean_pathos_z", "mean_ethos_z"]


@dataclass(frozen=True)
class OverlayPlotData:
    """Optional overlay dataset metadata and coefficients."""

    table: pd.DataFrame
    label: str
    nobs: int | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the test plot.

    Args:
        None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fit a slim rhetoric model and write a test coefficient plot."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Pre-filtered regression CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output coefficient summary CSV path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output coefficient plot PDF path.",
    )
    parser.add_argument(
        "--cov-type",
        default="none",
        help=(
            "Covariance type for uncertainty estimates. "
            "Use 'none' for classic OLS covariance."
        ),
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for coefficient intervals.",
    )
    parser.add_argument(
        "--overlay-summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional coefficient summary CSV to overlay for shared rhetoric terms. "
            "Expected columns: term, estimate, ci_low, ci_high (and optional nobs)."
        ),
    )
    parser.add_argument(
        "--overlay-label",
        default=OVERLAY_DEFAULT_LABEL,
        help="Legend label for overlayed coefficients.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=float(PAPER_SQUARE_FIGURE_SIZE_INCHES[0]),
        help="Figure width in inches for the output PDF.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=float(PAPER_SQUARE_FIGURE_SIZE_INCHES[1]),
        help="Figure height in inches for the output PDF.",
    )
    return parser.parse_args()


def load_model_frame(input_csv: Path) -> pd.DataFrame:
    """Load and validate the model dataframe.

    Args:
        input_csv: Path to the pre-filtered regression CSV.

    Returns:
        Dataframe with complete rows for required model columns.
    """
    required = [
        "delta_changed",
        "mean_logos_z",
        "mean_pathos_z",
        "mean_ethos_z",
        "baseline_belief_z",
    ]
    frame = pd.read_csv(input_csv)
    return frame.dropna(subset=required).copy()


def _result_term_names(result: Any) -> list[str]:
    """Return stable term names for a statsmodels fit result.

    Args:
        result: Statsmodels OLS result or robust-covariance result.

    Returns:
        Ordered term name list aligned with parameter vectors.
    """
    params = result.params
    if hasattr(params, "index"):
        return [str(term) for term in params.index]
    model = getattr(result, "model", None)
    if model is not None and hasattr(model, "exog_names"):
        return [str(term) for term in model.exog_names]
    return [f"term_{idx}" for idx in range(len(np.asarray(params, dtype=float)))]


def fit_model(
    frame: pd.DataFrame, *, cov_type: str, ci_level: float
) -> tuple[pd.DataFrame, int, float, float]:
    """Fit the slim model and return coefficient summary rows.

    Args:
        frame: Clean model dataframe.
        cov_type: Covariance type, or 'none' for classic OLS covariance.
        ci_level: Confidence interval level in (0, 1).

    Returns:
        Tuple of (coefficient table, nobs, r2, adjusted r2).
    """
    model = smf.ols(FORMULA, data=frame).fit()
    result = model
    if cov_type.strip().lower() != "none":
        result = model.get_robustcov_results(cov_type=cov_type)

    term_names = _result_term_names(result)
    params = np.asarray(result.params, dtype=float)
    std_err = np.asarray(result.bse, dtype=float)
    pvals = np.asarray(result.pvalues, dtype=float)
    ci_raw = np.asarray(result.conf_int(alpha=1.0 - ci_level), dtype=float)

    rows: list[dict[str, object]] = []
    for idx, term in enumerate(term_names):
        rows.append(
            {
                "term": term,
                "estimate": float(params[idx]),
                "std_err": float(std_err[idx]),
                "p_value": float(pvals[idx]),
                "ci_low": float(ci_raw[idx, 0]),
                "ci_high": float(ci_raw[idx, 1]),
            }
        )

    table = pd.DataFrame(rows)
    ordered = table[table["term"].isin(TERM_ORDER)].copy()
    ordered["term"] = pd.Categorical(
        ordered["term"], categories=TERM_ORDER, ordered=True
    )
    ordered = ordered.sort_values("term")
    ordered["term_label"] = ordered["term"].map(TERM_LABELS)
    return ordered, int(model.nobs), float(model.rsquared), float(model.rsquared_adj)


def load_overlay_table(
    overlay_summary_csv: Path,
) -> tuple[pd.DataFrame, int | None]:
    """Load optional overlay coefficients for shared rhetoric terms.

    Args:
        overlay_summary_csv: Path to a coefficient summary CSV.

    Returns:
        Tuple of filtered overlay dataframe and optional nobs value.
    """
    frame = pd.read_csv(overlay_summary_csv)
    required = ["term", "estimate", "ci_low", "ci_high"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Overlay summary missing required columns: {missing} "
            f"in {overlay_summary_csv}"
        )
    term_keep = list(OVERLAY_RHETORIC_TERMS)
    filtered = frame[frame["term"].isin(term_keep)].copy()
    filtered["term"] = pd.Categorical(
        filtered["term"], categories=TERM_ORDER, ordered=True
    )
    filtered = filtered.sort_values("term")
    filtered["term_label"] = filtered["term"].map(TERM_LABELS)
    nobs: int | None = None
    if "nobs" in frame.columns and not frame["nobs"].dropna().empty:
        nobs = int(frame["nobs"].dropna().iloc[0])
    return filtered, nobs


def _plot_primary_series(axis, table: pd.DataFrame, y_pos: np.ndarray) -> None:
    """Plot the primary coefficient series."""
    estimates = table["estimate"].to_numpy(dtype=float)
    ci_low = table["ci_low"].to_numpy(dtype=float)
    ci_high = table["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack((estimates - ci_low, ci_high - estimates))
    axis.errorbar(
        estimates,
        y_pos - 0.1,
        xerr=xerr,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.6,
        capsize=4.0,
        markersize=5.0,
        label="Ours",
    )


def _plot_overlay_series(
    axis,
    *,
    reference_table: pd.DataFrame,
    overlay: OverlayPlotData,
) -> None:
    """Plot optional overlay coefficients for shared terms."""
    if overlay.table.empty:
        return
    term_to_y = {term: pos for pos, term in enumerate(reference_table["term"].tolist())}
    overlay_frame = overlay.table[
        overlay.table["term"].map(lambda value: str(value) in term_to_y)
    ].copy()
    if overlay_frame.empty:
        return
    overlay_positions = overlay_frame["term"].map(lambda value: term_to_y[str(value)])
    overlay_estimates = overlay_frame["estimate"].to_numpy(dtype=float)
    overlay_low = overlay_frame["ci_low"].to_numpy(dtype=float)
    overlay_high = overlay_frame["ci_high"].to_numpy(dtype=float)
    overlay_xerr = np.vstack(
        (overlay_estimates - overlay_low, overlay_high - overlay_estimates)
    )
    axis.errorbar(
        overlay_estimates,
        overlay_positions.to_numpy(dtype=float) + 0.1,
        xerr=overlay_xerr,
        fmt="s",
        color="#d62728",
        ecolor="#d62728",
        elinewidth=1.3,
        capsize=3.5,
        markersize=4.3,
        label=overlay.label,
    )


def save_plot(
    table: pd.DataFrame,
    *,
    output_pdf: Path,
    overlay: OverlayPlotData | None,
    fig_width: float,
    fig_height: float,
) -> None:
    """Render and save the horizontal coefficient plot.

    Args:
        table: Coefficient table with confidence bounds.
        output_pdf: Destination PDF path.
        overlay: Optional overlay coefficient dataset.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.

    Returns:
        None.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    y_pos = np.arange(len(table), dtype=float)
    fig, axis = plt.subplots(figsize=(float(fig_width), float(fig_height)))
    axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.9)
    _plot_primary_series(axis, table, y_pos)
    if overlay is not None:
        _plot_overlay_series(
            axis,
            reference_table=table,
            overlay=overlay,
        )
    axis.set_yticks(y_pos)
    axis.set_yticklabels(table["term_label"].tolist())
    axis.invert_yaxis()
    axis.set_xlabel("Coefficient estimate")
    if overlay is not None and not overlay.table.empty:
        axis.legend(loc="best", fontsize=7.6, frameon=True)
    axis.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def main() -> None:
    """Run the test coefficient fit and plotting pipeline.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    frame = load_model_frame(args.input_csv)
    table, nobs, r2, adj_r2 = fit_model(
        frame, cov_type=args.cov_type, ci_level=args.ci_level
    )
    overlay_table: pd.DataFrame | None = None
    overlay_nobs: int | None = None
    overlay_data: OverlayPlotData | None = None
    if args.overlay_summary_csv is not None:
        overlay_table, overlay_nobs = load_overlay_table(args.overlay_summary_csv)
        overlay_data = OverlayPlotData(
            table=overlay_table,
            label=str(args.overlay_label),
            nobs=overlay_nobs,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    save_plot(
        table,
        output_pdf=args.output_pdf,
        overlay=overlay_data,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
    )

    print(f"model_formula: {FORMULA}")
    print(f"nobs={nobs} r2={r2:.6f} adj_r2={adj_r2:.6f}")
    if args.overlay_summary_csv is not None:
        print(f"overlay_summary_csv: {args.overlay_summary_csv}")
        if overlay_nobs is not None:
            print(f"overlay_nobs={overlay_nobs}")
    print(f"wrote_csv: {args.output_csv}")
    print(f"wrote_pdf: {args.output_pdf}")


if __name__ == "__main__":
    main()
