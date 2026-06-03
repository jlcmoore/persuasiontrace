"""
Plotting utilities for mechanism sample-size curve results.

This module reads the CSV produced by ``scripts/mechanism/sample_size_curve.py``,
aggregates metrics per dialogue-count ``N``, and produces simple curves for
validation performance:

- R^2 versus N
- RMSE versus N

Usage
-----
python -m analysis.mechanism_sample_size_plot \
    --csv analysis/mechanism_sample_size_curve.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plot_blocks import DEFAULT_FIG_DIR, save_or_show

FIG_DIR = DEFAULT_FIG_DIR


def load_sample_size_results(csv_path: str | Path) -> pd.DataFrame:
    """
    Load per-run sample-size metrics from CSV.

    The CSV is expected to contain at least the following columns:
    - N: number of dialogues used for training
    - rmse: held-out RMSE
    - r2: held-out R^2
    - seed: random seed identifier (used for counting runs per N)
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample-size CSV not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"N", "rmse", "r2"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    return df


def aggregate_sample_size_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run metrics to per-N summary statistics.

    The returned dataframe has one row per N with:
    - rmse_mean, rmse_std
    - r2_mean, r2_std
    - n_runs: number of runs contributing to the summary
    """

    if df.empty:
        return df.copy()

    grouped = (
        df.groupby("N", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_runs=("seed", "count") if "seed" in df.columns else ("rmse", "count"),
        )
        .sort_values("N")
    )
    return grouped


def _plot_metric_vs_n(
    *,
    df: pd.DataFrame,
    n_col: str,
    mean_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    filename: str,
    show: bool,
) -> None:
    """Draw a mean-with-95%-CI curve for one metric."""

    if df.empty:
        return

    xs = df[n_col].to_numpy(dtype=float)
    ys = df[mean_col].to_numpy(dtype=float)
    std = df[std_col].to_numpy(dtype=float)
    n_runs = df.get("n_runs", pd.Series(np.ones_like(std), index=df.index)).to_numpy(
        dtype=float
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.divide(std, np.sqrt(n_runs), out=np.zeros_like(std), where=n_runs > 0)
    yerr = 1.96 * se

    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw central curve
    ax.plot(xs, ys, marker="o", color="#2563EB", linewidth=1.8, label="Mean")

    # Error band: mean ± 95% confidence interval
    if np.any(np.isfinite(yerr)):
        lower = ys - yerr
        upper = ys + yerr
        ax.fill_between(
            xs,
            lower,
            upper,
            color="#93C5FD",
            alpha=0.35,
            label="Mean ± 95% CI",
        )

    ax.set_xlabel("Number of dialogues (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.legend()

    out_path = FIG_DIR / filename
    save_or_show(fig, out_path, show)


def plot_sample_size_curves(
    df_runs: pd.DataFrame,
    show: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-run metrics and produce R^2/RMSE versus N curves.

    Returns a tuple of:
    - df_runs: the original per-run dataframe (unchanged)
    - df_summary: the aggregated per-N summary dataframe
    """

    if df_runs.empty:
        return df_runs, df_runs.copy()

    df_summary = aggregate_sample_size_results(df_runs)

    # R^2 vs N (higher is better)
    _plot_metric_vs_n(
        df=df_summary,
        n_col="N",
        mean_col="r2_mean",
        std_col="r2_std",
        ylabel="Validation R\u00b2",
        title="Validation R\u00b2 versus number of dialogues",
        filename="08_sample_size_r2_vs_n.pdf",
        show=show,
    )

    # RMSE vs N (lower is better)
    _plot_metric_vs_n(
        df=df_summary,
        n_col="N",
        mean_col="rmse_mean",
        std_col="rmse_std",
        ylabel="Validation RMSE",
        title="Validation RMSE versus number of dialogues",
        filename="09_sample_size_rmse_vs_n.pdf",
        show=show,
    )

    return df_runs, df_summary


def main() -> None:
    """CLI entry point for plotting sample-size curves from CSV."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot validation R^2 and RMSE versus N from "
            "mechanism_sample_size_curve CSV results."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="analysis/mechanism_sample_size_curve.csv",
        help="Path to the per-run sample-size metrics CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    args = parser.parse_args()

    sns.set_style("whitegrid")

    df_runs = load_sample_size_results(args.csv)
    if df_runs.empty:
        print("Sample-size CSV is empty – nothing to plot.")
        return

    _, df_summary = plot_sample_size_curves(df_runs, show=args.show)

    print(
        "Plotted sample-size curves for "
        f"{df_summary['N'].nunique()} N values "
        f"({int(df_runs.shape[0])} total runs)."
    )


if __name__ == "__main__":
    main()
