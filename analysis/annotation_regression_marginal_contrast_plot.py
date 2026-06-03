"""
Plot model-based marginal means and contrasts for persuasion by condition.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis.plotting_utils import (
    add_dodge_argument,
    build_condition_axis_data,
    build_x_positions,
    dodge_positions,
    load_boolean_participant_df,
    load_condition_df,
    plot_errorbar,
    plot_persuader_errorbar_series,
    save_figure,
)


def _set_axis_ylim_from_ci(
    axis: plt.Axes,
    *,
    ci_frame: pd.DataFrame,
    ymin: float,
    ymax: float,
) -> None:
    """
    Set y-axis limits, auto-padding around confidence intervals when requested.

    Args:
        axis: Matplotlib axis to update.
        ci_frame: DataFrame containing ``conf_low`` and ``conf_high`` columns.
        ymin: Requested minimum y-axis value.
        ymax: Requested maximum y-axis value.
    """
    if ymin == -1.0 and ymax == 1.0:
        min_ci = ci_frame["conf_low"].min()
        max_ci = ci_frame["conf_high"].max()
        if pd.notna(min_ci) and pd.notna(max_ci):
            pad = 0.05 * max(1.0, abs(max_ci - min_ci))
            axis.set_ylim(min_ci - pad, max_ci + pad)
            return
    axis.set_ylim(ymin, ymax)


def _configure_condition_axis(
    axis: plt.Axes,
    *,
    x_positions: dict[bool, float],
    label_map: dict[bool, str],
    order: list[bool],
    ci_frame: pd.DataFrame,
    ymin: float,
    ymax: float,
) -> None:
    """
    Apply shared axis styling for condition-based regression plots.

    Args:
        axis: Matplotlib axis to update.
        x_positions: Mapping of condition key to x position.
        label_map: Mapping of condition key to display label.
        order: Condition display order.
        ci_frame: DataFrame containing confidence interval bounds.
        ymin: Requested minimum y-axis value.
        ymax: Requested maximum y-axis value.
    """
    axis.set_xticks(list(x_positions.values()))
    axis.set_xticklabels([label_map[val] for val in order])
    axis.set_ylabel("Predicted persuasion delta")
    axis.set_xlabel("Condition")
    axis.axhline(0, color="#222222", linewidth=1, alpha=0.4)
    _set_axis_ylim_from_ci(axis, ci_frame=ci_frame, ymin=ymin, ymax=ymax)
    axis.legend(frameon=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the interaction plot.

    Returns:
        Parsed arguments with CSV input and plot output paths.
    """

    parser = argparse.ArgumentParser(
        description="Plot marginal means and contrasts by condition."
    )
    parser.add_argument(
        "--marginal-csv",
        default="analysis/data/annotation_regression_marginals.csv",
        help="Marginal means CSV path.",
    )
    parser.add_argument(
        "--plot-path",
        default="analysis/figures/annotation_regression_interaction.pdf",
        help="Output path for the interaction plot.",
    )
    parser.add_argument(
        "--contrast-csv",
        default="analysis/data/annotation_regression_contrasts.csv",
        help="Contrast CSV path.",
    )
    parser.add_argument(
        "--contrast-plot-path",
        default="analysis/figures/annotation_regression_contrast.pdf",
        help="Output path for the contrast plot.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=4.5,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=3.5,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=-1.0,
        help="Minimum y-axis value.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Maximum y-axis value.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with N and df values.",
    )
    parser.add_argument(
        "--llm-label",
        default="GPT-5",
        help="Legend label for the LLM persuader.",
    )
    add_dodge_argument(parser)
    return parser.parse_args()


def load_marginals(csv_path: str) -> pd.DataFrame:
    """Load marginal means CSV and normalize fields.

    Args:
        csv_path: Path to marginal means CSV.

    Returns:
        Normalized DataFrame ready for plotting.
    """

    return load_condition_df(csv_path)


def load_contrasts(csv_path: str) -> pd.DataFrame:
    """Load contrast CSV and normalize fields.

    Args:
        csv_path: Path to contrast CSV.

    Returns:
        Normalized DataFrame ready for plotting.
    """

    return load_boolean_participant_df(csv_path)


def plot_interaction(
    df: pd.DataFrame,
    output_path: str,
    ymin: float,
    ymax: float,
    annotate: bool,
    width: float,
    height: float,
    dodge: float,
    llm_label: str,
) -> None:
    """Plot marginal means with confidence intervals.

    Args:
        df: Marginal means DataFrame.
        output_path: Output figure path.
        llm_label: Legend label for LLM series.

    Returns:
        None.
    """

    if df.empty:
        return

    label_map, order, x_positions = build_condition_axis_data()
    df["participant_label"] = df["participant_proposition"].map(label_map)

    colors = {"human": "#1f77b4", "llm": "#d62728"}
    series_labels = {"human": "Human", "llm": llm_label}
    fig, axis = plt.subplots(figsize=(width, height))

    for idx, persuader_type in enumerate(["human", "llm"]):
        subset = df[df["persuader_type"] == persuader_type]
        if subset.empty:
            continue
        offset = 0.0
        if dodge != 0.0:
            offset = -dodge if idx == 0 else dodge
        x_vals, y_vals = plot_persuader_errorbar_series(
            axis,
            frame=subset,
            x_positions=x_positions,
            offset=offset,
            color=colors.get(persuader_type, "#333333"),
            label=series_labels.get(persuader_type, persuader_type),
        )
        if annotate and "n" in subset.columns and "df" in subset.columns:
            for x_val, y_val, n_val, df_val in zip(
                x_vals,
                y_vals,
                subset["n"].tolist(),
                subset["df"].tolist(),
            ):
                axis.annotate(
                    f"N={int(n_val)} df={df_val:.1f}",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color=colors.get(persuader_type, "#333333"),
                )

    _configure_condition_axis(
        axis,
        x_positions=x_positions,
        label_map=label_map,
        order=order,
        ci_frame=df,
        ymin=ymin,
        ymax=ymax,
    )
    fig.tight_layout()

    save_figure(fig, output_path)


def plot_contrasts(
    df: pd.DataFrame,
    output_path: str,
    ymin: float,
    ymax: float,
    width: float,
    height: float,
) -> None:
    """Plot LLM-vs-human contrast with confidence intervals.

    Args:
        df: Contrast DataFrame.
        output_path: Output figure path.
        ymin: Minimum y-axis value.
        ymax: Maximum y-axis value.
        width: Figure width in inches.
        height: Figure height in inches.

    Returns:
        None.
    """

    if df.empty:
        return

    label_map = {False: "Generic", True: "Ppt. Prop."}
    order = [False, True]
    x_positions = build_x_positions(order)
    df = df.sort_values("participant_proposition")
    x_vals = dodge_positions(
        df["participant_proposition"].tolist(),
        x_positions,
        0.0,
    )
    y_vals = df["estimate"].astype(float).tolist()
    yerr_low = (df["estimate"] - df["conf_low"]).astype(float).tolist()
    yerr_high = (df["conf_high"] - df["estimate"]).astype(float).tolist()

    fig, axis = plt.subplots(figsize=(width, height))
    plot_errorbar(
        axis,
        x_vals,
        y_vals,
        yerr_low,
        yerr_high,
        "#2ca02c",
        "LLM - Human",
    )
    _configure_condition_axis(
        axis,
        x_positions=x_positions,
        label_map=label_map,
        order=order,
        ci_frame=df,
        ymin=ymin,
        ymax=ymax,
    )
    fig.tight_layout()
    save_figure(fig, output_path)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    df = load_marginals(args.marginal_csv)
    plot_interaction(
        df,
        args.plot_path,
        args.ymin,
        args.ymax,
        args.annotate,
        args.width,
        args.height,
        args.dodge,
        args.llm_label,
    )
    contrast_df = load_contrasts(args.contrast_csv)
    plot_contrasts(
        contrast_df,
        args.contrast_plot_path,
        args.ymin,
        args.ymax,
        args.width,
        args.height,
    )


if __name__ == "__main__":
    main()
