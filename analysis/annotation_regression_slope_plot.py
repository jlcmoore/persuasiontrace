"""
Plot rhetoric slopes by persuader and participant proposition.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis.plotting_utils import (
    add_dodge_argument,
    build_condition_axis_data,
    load_condition_df,
    plot_persuader_errorbar_series,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the slope plot.

    Returns:
        Parsed arguments with CSV input and plot output paths.
    """

    parser = argparse.ArgumentParser(
        description="Plot pathos/ethos slopes by persuader and participant proposition."
    )
    parser.add_argument(
        "--slopes-csv",
        default="analysis/data/annotation_regression_slopes.csv",
        help="Slope CSV path.",
    )
    parser.add_argument(
        "--plot-all-path",
        default="analysis/figures/annotation_regression_slopes.pdf",
        help="Output path for the all-features slope plot.",
    )
    parser.add_argument(
        "--plot-focus-path",
        default="analysis/figures/annotation_regression_ethos_slope.pdf",
        help="Output path for the focused slope plot.",
    )
    parser.add_argument(
        "--focus-feature",
        default="mean_ethos_z",
        help="Feature name for the focused slope plot.",
    )
    add_dodge_argument(parser)
    parser.add_argument(
        "--llm-label",
        default="GPT-5",
        help="Legend label for the LLM persuader.",
    )
    return parser.parse_args()


def load_slopes(csv_path: str) -> pd.DataFrame:
    """Load slopes CSV and normalize fields.

    Args:
        csv_path: Path to slope CSV.

    Returns:
        Normalized DataFrame ready for plotting.
    """

    return load_condition_df(csv_path, feature_col="feature")


def plot_slopes(
    df: pd.DataFrame,
    output_path: str,
    features: list[str],
    title_map: dict[str, str],
    dodge: float,
    llm_label: str,
) -> None:
    """Plot slopes with confidence intervals for selected features.

    Args:
        df: Slopes DataFrame.
        output_path: Output figure path.
        features: Ordered list of feature names to plot.
        title_map: Map from feature name to plot title.
        llm_label: Legend label for LLM series.

    Returns:
        None.
    """

    if df.empty:
        return

    label_map, order, x_positions = build_condition_axis_data()
    offsets = {"human": -dodge, "llm": dodge}
    colors = {"human": "#1f77b4", "llm": "#d62728"}
    series_labels = {"human": "Human", "llm": llm_label}

    col_count = 2
    row_count = (len(features) + col_count - 1) // col_count
    fig, axes = plt.subplots(
        row_count, col_count, figsize=(10, 3.4 * row_count), sharey=False
    )
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, feature in zip(axes_list, features):
        subset = df[df["feature"] == feature]
        if subset.empty:
            axis.set_axis_off()
            continue
        for persuader_type in ["human", "llm"]:
            group = subset[subset["persuader_type"] == persuader_type]
            if group.empty:
                continue
            plot_persuader_errorbar_series(
                axis,
                frame=group,
                x_positions=x_positions,
                offset=offsets.get(persuader_type, 0.0),
                color=colors.get(persuader_type, "#333333"),
                label=series_labels.get(persuader_type, persuader_type),
            )
        axis.axhline(0, color="#222222", linewidth=1, alpha=0.4)
        axis.set_xticks(list(x_positions.values()))
        axis.set_xticklabels([label_map[val] for val in order])
        axis.set_title(title_map.get(feature, feature))
        axis.set_xlabel("Condition")
        axis.set_ylabel("Slope (per 1 SD)")

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()

    save_figure(fig, output_path)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    df = load_slopes(args.slopes_csv)
    title_map = {
        "mean_logos_z": "Logos slope",
        "mean_pathos_z": "Pathos slope",
        "mean_ethos_z": "Ethos slope",
    }
    all_features = [
        "mean_logos_z",
        "mean_pathos_z",
        "mean_ethos_z",
    ]
    plot_slopes(
        df, args.plot_all_path, all_features, title_map, args.dodge, args.llm_label
    )
    plot_slopes(
        df,
        args.plot_focus_path,
        [args.focus_feature],
        title_map,
        args.dodge,
        args.llm_label,
    )


if __name__ == "__main__":
    main()
