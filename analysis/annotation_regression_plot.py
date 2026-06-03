"""
Plot forest charts for annotation regression summaries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from analysis.annotation_regression_utils import add_summary_csv_arg
from analysis.formatting import condition_color_map, split_condition_label
from annotation.records import ANNOTATION_CODES


@dataclass(frozen=True)
class ForestGroupInfo:
    """Metadata for grouped forest plot rendering.

    Parameters:
        term_labels: Ordered term labels to plot.
        groups: Group labels for each model subset.
        offsets: Vertical offsets for each group.
        color_map: Mapping from group label to color.

    Returns:
        None.
    """

    term_labels: list[str]
    groups: list[str]
    offsets: list[float]
    color_map: dict[str, str]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for forest plotting.

    Parameters:
        None.

    Returns:
        Parsed argparse.Namespace with summary_csv and plot_path.
    """

    parser = argparse.ArgumentParser(
        description="Plot forest charts from annotation regression summaries."
    )
    add_summary_csv_arg(parser)
    parser.add_argument(
        "--plot-path",
        default="analysis/figures/annotation_regression_forest.pdf",
        help="Output path for the forest plot.",
    )
    return parser.parse_args()


def normalize_group_label(label: str) -> str:
    """Normalize a model label into a group label.

    Parameters:
        label: The model label from the summary CSV.

    Returns:
        A normalized group label for plotting.
    """

    if not label:
        return "overall"
    if label == "overall":
        return "overall"
    if label.startswith("condition:"):
        return label.split("condition:", 1)[1]
    return label


def ordered_terms(term_labels: Sequence[str]) -> list[str]:
    """Order term labels using the canonical annotation code order.

    Parameters:
        term_labels: Term labels present in the summary CSV.

    Returns:
        Ordered list of term labels for plotting.
    """

    canonical = list(ANNOTATION_CODES)
    canonical_set = set(canonical)
    ordered = [label for label in canonical if label in term_labels]
    remaining = sorted(label for label in term_labels if label not in canonical_set)
    return ordered + remaining


def build_offsets(group_count: int) -> list[float]:
    """Return symmetric offsets for grouped points.

    Parameters:
        group_count: Number of grouped model series in the plot.

    Returns:
        List of y-axis offsets for each group.
    """

    if group_count <= 1:
        return [0.0]
    span = 0.4
    step = span / (group_count - 1)
    return [-span / 2 + idx * step for idx in range(group_count)]


def load_forest_plot_data(summary_csv: str) -> pd.DataFrame:
    """Load regression summary data for forest plotting.

    Parameters:
        summary_csv: Path to the regression summary CSV.

    Returns:
        DataFrame containing filtered and normalized summary rows.
    """

    summary_df = pd.read_csv(summary_csv)
    summary_df = summary_df[summary_df["term"] != "__error__"].copy()
    summary_df = summary_df[summary_df["term"].str.startswith("mean_")]
    summary_df["estimate"] = pd.to_numeric(summary_df["estimate"], errors="coerce")
    summary_df["std_err"] = pd.to_numeric(summary_df["std_err"], errors="coerce")
    summary_df = summary_df.dropna(subset=["estimate", "std_err"])
    summary_df["term_label"] = (
        summary_df["term"]
        .str.replace("mean_", "", regex=False)
        .str.replace("_z", "", regex=False)
    )
    summary_df["group_label"] = summary_df["model_label"].apply(normalize_group_label)
    return summary_df


def prepare_forest_plot_df(summary_csv: str) -> pd.DataFrame:
    """Load and prepare regression summary data for plotting.

    Parameters:
        summary_csv: Path to the regression summary CSV.

    Returns:
        DataFrame with confidence interval columns added.
    """

    plot_df = load_forest_plot_data(summary_csv)
    if plot_df.empty:
        return plot_df
    plot_df = plot_df.copy()
    plot_df["ci_low"] = plot_df["estimate"] - 1.96 * plot_df["std_err"]
    plot_df["ci_high"] = plot_df["estimate"] + 1.96 * plot_df["std_err"]
    return plot_df


def build_forest_groups(plot_df: pd.DataFrame) -> ForestGroupInfo:
    """Build group metadata for forest plotting.

    Parameters:
        plot_df: Prepared plotting DataFrame.

    Returns:
        ForestGroupInfo containing labels, groups, offsets, and colors.
    """

    term_labels = ordered_terms(plot_df["term_label"].unique().tolist())
    group_labels = sorted(
        {label for label in plot_df["group_label"].unique() if label != "overall"}
    )
    groups = ["overall"] + group_labels
    offsets = build_offsets(len(groups))
    color_map = condition_color_map(group_labels)
    return ForestGroupInfo(
        term_labels=term_labels,
        groups=groups,
        offsets=offsets,
        color_map=color_map,
    )


def collect_group_points(
    subset: pd.DataFrame,
    term_labels: Sequence[str],
    offset: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Collect plot points for a group.

    Parameters:
        subset: DataFrame subset for a single group.
        term_labels: Ordered term labels for the y-axis.
        offset: Vertical offset for the group.

    Returns:
        Tuple of x-values, y-values, lower errors, and upper errors.
    """

    x_vals: list[float] = []
    y_vals: list[float] = []
    xerr_low: list[float] = []
    xerr_high: list[float] = []
    for term_idx, term in enumerate(term_labels):
        row = subset[subset["term_label"] == term]
        if row.empty:
            continue
        entry = row.iloc[0]
        x_vals.append(float(entry["estimate"]))
        y_vals.append(term_idx + offset)
        xerr_low.append(float(entry["estimate"] - entry["ci_low"]))
        xerr_high.append(float(entry["ci_high"] - entry["estimate"]))
    return x_vals, y_vals, xerr_low, xerr_high


def build_forest_figure(term_count: int) -> tuple[plt.Figure, plt.Axes]:
    """Create a matplotlib figure for the forest plot.

    Parameters:
        term_count: Number of terms to plot.

    Returns:
        Tuple of matplotlib Figure and Axes.
    """

    fig_height = max(4, term_count * 1.2)
    fig, axis = plt.subplots(figsize=(10, fig_height))
    return fig, axis


def plot_forest_groups(
    axis: plt.Axes,
    plot_df: pd.DataFrame,
    group_info: ForestGroupInfo,
) -> None:
    """Plot grouped estimates onto a forest plot axis.

    Parameters:
        axis: Matplotlib axes to draw on.
        plot_df: Prepared plotting DataFrame.
        group_info: Group metadata including colors and offsets.

    Returns:
        None.
    """

    overall_color = "#222222"
    for group_idx, group in enumerate(group_info.groups):
        subset = plot_df[plot_df["group_label"] == group]
        if subset.empty:
            continue
        x_vals, y_vals, xerr_low, xerr_high = collect_group_points(
            subset, group_info.term_labels, group_info.offsets[group_idx]
        )
        if not x_vals:
            continue
        color = (
            overall_color
            if group == "overall"
            else group_info.color_map.get(group, "#555555")
        )
        label = "Overall" if group == "overall" else split_condition_label(group)
        axis.errorbar(
            x_vals,
            y_vals,
            xerr=[xerr_low, xerr_high],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            label=label,
        )


def build_legend_handles(
    axis: plt.Axes, group_info: ForestGroupInfo
) -> tuple[list[tuple[object, str]], list[tuple[object, str]]]:
    """Collect legend handles for overall and condition entries.

    Parameters:
        axis: Matplotlib axes to read legend handles from.
        group_info: Group metadata with label ordering.

    Returns:
        Tuple of overall entries and condition entries.
    """

    handles, labels = axis.get_legend_handles_labels()
    handle_map = dict(zip(labels, handles))
    overall_label = "Overall"
    overall_handle = handle_map.get(overall_label)
    overall_entries: list[tuple[object, str]] = []
    if overall_handle is not None:
        overall_entries.append((overall_handle, overall_label))

    condition_entries: list[tuple[object, str]] = []
    for group in group_info.groups:
        if group == "overall":
            continue
        label = split_condition_label(group)
        handle = handle_map.get(label)
        if handle is not None:
            condition_entries.append((handle, label))
    return overall_entries, condition_entries


def apply_forest_legends(
    axis: plt.Axes, overall_entries: Sequence[tuple[object, str]], condition_entries
) -> None:
    """Attach stacked legends under the forest plot.

    Parameters:
        axis: Matplotlib axes to attach legends to.
        overall_entries: Legend entries for the overall series.
        condition_entries: Legend entries for condition series.

    Returns:
        None.
    """

    if overall_entries:
        overall_handles, overall_labels = zip(*overall_entries)
        legend_overall = axis.legend(
            overall_handles,
            overall_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            frameon=False,
            ncol=1,
        )
        axis.add_artist(legend_overall)

    if condition_entries:
        cond_handles, cond_labels = zip(*condition_entries)
        axis.legend(
            cond_handles,
            cond_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            frameon=False,
            ncol=2,
        )


def style_forest_axis(
    axis: plt.Axes,
    term_labels: Sequence[str],
    group_info: ForestGroupInfo,
) -> None:
    """Apply labels and styling to a forest plot axis.

    Parameters:
        axis: Matplotlib axes to style.
        term_labels: Ordered term labels for y-axis ticks.
        group_info: Group metadata for legend ordering.

    Returns:
        None.
    """

    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(range(len(term_labels)))
    axis.set_yticklabels(term_labels)
    axis.set_xlabel("Estimate (95% CI)")
    axis.set_title("Annotation regression coefficients")
    overall_entries, condition_entries = build_legend_handles(axis, group_info)
    apply_forest_legends(axis, overall_entries, condition_entries)


def save_forest_figure(fig: plt.Figure, output_path: str) -> None:
    """Save the forest plot figure to disk.

    Parameters:
        fig: Matplotlib figure to save.
        output_path: Output file path for the plot image.

    Returns:
        None.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_forest(summary_csv: str, output_path: str) -> None:
    """Write a forest plot for regression coefficients.

    Parameters:
        summary_csv: Path to the regression summary CSV.
        output_path: Output path for the plot image.

    Returns:
        None.
    """

    summary_path = Path(summary_csv)
    if not summary_path.exists():
        return
    plot_df = prepare_forest_plot_df(summary_csv)
    if plot_df.empty:
        return

    group_info = build_forest_groups(plot_df)
    fig, axis = build_forest_figure(len(group_info.term_labels))
    plot_forest_groups(axis, plot_df, group_info)
    style_forest_axis(axis, group_info.term_labels, group_info)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save_forest_figure(fig, output_path)


def main() -> None:
    """CLI entrypoint for forest plotting.

    Parameters:
        None.

    Returns:
        None.
    """

    args = parse_args()
    plot_forest(args.summary_csv, args.plot_path)


if __name__ == "__main__":
    main()
