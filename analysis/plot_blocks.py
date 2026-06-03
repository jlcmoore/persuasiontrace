"""Plotting blocks for round-level persuasion analyses."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from experiment import ContinuousMeasure

from .data_loading import persuader_relative_name
from .formatting import condition_color_map, split_condition_label

SEG_OFFSET = 1
YLABEL_SUPPORT = "Support (0 ... 1)"
YLABEL_SUPPORT_P_REL = YLABEL_SUPPORT + "\n(in direction of persuader)"
DEFAULT_FIG_DIR = Path(__file__).parent / "figures"


def save_or_show(fig, path: Path, show: bool) -> None:
    """Save a figure and optionally show it interactively.

    Args:
        fig: Matplotlib figure to save.
        path: Output path.
        show: Whether to show interactively.

    Returns:
        None.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(path, dpi=300)
    print(f"Saved figure: {path.resolve()}")
    if show:
        fig.show()
    plt.close(fig)


def plot_proposition_hist(
    df: pd.DataFrame,
    show: bool,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot proposition frequency across all loaded rounds.

    Args:
        df: Analysis dataframe.
        show: Whether to show figures interactively.
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    cnts = (
        df["proposition"]
        .value_counts()
        .rename_axis("proposition")
        .reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cnts, x="count", y="proposition", ax=ax, color="#3B82F6")
    ax.set_title("Frequency of propositions across all rounds")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    save_or_show(fig, fig_dir / "proposition_hist.pdf", show)


def plot_condition_bars(
    df: pd.DataFrame,
    show: bool,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot mean persuader-direction belief delta by condition.

    Args:
        df: Analysis dataframe.
        show: Whether to show figures interactively.
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    order = (
        df.groupby("condition")["delta_dir"].mean().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = condition_color_map(order)
    sns.barplot(
        data=df,
        x="delta_dir",
        y="condition",
        hue="condition",
        order=order,
        errorbar="se",
        palette=palette,
        ax=ax,
        orient="h",
        legend=False,
    )
    ax.set(
        title="Mean belief change (signed, in direction of persuader)",
        xlabel="Δ belief (-1 ... 1)",
        ylabel="",
    )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([split_condition_label(cond) for cond in order])

    summary_stats = df.groupby("condition")["delta_dir"].agg(["mean", "count", "std"])
    summary_stats["se"] = summary_stats["std"] / np.sqrt(
        summary_stats["count"].clip(lower=1)
    )
    summary_stats["se"] = summary_stats["se"].fillna(0.0)
    min_bound = float((summary_stats["mean"] - summary_stats["se"]).min())
    max_bound = float((summary_stats["mean"] + summary_stats["se"]).max())
    padding = 0.05
    lower = max(-1.0, min_bound - padding)
    upper = min(1.0, max_bound + padding)
    if upper - lower < 0.2:
        mid = (upper + lower) / 2
        lower = max(-1.0, mid - 0.1)
        upper = min(1.0, mid + 0.1)
    ax.set_xlim(lower, upper)
    for idx, cond in enumerate(order):
        mean_val = summary_stats.loc[cond, "mean"]
        count_val = int(summary_stats.loc[cond, "count"])
        ax.text(
            x=mean_val,
            y=idx,
            s=f"{mean_val:+.2f} (n={count_val})",
            va="center",
            ha="left" if mean_val < 0.95 else "right",
            color="black",
            fontsize=8,
        )

    save_or_show(fig, fig_dir / "condition_avg_change.pdf", show)


def plot_serial_questions(
    df: pd.DataFrame,
    show: bool,
    persuader_relative: bool = False,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot serial trajectories by condition.

    Args:
        df: Analysis dataframe.
        show: Whether to show figures interactively.
        persuader_relative: Whether to use persuader-relative values.
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    initial_name = persuader_relative_name("initial", persuader_relative)
    final_name = persuader_relative_name("final", persuader_relative)
    serial_name = persuader_relative_name("serial", persuader_relative)

    if serial_name not in df.columns or df[serial_name].isna().all():
        return

    serial_df = df[df["continuous"] == ContinuousMeasure.SERIAL_QUESTIONS].copy()
    if serial_df.empty:
        return

    for cond, sub in serial_df.groupby("condition"):
        fig, ax = plt.subplots(figsize=(6, 4))

        max_len = 0
        trajectories = []
        for _, row in sub.iterrows():
            serial_pts = row[serial_name] if isinstance(row[serial_name], list) else []
            y_values = [row[initial_name], *serial_pts, row[final_name]]
            x_values = list(range(len(y_values)))
            if y_values:
                ax.plot(x_values, y_values, alpha=0.3, color="grey")
                trajectories.append(y_values)
                max_len = max(max_len, len(y_values))

        if not trajectories:
            plt.close(fig)
            continue

        values_by_idx = [[] for _ in range(max_len)]
        for y_values in trajectories:
            for idx, val in enumerate(y_values):
                values_by_idx[idx].append(val)

        means = [np.mean(vals) if vals else math.nan for vals in values_by_idx]
        sems = [
            (np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            for vals in values_by_idx
        ]

        xs = list(range(len(means)))
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            color="#EF4444",
            lw=1.8,
            marker="o",
            capsize=3,
            label="mean",
        )

        ax.set_xticks(xs)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.minorticks_off()

        label = split_condition_label(cond)
        ax.set(
            xlabel="Messages received",
            ylabel=YLABEL_SUPPORT_P_REL if persuader_relative else YLABEL_SUPPORT,
            ylim=(0, 1),
            title=f"Serial-question trace\n{label}",
        )

        ax.legend()
        filename = f"serial_trace_{cond.replace(' ', '_').replace('/', '_')}.pdf"
        save_or_show(fig, fig_dir / filename, show)


def plot_serial_mean_all(
    df: pd.DataFrame,
    show: bool,
    persuader_relative: bool = False,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot condition-wise mean serial trajectories.

    Args:
        df: Analysis dataframe.
        show: Whether to show figures interactively.
        persuader_relative: Whether to use persuader-relative values.
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    initial_name = persuader_relative_name("initial", persuader_relative)
    final_name = persuader_relative_name("final", persuader_relative)
    serial_name = persuader_relative_name("serial", persuader_relative)

    serial_df = df[df["continuous"] == ContinuousMeasure.SERIAL_QUESTIONS].copy()
    if serial_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    conditions = sorted(serial_df["condition"].dropna().unique().tolist())
    color_map = condition_color_map(conditions)

    global_max_len = 0
    message_lengths: list[int] = []
    max_ci = 0.0
    plotted_any = False

    for cond, sub in serial_df.groupby("condition"):
        color = color_map.get(cond, "#000000")
        valid_rows = [
            row for _, row in sub.iterrows() if isinstance(row[serial_name], list)
        ]
        if not valid_rows:
            continue

        max_len = max(len(row[serial_name]) for row in valid_rows) + 2
        global_max_len = max(global_max_len, max_len)
        message_lengths.extend([len(row[serial_name]) + 2 for row in valid_rows])

        values_by_idx = [[] for _ in range(max_len)]
        for row in valid_rows:
            y_values = [row[initial_name], *row[serial_name], row[final_name]]
            for idx, val in enumerate(y_values):
                values_by_idx[idx].append(val)

        means = [np.mean(vals) if vals else math.nan for vals in values_by_idx]
        sems = [
            (np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            for vals in values_by_idx
        ]
        for mean_val, sem_val in zip(means, sems):
            if math.isnan(mean_val):
                continue
            max_ci = max(max_ci, mean_val + sem_val)
        xs = list(range(len(means)))
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            label=split_condition_label(cond),
            color=color,
            lw=1.6,
            marker="o",
            capsize=3,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    max_x = global_max_len
    if message_lengths:
        mean_len = float(np.mean(message_lengths))
        std_len = float(np.std(message_lengths, ddof=0))
        cutoff = int(round(mean_len + 2 * std_len))
        max_x = max(2, min(global_max_len, cutoff))

    ax.set_xticks(list(range(max_x)))
    ax.set_xlim(-0.5, max_x - 0.5)
    ax.minorticks_off()

    ax.set(
        title="Mean serial-question trajectories across conditions",
        xlabel="Messages received",
        ylabel=YLABEL_SUPPORT_P_REL if persuader_relative else YLABEL_SUPPORT,
    )
    upper = min(1.0, max_ci + 0.05) if max_ci > 0 else 1.0
    ax.set_ylim(0, upper)
    ax.legend(
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_or_show(fig, fig_dir / "serial_means_all.pdf", show)


def plot_serial_first_vs_rest(
    serial_df: pd.DataFrame,
    show: bool,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot first-vs-rest serial deltas by condition.

    Args:
        serial_df: First-vs-rest dataframe from stats blocks.
        show: Whether to show figures interactively.
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    if serial_df.empty:
        return

    order = (
        serial_df.groupby("condition")["first_delta"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    label_map = (
        serial_df[["condition", "condition_label"]]
        .drop_duplicates()
        .set_index("condition")["condition_label"]
        .to_dict()
    )
    order_labels = [label_map[condition] for condition in order]

    long_df = pd.DataFrame(
        {
            "condition": np.repeat(serial_df["condition"].values, 2),
            "condition_label": np.repeat(serial_df["condition_label"].values, 2),
            "component": np.tile(["First", "Rest"], len(serial_df)),
            "value": np.concatenate(
                [serial_df["first_delta"].values, serial_df["rest_delta"].values]
            ),
        }
    )

    fig_width = max(7.0, 0.7 * len(order) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    palette = {"First": "#EF4444", "Rest": "#3B82F6"}

    sns.boxplot(
        data=long_df,
        x="value",
        y="condition_label",
        hue="component",
        order=order_labels,
        palette=palette,
        ax=ax,
        orient="h",
    )

    ax.axvline(0, color="black", alpha=0.2, linewidth=1)
    ax.set(
        title="First vs rest belief change by condition",
        xlabel="Change in support (persuader-relative)",
        ylabel="",
        xlim=(-1, 1),
    )
    ax.legend(title="")
    save_or_show(fig, fig_dir / "serial_first_turn_bars.pdf", show)


def plot_mouse_trace_segment_aligned(
    df: pd.DataFrame,
    show: bool,
    persuader_relative: bool = False,
    color_rounds: bool = True,
    draw_lines_within_segments: bool = True,
    normalize_time: bool = False,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot per-round mouse traces segmented by message index.

    Args:
        df: Analysis dataframe.
        show: Whether to show figures interactively.
        persuader_relative: Whether to use persuader-relative values.
        color_rounds: Whether to color each round separately.
        draw_lines_within_segments: Whether to connect points within segments.
        normalize_time: Whether to normalize each segment's local time to [0, 1].
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    initial_name = persuader_relative_name("initial", persuader_relative)
    final_name = persuader_relative_name("final", persuader_relative)
    mouse_trace_name = persuader_relative_name("mouse_trace", persuader_relative)

    trace_df = df[df["continuous"] == ContinuousMeasure.MOUSE_TRACE].copy()
    if trace_df.empty:
        return

    for cond, sub in trace_df.groupby("condition"):
        fig, ax = plt.subplots(figsize=(6, 4))
        traces = []
        max_segments = 0
        for _, row in sub.iterrows():
            trace = row[mouse_trace_name]
            if (
                not trace
                or not isinstance(trace, list)
                or not all(isinstance(seg, list) for seg in trace)
            ):
                continue
            if not all(
                all(
                    isinstance(pt, dict) and "timestamp" in pt and "position" in pt
                    for pt in seg
                )
                for seg in trace
            ):
                continue
            traces.append(
                {
                    "segments": trace,
                    initial_name: row.get(initial_name, np.nan),
                    final_name: row.get(final_name, np.nan),
                }
            )
            max_segments = max(max_segments, len(trace))

        if not traces:
            plt.close(fig)
            continue

        ref_durations = [0.0] * max_segments
        for seg_idx in range(max_segments):
            durations = []
            for item in traces:
                trace = item["segments"]
                if seg_idx >= len(trace) or not trace[seg_idx]:
                    continue
                segment = trace[seg_idx]
                xs = sorted(float(pt["timestamp"]) for pt in segment)
                if len(xs) >= 2:
                    durations.append(xs[-1] - xs[0])
            ref_durations[seg_idx] = max(durations) if durations else 0.0

        round_ids = list(range(len(traces)))
        color_map = {}
        if color_rounds and round_ids:
            palette = sns.color_palette("tab20", n_colors=max(20, len(round_ids)))
            color_map = {
                round_id: palette[i % len(palette)]
                for i, round_id in enumerate(round_ids)
            }

        for round_id, item in enumerate(traces):
            trace = item["segments"]
            round_color = color_map.get(round_id, "black") if color_rounds else "black"

            for seg_idx, segment in enumerate(trace):
                if not segment:
                    continue

                seg_sorted = sorted(segment, key=lambda pt: float(pt["timestamp"]))
                xs = [float(pt["timestamp"]) for pt in seg_sorted]
                ys = [float(pt["position"]) for pt in seg_sorted]

                start_t, end_t = xs[0], xs[-1]
                duration = end_t - start_t

                if normalize_time:
                    if duration <= 0:
                        x_vals = [SEG_OFFSET + seg_idx + 0.5] * len(xs)
                    else:
                        x_vals = [
                            SEG_OFFSET + seg_idx + (t - start_t) / duration for t in xs
                        ]
                else:
                    ref = ref_durations[seg_idx]
                    if ref <= 0:
                        x_vals = [SEG_OFFSET + seg_idx] * len(xs)
                    else:
                        x_vals = [
                            SEG_OFFSET
                            + seg_idx
                            + max(0.0, min((t - start_t) / ref, 1.0))
                            for t in xs
                        ]

                ax.scatter(x_vals, ys, s=8, alpha=0.35, color=round_color)
                if draw_lines_within_segments and len(x_vals) > 1:
                    ax.plot(x_vals, ys, alpha=0.45, linewidth=1.0, color=round_color)

            init_y = item[initial_name]
            fin_y = item[final_name]
            if pd.notna(init_y):
                ax.scatter(
                    [SEG_OFFSET - 1],
                    [init_y],
                    marker="s",
                    facecolors="white",
                    edgecolors=round_color,
                    linewidths=1.2,
                    s=40,
                    alpha=0.9,
                    zorder=3,
                )
            if pd.notna(fin_y):
                ax.scatter(
                    [SEG_OFFSET + len(trace)],
                    [fin_y],
                    marker="D",
                    facecolors=round_color,
                    edgecolors="black",
                    linewidths=0.8,
                    s=46,
                    alpha=0.9,
                    zorder=3,
                )

        for boundary in range(1, max_segments + 2):
            ax.axvline(boundary, color="black", alpha=0.08, linewidth=1)

        title = "Mouse-trace by message index"
        if normalize_time:
            title += " (normalized)"
        title += f"\n{split_condition_label(cond)}"
        ax.set(
            title=title,
            xlabel="Messages received",
            ylabel=YLABEL_SUPPORT_P_REL if persuader_relative else YLABEL_SUPPORT,
            ylim=(0, 1),
            xlim=(-0.05, max(1, max_segments) + 1.05),
        )
        ax.set_xticks(list(range(0, max_segments + SEG_OFFSET)))
        ax.minorticks_off()

        marker_legend = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                label="Initial belief",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                label="Final belief",
                markerfacecolor="black",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=6,
            ),
        ]
        ax.legend(handles=marker_legend, fontsize=8, loc="lower right")

        filename = (
            f"mouse_trace_by_segment_{cond.replace(' ', '_').replace('/', '_')}.pdf"
        )
        save_or_show(fig, fig_dir / filename, show)


def plot_mouse_mean_all_segment_aligned(
    df: pd.DataFrame,
    show: bool,
    persuader_relative: bool = False,
    bins_per_segment: int = 50,
    show_sem: bool = True,
    normalize_time: bool = False,
    prefer_points: bool = True,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot mean mouse-trace trajectories by condition and segment.

    Args:
        df: Analysis dataframe.
        show: Whether to show figures interactively.
        persuader_relative: Whether to use persuader-relative values.
        bins_per_segment: Number of bins used per segment.
        show_sem: Whether to show SEM bands.
        normalize_time: Whether to normalize each segment's local time to [0, 1].
        prefer_points: Whether to snap bins to observed points when present.
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    initial_name = persuader_relative_name("initial", persuader_relative)
    final_name = persuader_relative_name("final", persuader_relative)
    mouse_trace_name = persuader_relative_name("mouse_trace", persuader_relative)

    trace_df = df[df["continuous"] == ContinuousMeasure.MOUSE_TRACE].copy()
    if trace_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    conditions = sorted(trace_df["condition"].dropna().unique().tolist())
    color_map = condition_color_map(conditions)
    target_x_seg = np.linspace(0.0, 1.0, bins_per_segment)
    dx = target_x_seg[1] - target_x_seg[0] if bins_per_segment > 1 else 1.0
    edges = np.concatenate(
        (
            [target_x_seg[0] - dx / 2],
            (target_x_seg[:-1] + target_x_seg[1:]) / 2,
            [target_x_seg[-1] + dx / 2],
        )
    )

    global_max_segments = 0

    for cond, sub in trace_df.groupby("condition"):
        color = color_map.get(cond, "#000000")
        traces = []
        max_segments = 0
        for _, row in sub.iterrows():
            trace = row[mouse_trace_name]
            if (
                not trace
                or not isinstance(trace, list)
                or not all(isinstance(seg, list) for seg in trace)
            ):
                continue
            if not all(
                all(
                    isinstance(pt, dict) and "timestamp" in pt and "position" in pt
                    for pt in seg
                )
                for seg in trace
            ):
                continue
            traces.append(trace)
            max_segments = max(max_segments, len(trace))

        if not traces:
            continue

        global_max_segments = max(global_max_segments, max_segments)

        ref_durations = [0.0] * max_segments
        for seg_idx in range(max_segments):
            durations = []
            for trace in traces:
                if seg_idx >= len(trace) or not trace[seg_idx]:
                    continue
                xs = sorted(float(pt["timestamp"]) for pt in trace[seg_idx])
                if len(xs) >= 2:
                    durations.append(xs[-1] - xs[0])
            ref_durations[seg_idx] = max(durations) if durations else 0.0

        label_used = False

        for seg_idx in range(max_segments):
            resampled = []

            for trace in traces:
                if seg_idx >= len(trace) or not trace[seg_idx]:
                    continue
                segment = trace[seg_idx]
                if not segment:
                    continue

                seg_sorted = sorted(segment, key=lambda pt: float(pt["timestamp"]))
                xs = np.array(
                    [float(pt["timestamp"]) for pt in seg_sorted], dtype=float
                )
                ys = np.array([float(pt["position"]) for pt in seg_sorted], dtype=float)

                start_t, end_t = xs[0], xs[-1]
                duration = end_t - start_t

                if normalize_time:
                    if duration <= 0:
                        y_bins = np.full(bins_per_segment, ys[-1], dtype=float)
                        resampled.append(y_bins)
                        continue
                    ux = (xs - start_t) / duration
                    xmax = 1.0
                else:
                    ref = ref_durations[seg_idx]
                    if ref <= 0:
                        y_bins = np.full(bins_per_segment, np.nan, dtype=float)
                        y_bins[0] = ys[-1]
                        resampled.append(y_bins)
                        continue
                    ux = np.clip((xs - start_t) / ref, 0.0, 1.0)
                    xmax = ux[-1] if len(ux) else 0.0

                series = pd.Series(ys, index=ux).sort_index()
                series = series.loc[~series.index.duplicated(keep="last")]
                ux = series.index.to_numpy()
                uy = series.to_numpy()

                y_bins = np.full(bins_per_segment, np.nan, dtype=float)

                if len(ux) == 1:
                    if normalize_time:
                        y_bins[:] = uy[0]
                    else:
                        y_bins[0] = uy[0]
                    resampled.append(y_bins)
                    continue

                valid_mask = target_x_seg <= (xmax + 1e-9)
                if valid_mask.any():
                    y_bins[valid_mask] = np.interp(target_x_seg[valid_mask], ux, uy)

                if prefer_points:
                    in_span = ux <= (xmax + 1e-9)
                    if np.any(in_span):
                        ux_in = ux[in_span]
                        uy_in = uy[in_span]
                        bin_idx = np.searchsorted(edges, ux_in, side="right") - 1
                        bin_idx = np.clip(bin_idx, 0, bins_per_segment - 1)
                        y_bins[bin_idx] = uy_in

                resampled.append(y_bins)

            if not resampled:
                continue

            seg_stack = np.vstack(resampled)
            seg_mean = np.nanmean(seg_stack, axis=0)
            if show_sem and seg_stack.shape[0] > 1:
                seg_sem = np.nanstd(seg_stack, axis=0, ddof=1) / math.sqrt(
                    seg_stack.shape[0]
                )
            else:
                seg_sem = None

            x_vals = SEG_OFFSET + seg_idx + target_x_seg
            finite = np.isfinite(seg_mean)
            if seg_sem is not None:
                sem_finite = finite & np.isfinite(seg_sem)
                if sem_finite.any():
                    ax.fill_between(
                        x_vals[sem_finite],
                        (seg_mean - seg_sem)[sem_finite],
                        (seg_mean + seg_sem)[sem_finite],
                        color=color,
                        alpha=0.15,
                        linewidth=0,
                    )

            ax.plot(
                x_vals[finite],
                seg_mean[finite],
                color=color,
                label=(cond if not label_used else None),
                lw=1.8,
            )
            label_used = True

        init_mean = pd.to_numeric(sub[initial_name], errors="coerce").mean()
        fin_mean = pd.to_numeric(sub[final_name], errors="coerce").mean()
        if not math.isnan(init_mean):
            ax.scatter(
                [SEG_OFFSET - 1],
                [init_mean],
                marker="s",
                facecolors="white",
                edgecolors=color,
                linewidths=1.3,
                s=60,
                zorder=3,
            )
        if not math.isnan(fin_mean):
            ax.scatter(
                [SEG_OFFSET + max_segments],
                [fin_mean],
                marker="D",
                facecolors=color,
                edgecolors="black",
                linewidths=0.8,
                s=64,
                zorder=3,
            )

    for boundary in range(1, global_max_segments + 2):
        ax.axvline(boundary, color="black", alpha=0.08, linewidth=1)

    title = "Mean mouse-trace by message index"
    if normalize_time:
        title += " (normalized)"

    ax.set(
        title=title,
        xlabel="Messages received",
        ylabel=YLABEL_SUPPORT_P_REL if persuader_relative else YLABEL_SUPPORT,
        ylim=(0, 1),
        xlim=(-0.05, max(1, global_max_segments) + 1.05),
    )
    ax.set_xticks(list(range(0, global_max_segments + SEG_OFFSET)))

    cond_leg = ax.legend(fontsize=8, title="Conditions")
    ax.add_artist(cond_leg)
    marker_legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            label="Initial belief",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            label="Final belief",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
        ),
    ]
    ax.legend(handles=marker_legend, fontsize=8, loc="lower right")

    save_or_show(fig, fig_dir / "mouse_means_by_segment.pdf", show)


def plot_pre_post_boxes(
    df: pd.DataFrame,
    show: bool,
    persuader_relative: bool = False,
    *,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> None:
    """Plot condition-wise pre/post violins with participant trajectories.

    Args:
        df: Analysis dataframe.
        show: Whether to show figures interactively.
        persuader_relative: Whether to use persuader-relative values.
        fig_dir: Output figure directory.

    Returns:
        None.
    """
    initial_name = persuader_relative_name("initial", persuader_relative)
    final_name = persuader_relative_name("final", persuader_relative)

    if initial_name not in df.columns or final_name not in df.columns:
        return

    def stance_color(supports: bool | None) -> str:
        if supports is True:
            return "#1f77b4"
        if supports is False:
            return "#d62728"
        return "#6b7280"

    conditions = sorted(df["condition"].dropna().unique().tolist())

    if not conditions:
        return

    num_conditions = len(conditions)
    fig_width = max(7, 3.2 * num_conditions)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_conditions,
        figsize=(fig_width, 4.2),
        sharey=True,
    )
    if num_conditions == 1:
        axes = [axes]

    color_map = condition_color_map(conditions)
    jitter_scale = 0.06

    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond].copy()
        sub = sub[pd.notna(sub[initial_name]) & pd.notna(sub[final_name])]
        if sub.empty:
            ax.set_visible(False)
            continue

        pre_vals = sub[initial_name].astype(float).to_numpy()
        post_vals = sub[final_name].astype(float).to_numpy()
        mean_delta = float(sub["delta_dir"].mean())

        melt_df = pd.DataFrame(
            {
                "group": ["Pre"] * len(pre_vals) + ["Post"] * len(post_vals),
                "value": np.concatenate([pre_vals, post_vals]),
            }
        )

        base_color = color_map.get(cond, "#3B82F6")
        base_rgb = np.array(mcolors.to_rgb(base_color))
        pre_rgb = tuple((0.55 * base_rgb + 0.45)[:3])
        pre_color = (*pre_rgb, 0.6)
        post_color = (*base_rgb, 0.9)
        sns.violinplot(
            data=melt_df,
            x="group",
            y="value",
            hue="group",
            order=["Pre", "Post"],
            palette={"Pre": pre_color, "Post": post_color},
            cut=0,
            inner=None,
            dodge=False,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        med_pre = float(np.median(pre_vals)) if len(pre_vals) else float("nan")
        med_post = float(np.median(post_vals)) if len(post_vals) else float("nan")
        for xpos, med in zip([0, 1], [med_pre, med_post]):
            if not math.isnan(med):
                ax.hlines(med, xpos - 0.18, xpos + 0.18, colors="black", linewidth=2)

        for _, row in sub.iterrows():
            y_pre = float(row[initial_name])
            y_post = float(row[final_name])
            color = stance_color(row.get("persuader_supports_proposition", None))

            x0 = 0 + np.random.uniform(-jitter_scale, jitter_scale)
            x1 = 1 + np.random.uniform(-jitter_scale, jitter_scale)
            ax.plot([x0, x1], [y_pre, y_post], color=color, alpha=0.35, linewidth=1.0)
            ax.scatter([x0], [y_pre], color=color, alpha=0.85, s=14)
            ax.scatter([x1], [y_post], color=color, alpha=0.85, s=14)

        ax.set(
            xticks=[0, 1],
            xticklabels=["Pre", "Post"],
            ylim=(0, 1),
            xlabel=split_condition_label(str(cond)),
        )
        ax.minorticks_off()
        ax.text(
            0.5,
            1.02,
            f"Δ = {mean_delta:+.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ylabel = YLABEL_SUPPORT_P_REL if persuader_relative else YLABEL_SUPPORT
    axes[0].set_ylabel(ylabel)
    for ax in axes[1:]:
        ax.set_ylabel("")

    legend_elems = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="Pro"),
        Line2D([0], [0], color="#d62728", lw=2, label="Anti"),
    ]
    fig.legend(handles=legend_elems, fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    save_or_show(fig, fig_dir / "pre_post_violins_all.pdf", show)
