"""Fit and export a human trajectory cluster model artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from analysis.simulator_plot_style import PAPER_SQUARE_FIGURE_SIZE_INCHES
from simulation.human_likeness import (
    load_serial_trajectories,
    parse_min_date,
    select_human_reference,
)
from simulation.human_likeness_eval.trajectory_metrics import (
    _belief_bin_sort_key,
    _initial_belief_bin_from_value,
)

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_PATH = Path(
    "src/simulation/data/human_trajectory_cluster_model_k3_v1.json"
)
DEFAULT_SUMMARY_PATH = Path("analysis/data/human_clusters_k3_summary.csv")
DEFAULT_BY_INIT_BIN_PATH = Path("analysis/data/human_clusters_k3_by_init_bin.csv")
DEFAULT_BY_INIT_BIN_HEATMAP_PATH = Path(
    "analysis/data/human_clusters_k3_by_init_bin_heatmap.pdf"
)
DEFAULT_CLUSTER_SHAPES_PATH = Path("analysis/data/human_clusters_k3_shapes.pdf")
DEFAULT_PCA_SCATTER_PATH = Path("analysis/data/human_clusters_k3_pca_scatter.pdf")
DEFAULT_MIN_CLUSTER_COUNTS = (10, 20, 30)
DEFAULT_PCA_FIG_WIDTH = float(PAPER_SQUARE_FIGURE_SIZE_INCHES[0])
DEFAULT_PCA_FIG_HEIGHT = float(PAPER_SQUARE_FIGURE_SIZE_INCHES[1])


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fit KMeans clusters on human serial-question trajectories."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root results directory containing round JSONL files.",
    )
    parser.add_argument(
        "--min-date",
        "--min",
        type=str,
        default=None,
        help="Only load results at or after YYYY-MM-DD.",
    )
    parser.add_argument(
        "--human-source",
        choices=["llm-human-target", "human-human", "all-human-target"],
        default="llm-human-target",
        help="Human reference corpus selector.",
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        help="Include control-dialogue rounds. Default excludes them.",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Include audio rounds. Default uses text-only rounds.",
    )
    parser.add_argument(
        "--persuader-model",
        type=str,
        default=None,
        help="Optional exact filter on roles.llm_persuader.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=None,
        help="Optional exact turn-limit filter.",
    )
    parser.add_argument(
        "--participant-proposition",
        choices=["any", "true", "false"],
        default="any",
        help="Filter by participant_proposition flag.",
    )
    parser.add_argument(
        "--condition-substring",
        type=str,
        default=None,
        help=(
            "Optional substring matched against condition.to_dir() and "
            "human-readable condition text."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=["trajectory", "init_only", "full_bn_survey"],
        default="trajectory",
        help=(
            "Feature set used for clustering: full trajectory dynamics, initial "
            "belief only, or full BN-survey summaries."
        ),
    )
    parser.add_argument(
        "--require-bn-survey",
        action="store_true",
        help=(
            "Require enable_node_belief_survey=True at the condition level before "
            "feature extraction."
        ),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path for cluster model artifact.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output CSV path for cluster summary stats.",
    )
    parser.add_argument(
        "--output-by-init-bin-csv",
        type=Path,
        default=DEFAULT_BY_INIT_BIN_PATH,
        help="Output CSV path for cluster x initial-belief-bin counts/proportions.",
    )
    parser.add_argument(
        "--output-by-init-bin-heatmap",
        type=Path,
        default=DEFAULT_BY_INIT_BIN_HEATMAP_PATH,
        help="Output figure path for P(cluster | initial-belief-bin) heatmap.",
    )
    parser.add_argument(
        "--output-cluster-shapes",
        type=Path,
        default=DEFAULT_CLUSTER_SHAPES_PATH,
        help="Output figure path for cluster trajectory-shape summary.",
    )
    parser.add_argument(
        "--output-pca-scatter",
        type=Path,
        default=DEFAULT_PCA_SCATTER_PATH,
        help="Output figure path for 2D PCA scatter of cluster assignments.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of KMeans clusters.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=21,
        help="Number of normalized cumulative points used in feature vectors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="KMeans random seed.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=60,
        help="KMeans n_init setting.",
    )
    parser.add_argument(
        "--disable-normalization-if-fixed-length",
        action="store_true",
        help=(
            "Deprecated. Raw fixed-length features are now the default when all "
            "rounds have the same number of updates."
        ),
    )
    parser.add_argument(
        "--normalize-init-bin-distribution",
        action="store_true",
        help=(
            "Plot normalized P(cluster | init_bin) in the init-bin heatmap. "
            "Default plots raw counts."
        ),
    )
    parser.add_argument(
        "--normalize-trajectories",
        action="store_true",
        help=(
            "Use normalized interpolation for trajectory features/plots. "
            "Default uses raw turn-index trajectories with padding."
        ),
    )
    parser.add_argument(
        "--pca-fig-width",
        type=float,
        default=DEFAULT_PCA_FIG_WIDTH,
        help="Figure width in inches for PCA scatter output.",
    )
    parser.add_argument(
        "--pca-fig-height",
        type=float,
        default=DEFAULT_PCA_FIG_HEIGHT,
        help="Figure height in inches for PCA scatter output.",
    )
    parser.add_argument(
        "--pca-hide-title",
        action="store_true",
        help="Hide title text in PCA scatter output.",
    )
    parser.add_argument(
        "--min-cluster-count-targets",
        type=str,
        default="10,20,30",
        help=(
            "Comma-separated target minimum per-cluster counts used for "
            "sample-size adequacy estimates."
        ),
    )
    return parser.parse_args()


def _trajectory_curve(
    updates: np.ndarray,
    *,
    grid_points: int,
    feature_mode: str,
    fixed_turn_count: int | None = None,
) -> np.ndarray:
    """
    Build trajectory curve representation used in cluster features.

    Args:
        updates: Persuader-relative per-message updates.
        grid_points: Number of interpolation points for normalized mode.
        feature_mode: One of ``normalized`` or ``raw_padded``.
        fixed_turn_count: Required padded turn count for ``raw_padded`` mode.

    Returns:
        Cumulative trajectory values including x=0.
    """
    cumulative = np.concatenate(
        [
            np.asarray([0.0], dtype=float),
            np.cumsum(updates, dtype=float),
        ]
    )
    if feature_mode == "normalized":
        x_values = np.linspace(0.0, 1.0, cumulative.size, dtype=float)
        grid = np.linspace(0.0, 1.0, int(grid_points), dtype=float)
        return np.interp(grid, x_values, cumulative)
    if feature_mode == "raw_padded":
        if fixed_turn_count is None or fixed_turn_count < 1:
            raise ValueError("raw_padded mode requires fixed_turn_count >= 1.")
        if updates.size > fixed_turn_count:
            raise ValueError("raw_padded mode cannot truncate updates.")
        if updates.size == fixed_turn_count:
            return cumulative
        pad_len = int(fixed_turn_count - updates.size)
        if cumulative.size == 0:
            return np.zeros((fixed_turn_count + 1,), dtype=float)
        tail_value = float(cumulative[-1])
        padded_tail = np.full((pad_len,), tail_value, dtype=float)
        return np.concatenate([cumulative, padded_tail], axis=0)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def _cluster_label(cluster_id: int) -> str:
    """
    Return a neutral cluster label.

    Args:
        cluster_id: Sorted cluster id.

    Returns:
        Cluster label.
    """
    return f"cluster_{cluster_id}"


def _filter_rows_by_condition_substring(
    rows: list[Any], condition_substring: str | None
) -> list[Any]:
    """
    Filter rows by substring match on encoded and human-readable condition text.

    Args:
        rows: Candidate human rows.
        condition_substring: Optional case-insensitive substring selector.

    Returns:
        Filtered row list.
    """
    if condition_substring is None or not condition_substring.strip():
        return rows
    needle = condition_substring.strip().lower()
    selected: list[Any] = []
    for row in rows:
        haystack = f"{row.condition.to_dir()} {row.condition}".lower()
        if needle in haystack:
            selected.append(row)
    return selected


def _filter_rows_with_bn_survey(rows: list[Any]) -> list[Any]:
    """
    Keep only rows whose condition enabled node-belief survey collection.

    Args:
        rows: Candidate human rows.

    Returns:
        Rows where ``condition.enable_node_belief_survey`` is true.
    """
    selected: list[Any] = []
    for row in rows:
        if bool(row.condition.enable_node_belief_survey):
            selected.append(row)
    return selected


def _parse_count_targets(raw: str) -> tuple[int, ...]:
    """
    Parse per-cluster minimum-count targets from a comma-delimited string.

    Args:
        raw: Comma-separated positive integer list.

    Returns:
        Sorted unique positive integer targets.
    """
    pieces = [piece.strip() for piece in str(raw).split(",")]
    values: list[int] = []
    for piece in pieces:
        if not piece:
            continue
        parsed = int(piece)
        if parsed <= 0:
            raise ValueError("--min-cluster-count-targets must be positive integers.")
        values.append(int(parsed))
    if not values:
        return DEFAULT_MIN_CLUSTER_COUNTS
    return tuple(sorted(set(values)))


def _belief_node_sort_key(node_id: str) -> tuple[int, str]:
    """
    Sort belief node ids by numeric suffix when present.

    Args:
        node_id: Node identifier string.

    Returns:
        Numeric-aware sort key.
    """
    if node_id.startswith("Belief_"):
        suffix = node_id.replace("Belief_", "", 1)
        if suffix.isdigit():
            return int(suffix), node_id
    return 10**9, node_id


def _node_belief_vector(payload: Any) -> np.ndarray | None:
    """
    Convert a node-belief mapping payload to a sorted numeric vector.

    Args:
        payload: Mapping from node id to belief value.

    Returns:
        Sorted float vector or ``None`` when invalid.
    """
    if not isinstance(payload, dict) or not payload:
        return None
    entries: list[tuple[str, float]] = []
    for key_raw, value_raw in payload.items():
        if not isinstance(value_raw, (int, float)):
            return None
        entries.append((str(key_raw), float(value_raw)))
    entries.sort(key=lambda item: _belief_node_sort_key(item[0]))
    return np.asarray([value for _, value in entries], dtype=float)


def _initial_target_belief_feature(row: Any) -> np.ndarray | None:
    """
    Build initial-target-belief-only feature vector.

    Args:
        row: Human trajectory row.

    Returns:
        One-dimensional feature vector or ``None`` when unavailable.
    """
    initial = row.round_obj.target_initial_belief
    if not isinstance(initial, (int, float)):
        return None
    return np.asarray([float(initial)], dtype=float)


def _bn_survey_feature_vector(row: Any) -> np.ndarray | None:
    """
    Build fixed-width BN-survey summary feature vector.

    Args:
        row: Human trajectory row.

    Returns:
        Feature vector with target/node initial/final summary statistics, or
        ``None`` when required survey payloads are unavailable.
    """
    initial_target_raw = row.round_obj.target_initial_belief
    final_target_raw = row.round_obj.target_final_belief
    if not isinstance(initial_target_raw, (int, float)):
        return None
    if not isinstance(final_target_raw, (int, float)):
        return None

    initial_nodes = _node_belief_vector(row.round_obj.target_initial_node_beliefs)
    final_nodes = _node_belief_vector(row.round_obj.target_final_node_beliefs)
    if initial_nodes is None or final_nodes is None:
        return None
    if initial_nodes.size == 0 or final_nodes.size == 0:
        return None
    if initial_nodes.size != final_nodes.size:
        return None

    node_delta = final_nodes - initial_nodes
    initial_target = float(initial_target_raw)
    final_target = float(final_target_raw)
    target_delta = float(final_target - initial_target)
    return np.asarray(
        [
            initial_target,
            final_target,
            target_delta,
            float(initial_nodes.size),
            float(np.mean(initial_nodes)),
            float(np.std(initial_nodes)),
            float(np.min(initial_nodes)),
            float(np.max(initial_nodes)),
            float(np.mean(final_nodes)),
            float(np.std(final_nodes)),
            float(np.min(final_nodes)),
            float(np.max(final_nodes)),
            float(np.mean(node_delta)),
            float(np.std(node_delta)),
            float(np.min(node_delta)),
            float(np.max(node_delta)),
            float(np.mean(np.abs(node_delta))),
        ],
        dtype=float,
    )


def _feature_schema(feature_set: str, feature_mode: str) -> str:
    """
    Return a human-readable schema label for the selected feature set.

    Args:
        feature_set: Feature-set selector.
        feature_mode: Trajectory interpolation mode.

    Returns:
        Schema descriptor string.
    """
    if feature_set == "trajectory":
        if feature_mode == "normalized":
            return "normalized_cumulative[1:] + [turn_count]"
        if feature_mode == "raw_padded":
            return "raw_padded_cumulative[1:] + [turn_count]"
        return "raw_fixed_length_cumulative[1:] + [turn_count]"
    if feature_set == "init_only":
        return "target_initial_belief"
    if feature_set == "full_bn_survey":
        return (
            "target_init/final/delta + node_count + "
            "node_init_stats + node_final_stats + node_delta_stats"
        )
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _feature_vector_for_row(
    *,
    row: Any,
    feature_set: str,
    trajectory_curve: np.ndarray,
) -> np.ndarray | None:
    """
    Build clustering feature vector for one row under the selected feature set.

    Args:
        row: Human trajectory row.
        feature_set: Feature-set selector.
        trajectory_curve: Cumulative trajectory values for this row.

    Returns:
        Feature vector, or ``None`` when this row is ineligible.
    """
    updates = np.asarray(row.updates, dtype=float)
    if feature_set == "trajectory":
        return np.concatenate(
            [
                trajectory_curve[1:],
                np.asarray([float(updates.size)], dtype=float),
            ],
            axis=0,
        )
    if feature_set == "init_only":
        return _initial_target_belief_feature(row)
    if feature_set == "full_bn_survey":
        return _bn_survey_feature_vector(row)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _cluster_sample_size_targets(
    *,
    cluster_sizes: np.ndarray,
    total_rows: int,
    min_count_targets: tuple[int, ...],
) -> list[dict[str, int | float]]:
    """
    Estimate total N required to achieve target minimum cluster counts.

    Args:
        cluster_sizes: Count per cluster.
        total_rows: Current corpus size.
        min_count_targets: Desired minimum per-cluster counts.

    Returns:
        One row per target threshold with required-total and additional counts.
    """
    nonzero = cluster_sizes[cluster_sizes > 0]
    if nonzero.size == 0 or total_rows <= 0:
        return []
    min_cluster_share = float(np.min(nonzero) / float(total_rows))
    if min_cluster_share <= 0.0:
        return []
    rows: list[dict[str, int | float]] = []
    for target in min_count_targets:
        required_total = int(math.ceil(float(target) / min_cluster_share))
        rows.append(
            {
                "target_min_cluster_n": int(target),
                "required_total_n": int(required_total),
                "additional_n_needed": int(max(0, required_total - int(total_rows))),
                "smallest_cluster_share": float(min_cluster_share),
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """
    Write tabular rows to CSV.

    Args:
        path: Output file path.
        rows: Rows to serialize.
        columns: CSV column order.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_fraction(numerator: int, denominator: int) -> float:
    """
    Compute a finite ratio when denominator is positive.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.

    Returns:
        Ratio or NaN when denominator is zero.
    """
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _row_init_belief_bin(row: Any) -> str:
    """
    Resolve the initial-belief bin label for a human trajectory row.

    Args:
        row: Round trajectory row with a round object.

    Returns:
        Initial belief bin label.
    """
    initial_raw = row.round_obj.target_initial_belief
    if isinstance(initial_raw, (int, float)):
        return _initial_belief_bin_from_value(float(initial_raw))
    return "unknown"


def _cluster_init_bin_row(
    *,
    cluster_id: int,
    init_bin: str,
    count: int,
    totals: tuple[int, int, int],
) -> dict[str, object]:
    """
    Build one cluster x initial-bin output row.

    Args:
        cluster_id: Cluster id.
        init_bin: Initial-belief bin label.
        count: Cell count.
        totals: `(cluster_total, init_bin_total, corpus_total)`.

    Returns:
        Output row dictionary.
    """
    cluster_total, init_bin_total, corpus_total = totals
    return {
        "cluster_id": int(cluster_id),
        "cluster_name": _cluster_label(int(cluster_id)),
        "init_belief_bin": init_bin,
        "count": int(count),
        "cluster_total": int(cluster_total),
        "init_bin_total": int(init_bin_total),
        "corpus_total": int(corpus_total),
        "prop_within_cluster": _safe_fraction(int(count), int(cluster_total)),
        "prop_within_init_bin": _safe_fraction(int(count), int(init_bin_total)),
        "prop_within_corpus": _safe_fraction(int(count), int(corpus_total)),
    }


def _cluster_init_bin_rows(
    *,
    human_rows: list[Any],
    remapped_labels: np.ndarray,
    k: int,
) -> list[dict[str, object]]:
    """
    Build cluster x initial-belief-bin counts and normalized proportions.

    Args:
        human_rows: Human reference rows used for fitting.
        remapped_labels: Cluster assignment per row after end-delta ordering.
        k: Number of clusters.

    Returns:
        One row per cluster/bin cell including count and proportion columns.
    """
    counts: dict[tuple[int, str], int] = {}
    cluster_totals: dict[int, int] = {cluster_id: 0 for cluster_id in range(k)}
    init_bin_totals: dict[str, int] = {}

    for cluster_id_raw, row in zip(remapped_labels, human_rows):
        cluster_id = int(cluster_id_raw)
        init_bin = _row_init_belief_bin(row)
        counts[(cluster_id, init_bin)] = counts.get((cluster_id, init_bin), 0) + 1
        cluster_totals[cluster_id] = cluster_totals.get(cluster_id, 0) + 1
        init_bin_totals[init_bin] = init_bin_totals.get(init_bin, 0) + 1

    ordered_bins = sorted(init_bin_totals, key=_belief_bin_sort_key)
    total_rows = int(len(human_rows))
    rows_out: list[dict[str, object]] = []
    for cluster_id in range(k):
        cluster_total = int(cluster_totals.get(cluster_id, 0))
        for init_bin in ordered_bins:
            rows_out.append(
                _cluster_init_bin_row(
                    cluster_id=cluster_id,
                    init_bin=init_bin,
                    count=int(counts.get((cluster_id, init_bin), 0)),
                    totals=(
                        cluster_total,
                        int(init_bin_totals.get(init_bin, 0)),
                        total_rows,
                    ),
                )
            )
    return rows_out


def _cluster_init_bin_matrices(
    *,
    rows: list[dict[str, object]],
    k: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Build heatmap matrices from cluster-by-initial-bin rows.

    Args:
        rows: Rows from `_cluster_init_bin_rows`.
        k: Number of clusters.

    Returns:
        Tuple of ordered bins, value matrix, and count matrix.
    """
    init_bins = sorted(
        {str(row["init_belief_bin"]) for row in rows},
        key=_belief_bin_sort_key,
    )
    matrix = np.full((k, len(init_bins)), np.nan, dtype=float)
    count_matrix = np.zeros((k, len(init_bins)), dtype=int)
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        lookup[(int(row["cluster_id"]), str(row["init_belief_bin"]))] = row

    for cluster_id in range(k):
        for col_idx, init_bin in enumerate(init_bins):
            item = lookup.get((cluster_id, init_bin))
            if item is None:
                continue
            matrix[cluster_id, col_idx] = float(item["prop_within_init_bin"])
            count_matrix[cluster_id, col_idx] = int(item["count"])
    return init_bins, matrix, count_matrix


def _plot_cluster_by_init_bin_heatmap(
    *,
    path: Path,
    rows: list[dict[str, object]],
    k: int,
    feature_set: str,
    normalize: bool,
) -> None:
    """
    Plot heatmap of cluster x initial-belief-bin from human-only rows.

    Args:
        path: Output figure path.
        rows: Rows from `_cluster_init_bin_rows`.
        k: Number of clusters.
        feature_set: Feature-set used for clustering.
        normalize: Whether to plot normalized proportions instead of counts.

    Returns:
        None.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    init_bins, proportion_matrix, count_matrix = _cluster_init_bin_matrices(
        rows=rows, k=k
    )
    if normalize:
        matrix = proportion_matrix
        vmax = 1.0
        colorbar_label = "P(cluster | init_bin)"
    else:
        matrix = count_matrix.astype(float)
        vmax = float(np.max(matrix)) if np.max(matrix) > 0 else 1.0
        colorbar_label = "Count"

    fig, axis = plt.subplots(figsize=(max(6.8, 1.35 * len(init_bins)), 4.8))
    image = axis.imshow(
        matrix,
        vmin=0.0,
        vmax=vmax,
        cmap="Blues",
        aspect="auto",
        interpolation="nearest",
    )
    for cluster_id in range(k):
        for col_idx, _init_bin in enumerate(init_bins):
            value = matrix[cluster_id, col_idx]
            count = count_matrix[cluster_id, col_idx]
            if not np.isfinite(value) or count <= 0:
                continue
            text_color = "white" if value >= (0.55 * vmax) else "black"
            if normalize:
                label_text = f"{value:.2f}\n(n={count})"
            else:
                label_text = f"{int(count)}"
            axis.text(
                col_idx,
                cluster_id,
                label_text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    axis.set_xticks(np.arange(len(init_bins), dtype=float))
    axis.set_xticklabels(init_bins, rotation=25, ha="right")
    axis.set_yticks(np.arange(k, dtype=float))
    axis.set_yticklabels([f"C{cluster_id}" for cluster_id in range(k)])
    axis.set_xlabel("Initial belief bin")
    axis.set_ylabel("Cluster")
    fig.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label=colorbar_label,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cluster_shapes(
    *,
    path: Path,
    curve_matrix: np.ndarray,
    remapped_labels: np.ndarray,
    k: int,
    feature_mode: str,
    feature_set: str,
) -> None:
    """
    Plot mean and IQR cumulative trajectory shapes for each cluster.

    Args:
        path: Output figure path.
        curve_matrix: Cumulative trajectory curve matrix.
        remapped_labels: Cluster labels aligned with curve rows.
        k: Number of clusters.
        feature_mode: Feature mode (`normalized` or `raw_padded`).
        feature_set: Feature-set used for clustering.

    Returns:
        None.
    """
    if curve_matrix.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = int(curve_matrix.shape[1])
    if feature_mode == "normalized":
        x_values = np.linspace(0.0, 1.0, n_points, dtype=float)
        x_label = "Normalized conversation progress"
    else:
        x_values = np.arange(n_points, dtype=float)
        x_label = "Turn index (0 = initial)"

    fig, axis = plt.subplots(figsize=(8.2, 4.9))
    cmap = plt.get_cmap("tab10")
    for cluster_id in range(k):
        member_idx = np.where(remapped_labels == cluster_id)[0]
        if member_idx.size == 0:
            continue
        member_curves = curve_matrix[member_idx]
        mean_curve = np.mean(member_curves, axis=0)
        q25_curve = np.quantile(member_curves, 0.25, axis=0)
        q75_curve = np.quantile(member_curves, 0.75, axis=0)
        color = cmap(cluster_id % 10)
        axis.plot(
            x_values,
            mean_curve,
            color=color,
            linewidth=2.0,
            label=f"C{cluster_id} (n={int(member_idx.size)})",
        )
        axis.fill_between(
            x_values,
            q25_curve,
            q75_curve,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )

    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.25)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Cumulative belief change (persuader-relative)")
    axis.grid(alpha=0.20, linestyle=":")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _ellipse_geometry(
    points: np.ndarray,
    *,
    n_std: float = 2.0,
) -> tuple[float, float, float] | None:
    """
    Estimate 2D ellipse geometry from point covariance.

    Args:
        points: Two-dimensional point matrix with shape ``(n_points, 2)``.
        n_std: Number of standard deviations for ellipse radii.

    Returns:
        Tuple ``(width, height, angle_deg)`` or ``None`` when unavailable.
    """
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        return None
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    major_axis = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    safe_values = np.maximum(eigenvalues, 1e-12)
    width = float(2.0 * n_std * np.sqrt(safe_values[0]))
    height = float(2.0 * n_std * np.sqrt(safe_values[1]))
    return (width, height, major_axis)


def _plot_cluster_pca_scatter(
    *,
    path: Path,
    feature_matrix_scaled: np.ndarray,
    remapped_labels: np.ndarray,
    k: int,
    silhouette: float,
    figure_size: tuple[float, float],
    show_title: bool,
) -> None:
    """
    Plot a 2D PCA scatter view of assigned human trajectory clusters.

    Args:
        path: Output figure path.
        feature_matrix_scaled: Standardized feature matrix.
        remapped_labels: Final cluster labels aligned with feature rows.
        k: Number of clusters.
        silhouette: Silhouette score for the fitted labels.
        figure_size: Figure size in inches as ``(width, height)``.
        show_title: Whether to render a title.

    Returns:
        None.
    """
    if feature_matrix_scaled.size == 0 or remapped_labels.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    if feature_matrix_scaled.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=17)
        embedding = pca.fit_transform(feature_matrix_scaled)
        explained = pca.explained_variance_ratio_
    else:
        x_axis = feature_matrix_scaled[:, 0]
        embedding = np.column_stack([x_axis, np.zeros_like(x_axis)])
        explained = np.asarray([1.0, 0.0], dtype=float)

    fig, axis = plt.subplots(figsize=figure_size)
    cmap = plt.get_cmap("tab10")
    for cluster_id in range(k):
        member_idx = np.where(remapped_labels == cluster_id)[0]
        if member_idx.size == 0:
            continue
        cluster_points = embedding[member_idx]
        color = cmap(cluster_id % 10)
        axis.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=18.0,
            alpha=0.80,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=f"C{cluster_id} (n={int(member_idx.size)})",
        )
        geometry = _ellipse_geometry(cluster_points, n_std=2.0)
        if geometry is not None:
            width, height, angle = geometry
            center = np.mean(cluster_points, axis=0)
            axis.add_patch(
                Ellipse(
                    xy=(float(center[0]), float(center[1])),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.1,
                    alpha=0.13,
                )
            )

    axis.axhline(0.0, color="black", alpha=0.12, linewidth=0.8)
    axis.axvline(0.0, color="black", alpha=0.12, linewidth=0.8)
    axis.grid(alpha=0.18, linestyle=":")
    axis.set_xlabel(f"PC1 ({100.0 * float(explained[0]):.1f}% var)", fontsize=8)
    axis.set_ylabel(f"PC2 ({100.0 * float(explained[1]):.1f}% var)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.legend(frameon=False, loc="upper left", fontsize=6)
    if show_title:
        axis.set_title("Human trajectory clusters in 2D PCA space", fontsize=9)

    silhouette_text = "nan"
    if np.isfinite(silhouette):
        silhouette_text = f"{float(silhouette):.3f}"
    axis.text(
        0.99,
        0.01,
        f"N={int(remapped_labels.size)}  k={int(k)}  silhouette={silhouette_text}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.86,
            "edgecolor": "#cccccc",
        },
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    Fit KMeans on human trajectories and export model + summary.
    """
    args = parse_args()
    min_count_targets = _parse_count_targets(args.min_cluster_count_targets)
    min_date = parse_min_date(args.min_date)
    rows = load_serial_trajectories(args.results_dir, min_date=min_date)
    human_rows = select_human_reference(
        rows,
        human_source=args.human_source,
        include_control=bool(args.include_control),
        include_audio=bool(args.include_audio),
        persuader_model=args.persuader_model,
        turn_limit=args.turn_limit,
        participant_proposition=args.participant_proposition,
    )
    human_rows = _filter_rows_by_condition_substring(
        human_rows,
        args.condition_substring,
    )
    if args.require_bn_survey:
        human_rows = _filter_rows_with_bn_survey(human_rows)
    if not human_rows:
        raise ValueError("No human serial trajectories available for fitting.")

    update_lengths = {len(row.updates) for row in human_rows}
    if args.normalize_trajectories:
        curve_feature_mode = "normalized"
        fixed_turn_count = None
    else:
        curve_feature_mode = "raw_padded"
        fixed_turn_count = int(max(update_lengths))

    selected_rows: list[Any] = []
    curves: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    ends: list[float] = []
    first_deltas: list[float] = []
    delta_stds: list[float] = []
    turn_counts: list[float] = []
    dropped_for_feature_set = 0
    for row in human_rows:
        updates = np.asarray(row.updates, dtype=float)
        curve = _trajectory_curve(
            updates,
            grid_points=int(args.grid_points),
            feature_mode=curve_feature_mode,
            fixed_turn_count=fixed_turn_count,
        )
        feature_vector = _feature_vector_for_row(
            row=row,
            feature_set=args.feature_set,
            trajectory_curve=curve,
        )
        if feature_vector is None:
            dropped_for_feature_set += 1
            continue
        selected_rows.append(row)
        curves.append(curve)
        end_value = float(np.sum(updates))
        first_deltas.append(float(updates[0]))
        delta_stds.append(float(np.std(updates)))
        turn_counts.append(float(updates.size))
        feature_rows.append(feature_vector)
        ends.append(end_value)

    if not selected_rows:
        raise ValueError("No rows remained after applying feature eligibility filters.")
    if int(args.k) < 1:
        raise ValueError("--k must be at least 1.")
    if int(args.k) > len(selected_rows):
        raise ValueError("--k cannot exceed the number of eligible human rounds.")

    feature_matrix = np.vstack(feature_rows)
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    kmeans = KMeans(
        n_clusters=int(args.k),
        random_state=int(args.seed),
        n_init=int(args.n_init),
    )
    labels_raw = kmeans.fit_predict(feature_matrix_scaled)
    unique_labels = np.unique(labels_raw)
    if 1 < unique_labels.size < feature_matrix_scaled.shape[0]:
        silhouette = float(silhouette_score(feature_matrix_scaled, labels_raw))
    else:
        silhouette = float("nan")

    cluster_order = sorted(
        set(labels_raw),
        key=lambda cluster_id: float(
            np.mean(
                [
                    ends[idx]
                    for idx, label in enumerate(labels_raw)
                    if label == cluster_id
                ]
            )
        ),
    )
    centers_reordered = [
        kmeans.cluster_centers_[cluster_id].tolist() for cluster_id in cluster_order
    ]

    model_payload: dict[str, Any] = {
        "model_name": f"human_trajectory_clusters_k{int(args.k)}_v1",
        "grid_points": int(args.grid_points),
        "feature_mode": curve_feature_mode,
        "fixed_turn_count": fixed_turn_count,
        "feature_set": args.feature_set,
        "feature_schema": _feature_schema(args.feature_set, curve_feature_mode),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "centers_scaled": centers_reordered,
        "clusters": [
            {
                "id": int(cluster_idx),
                "name": _cluster_label(int(cluster_idx)),
            }
            for cluster_idx in range(int(args.k))
        ],
        "metadata": {
            "trained_on": {
                "human_source": args.human_source,
                "include_control": bool(args.include_control),
                "include_audio": bool(args.include_audio),
                "participant_proposition": args.participant_proposition,
                "turn_limit": args.turn_limit,
                "persuader_model": args.persuader_model,
                "min_date": str(min_date) if min_date is not None else None,
                "condition_substring": args.condition_substring,
                "require_bn_survey": bool(args.require_bn_survey),
            },
            "n_human_rounds_before_feature_filter": int(len(human_rows)),
            "n_human_rounds": int(len(selected_rows)),
            "dropped_rows_for_feature_set": int(dropped_for_feature_set),
            "kmeans": {
                "k": int(args.k),
                "n_init": int(args.n_init),
                "random_state": int(args.seed),
            },
            "feature_mode_requested_disable_normalization_if_fixed_length": bool(
                args.disable_normalization_if_fixed_length
            ),
            "normalize_trajectories": bool(args.normalize_trajectories),
            "silhouette": silhouette,
        },
    }

    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_path.write_text(
        json.dumps(model_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_rows: list[dict[str, object]] = []
    remap = {old_id: new_id for new_id, old_id in enumerate(cluster_order)}
    remapped_labels = np.asarray([remap[int(label)] for label in labels_raw], dtype=int)
    curve_matrix = np.vstack(curves)
    first_array = np.asarray(first_deltas, dtype=float)
    end_array = np.asarray(ends, dtype=float)
    std_array = np.asarray(delta_stds, dtype=float)
    turn_count_array = np.asarray(turn_counts, dtype=float)
    curve_points = int(curve_matrix.shape[1])
    idx_25 = int(round((curve_points - 1) * 0.25))
    idx_50 = int(round((curve_points - 1) * 0.50))
    idx_75 = int(round((curve_points - 1) * 0.75))
    idx_100 = int(round((curve_points - 1) * 1.00))
    for cluster_idx in range(int(args.k)):
        members = np.where(remapped_labels == cluster_idx)[0]
        if members.size == 0:
            continue
        member_curves = curve_matrix[members]
        summary_rows.append(
            {
                "cluster_id": int(cluster_idx),
                "cluster_name": _cluster_label(int(cluster_idx)),
                "n": int(members.size),
                "share": float(members.size / len(selected_rows)),
                "mean_first_delta": float(np.mean(first_array[members])),
                "mean_end_delta": float(np.mean(end_array[members])),
                "mean_delta_std": float(np.mean(std_array[members])),
                "mean_turn_count": float(np.mean(turn_count_array[members])),
                "center_cum_25pct": float(np.mean(member_curves[:, idx_25])),
                "center_cum_50pct": float(np.mean(member_curves[:, idx_50])),
                "center_cum_75pct": float(np.mean(member_curves[:, idx_75])),
                "center_cum_100pct": float(np.mean(member_curves[:, idx_100])),
                "feature_set": args.feature_set,
            }
        )

    output_summary_path = Path(args.output_summary_csv)
    _write_rows(
        output_summary_path,
        summary_rows,
        columns=list(summary_rows[0].keys()),
    )

    cluster_by_init_bin_rows = _cluster_init_bin_rows(
        human_rows=selected_rows,
        remapped_labels=remapped_labels,
        k=int(args.k),
    )
    output_by_init_bin_path = Path(args.output_by_init_bin_csv)
    _write_rows(
        output_by_init_bin_path,
        cluster_by_init_bin_rows,
        columns=[
            "cluster_id",
            "cluster_name",
            "init_belief_bin",
            "count",
            "cluster_total",
            "init_bin_total",
            "corpus_total",
            "prop_within_cluster",
            "prop_within_init_bin",
            "prop_within_corpus",
        ],
    )
    output_by_init_bin_heatmap_path = Path(args.output_by_init_bin_heatmap)
    _plot_cluster_by_init_bin_heatmap(
        path=output_by_init_bin_heatmap_path,
        rows=cluster_by_init_bin_rows,
        k=int(args.k),
        feature_set=args.feature_set,
        normalize=bool(args.normalize_init_bin_distribution),
    )
    output_cluster_shapes_path = Path(args.output_cluster_shapes)
    _plot_cluster_shapes(
        path=output_cluster_shapes_path,
        curve_matrix=curve_matrix,
        remapped_labels=remapped_labels,
        k=int(args.k),
        feature_mode=curve_feature_mode,
        feature_set=args.feature_set,
    )
    output_pca_scatter_path = Path(args.output_pca_scatter)
    _plot_cluster_pca_scatter(
        path=output_pca_scatter_path,
        feature_matrix_scaled=feature_matrix_scaled,
        remapped_labels=remapped_labels,
        k=int(args.k),
        silhouette=silhouette,
        figure_size=(float(args.pca_fig_width), float(args.pca_fig_height)),
        show_title=not bool(args.pca_hide_title),
    )

    cluster_sizes = np.asarray([int(row["n"]) for row in summary_rows], dtype=int)
    sample_size_targets = _cluster_sample_size_targets(
        cluster_sizes=cluster_sizes,
        total_rows=int(len(selected_rows)),
        min_count_targets=min_count_targets,
    )

    print(
        json.dumps(
            {
                "output_model": str(output_model_path),
                "output_summary_csv": str(output_summary_path),
                "output_by_init_bin_csv": str(output_by_init_bin_path),
                "output_by_init_bin_heatmap": str(output_by_init_bin_heatmap_path),
                "output_cluster_shapes": str(output_cluster_shapes_path),
                "output_pca_scatter": str(output_pca_scatter_path),
                "n_human_rounds_before_feature_filter": int(len(human_rows)),
                "n_human_rounds": int(len(selected_rows)),
                "dropped_rows_for_feature_set": int(dropped_for_feature_set),
                "results_dir": str(Path(args.results_dir)),
                "min_date": str(min_date) if min_date is not None else None,
                "human_source": args.human_source,
                "include_control": bool(args.include_control),
                "include_audio": bool(args.include_audio),
                "persuader_model": args.persuader_model,
                "turn_limit": args.turn_limit,
                "participant_proposition": args.participant_proposition,
                "condition_substring": args.condition_substring,
                "require_bn_survey": bool(args.require_bn_survey),
                "k": int(args.k),
                "feature_set": args.feature_set,
                "feature_mode": curve_feature_mode,
                "fixed_turn_count": fixed_turn_count,
                "normalize_trajectories": bool(args.normalize_trajectories),
                "normalize_init_bin_distribution": bool(
                    args.normalize_init_bin_distribution
                ),
                "silhouette": silhouette,
                "min_cluster_count_targets": list(min_count_targets),
                "sample_size_targets": sample_size_targets,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
