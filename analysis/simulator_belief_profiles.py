"""Compare simulator and human node-level belief profiles."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from simulation.belief_profiles import (
    BELIEF_PROFILE_CORPUS_ORDER,
    BeliefProfileRow,
    assign_bin_label,
    belief_sort_key,
    build_belief_profile_corpora,
    profile_row_from_trajectory,
)
from simulation.human_likeness import (
    RoundTrajectory,
    load_serial_trajectories,
    parse_min_date,
)
from simulation.target_bins import TARGET_BELIEF_BIN_RANGES

from .simulator_common import (
    add_common_human_simulator_filter_args,
    add_include_vanilla_llm_target_arg,
    selector_kwargs_from_args,
)
from .simulator_plot_style import (
    COMPARISON_CORPUS_COLOR_MAP,
    COMPARISON_CORPUS_LABEL_MAP,
)
from .tables import print_table

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_PREFIX = Path("analysis/data/simulator_belief_profiles")
DEFAULT_MIN_BIN_N = 3
DEFAULT_HUMAN_CEILING_INIT_QUANTILES = 5
DEFAULT_HUMAN_CEILING_BIN_SCHEME = "quantile"
DEFAULT_HUMAN_CEILING_MIN_TRAIN_PER_BIN = 5
DEFAULT_HUMAN_CEILING_MIN_TEST_PER_BIN = 1


@dataclass(frozen=True)
class _ProfileRecord:
    """Store one profile row plus metadata used for fold assignments."""

    profile: BeliefProfileRow
    row_key: str
    human_unit_id: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare simulator corpora to a human reference using node-level "
            "pre/post belief profiles."
        )
    )
    add_common_human_simulator_filter_args(
        parser,
        include_results_dir=True,
        default_results_dir=DEFAULT_RESULTS_DIR,
        include_proposition_match=True,
    )
    add_include_vanilla_llm_target_arg(
        parser,
        default=True,
        help_text="Include vanilla llm_target rows in comparator outputs.",
    )
    parser.add_argument(
        "--min-bin-n",
        type=int,
        default=DEFAULT_MIN_BIN_N,
        help="Minimum count per corpus/bin for within-bin gap metrics.",
    )
    parser.add_argument(
        "--human-ceiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run leave-one-unit-out human ceiling analysis.",
    )
    parser.add_argument(
        "--human-ceiling-unit",
        choices=["target-participant", "round"],
        default="target-participant",
        help="Fold unit for leave-one-out human ceiling analysis.",
    )
    parser.add_argument(
        "--human-ceiling-bin-scheme",
        choices=["target_bins", "quantile"],
        default=DEFAULT_HUMAN_CEILING_BIN_SCHEME,
        help=(
            "Initial-state conditional binning scheme for human-ceiling analysis: "
            "'target_bins' uses fixed simulator-style belief ranges, "
            "'quantile' uses quantile(target_initial_belief)."
        ),
    )
    parser.add_argument(
        "--human-ceiling-init-quantiles",
        type=int,
        default=DEFAULT_HUMAN_CEILING_INIT_QUANTILES,
        help=(
            "Number of quantile bins for initial-belief conditional analysis "
            "(used only when --human-ceiling-bin-scheme=quantile)."
        ),
    )
    parser.add_argument(
        "--human-ceiling-min-train-per-bin",
        type=int,
        default=DEFAULT_HUMAN_CEILING_MIN_TRAIN_PER_BIN,
        help="Minimum training rounds required for a bin to be scored.",
    )
    parser.add_argument(
        "--human-ceiling-min-test-per-bin",
        type=int,
        default=DEFAULT_HUMAN_CEILING_MIN_TEST_PER_BIN,
        help="Minimum held-out rounds required for a bin to be scored.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix for CSV and plot artifacts.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf"],
        default="png",
        help="Output format for the human-likeness bar plot.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV using explicit field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV: {path.resolve()}")


def _safe_corpus_sort_key(corpus_name: str) -> tuple[int, str]:
    """Sort corpora by canonical order, then lexicographically for extras."""
    if corpus_name in BELIEF_PROFILE_CORPUS_ORDER:
        return (BELIEF_PROFILE_CORPUS_ORDER.index(corpus_name), corpus_name)
    return (len(BELIEF_PROFILE_CORPUS_ORDER), corpus_name)


def _safe_nanmean(values: list[float]) -> float:
    """Compute a NaN-safe mean from scalar values."""
    clean = np.asarray(
        [value for value in values if not math.isnan(value)], dtype=float
    )
    if clean.size == 0:
        return math.nan
    return float(np.mean(clean))


def _quantile_or_nan(values: np.ndarray, q: float) -> float:
    """Compute quantile for finite values and return NaN when empty."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return math.nan
    return float(np.quantile(clean, q))


def _row_key_from_trajectory(row: RoundTrajectory) -> str:
    """Build a stable row key from source file and line metadata."""
    round_index = (
        "none" if row.source_round_index is None else str(row.source_round_index)
    )
    return f"{row.source_path}:{row.source_line_index}:{round_index}"


def _human_unit_id_from_trajectory(row: RoundTrajectory) -> str | None:
    """Extract a stable human-target id for leave-one-out fold assignment."""
    round_obj = row.round_obj
    if round_obj.human_target_id is not None:
        return f"human_target_id:{round_obj.human_target_id}"
    if row.condition.roles.human_target and round_obj.target_id is not None:
        return f"target_id:{round_obj.target_id}"
    return None


def _human_unit_map(
    human_records: list[_ProfileRecord],
    *,
    unit_mode: str,
) -> tuple[dict[str, list[BeliefProfileRow]], int]:
    """Group human rows by fold unit and count fallback row-based assignments."""
    grouped: dict[str, list[BeliefProfileRow]] = defaultdict(list)
    fallback_round_count = 0
    for record in human_records:
        if unit_mode == "target-participant":
            unit_id = record.human_unit_id
        elif unit_mode == "round":
            unit_id = record.row_key
        else:
            raise ValueError(f"Unknown unit mode: {unit_mode}")

        if unit_id is None:
            unit_id = record.row_key
            fallback_round_count += 1
        grouped[unit_id].append(record.profile)
    return dict(grouped), fallback_round_count


def _initial_bin_edges(
    human_rows: list[BeliefProfileRow],
    *,
    n_quantiles: int,
) -> np.ndarray:
    """Build quantile bin edges for target initial beliefs."""
    if n_quantiles < 2:
        raise ValueError("--human-ceiling-init-quantiles must be at least 2.")
    values = np.asarray([row.pre_target for row in human_rows], dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_quantiles + 1)
    return np.asarray(np.quantile(values, quantiles), dtype=float)


def _initial_state_bin_label(
    row: BeliefProfileRow,
    *,
    bin_scheme: str,
    bin_edges: np.ndarray | None = None,
) -> str:
    """Assign a round to an initial-state bin crossed with persuader stance."""
    init_bin: str
    if bin_scheme == "quantile":
        if bin_edges is None:
            raise ValueError("bin_edges is required for quantile binning.")
        interior = bin_edges[1:-1]
        quantile_idx = int(np.searchsorted(interior, row.pre_target, side="right"))
        init_bin = f"q{quantile_idx + 1}"
    elif bin_scheme == "target_bins":
        init_bin = _target_bin_label(float(row.pre_target))
    else:
        raise ValueError(f"Unknown bin_scheme: {bin_scheme}")

    stance = "support" if row.supports_proposition else "oppose"
    return f"{init_bin}|{stance}"


def _target_bin_label(value: float) -> str:
    """Map initial target belief values to fixed simulator-style bins."""
    for idx, (label, (low, high)) in enumerate(TARGET_BELIEF_BIN_RANGES.items()):
        is_last = idx == len(TARGET_BELIEF_BIN_RANGES) - 1
        if (low <= value < high) or (is_last and low <= value <= high):
            return label
    return next(reversed(TARGET_BELIEF_BIN_RANGES))


def _rows_by_initial_bin(
    rows: list[BeliefProfileRow],
    *,
    bin_scheme: str,
    bin_edges: np.ndarray | None = None,
) -> dict[str, list[BeliefProfileRow]]:
    """Group profile rows by initial-state bin crossed with stance."""
    grouped: dict[str, list[BeliefProfileRow]] = defaultdict(list)
    for row in rows:
        grouped[
            _initial_state_bin_label(
                row,
                bin_scheme=bin_scheme,
                bin_edges=bin_edges,
            )
        ].append(row)
    return grouped


def _delta_matrices(
    rows: list[BeliefProfileRow],
    *,
    node_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract target and node delta matrices for a row collection."""
    target_delta = np.asarray([row.target_delta for row in rows], dtype=float)
    node_delta = np.asarray(
        [
            [row.post_nodes[node_id] - row.pre_nodes[node_id] for node_id in node_ids]
            for row in rows
        ],
        dtype=float,
    )
    return target_delta, node_delta


def _unconditional_distribution_gaps(
    reference_rows: list[BeliefProfileRow],
    held_out_rows: list[BeliefProfileRow],
    *,
    node_ids: list[str],
) -> tuple[float, float]:
    """Compute unconditional target and node delta distribution gaps."""
    ref_target_delta, ref_node_delta = _delta_matrices(
        reference_rows, node_ids=node_ids
    )
    test_target_delta, test_node_delta = _delta_matrices(
        held_out_rows, node_ids=node_ids
    )

    target_w1 = float(wasserstein_distance(ref_target_delta, test_target_delta))
    node_w1 = float(
        np.mean(
            [
                wasserstein_distance(ref_node_delta[:, idx], test_node_delta[:, idx])
                for idx in range(len(node_ids))
            ]
        )
    )
    return target_w1, node_w1


def _bin_occupancy_js(
    reference_rows: list[BeliefProfileRow],
    held_out_rows: list[BeliefProfileRow],
    *,
    bin_scheme: str,
    bin_edges: np.ndarray | None = None,
) -> float:
    """Compute bin-occupancy JS distance for initial-state composition shift."""
    ref_by_bin = _rows_by_initial_bin(
        reference_rows,
        bin_scheme=bin_scheme,
        bin_edges=bin_edges,
    )
    test_by_bin = _rows_by_initial_bin(
        held_out_rows,
        bin_scheme=bin_scheme,
        bin_edges=bin_edges,
    )
    ref_counts = {label: len(rows) for label, rows in ref_by_bin.items()}
    test_counts = {label: len(rows) for label, rows in test_by_bin.items()}

    labels = sorted(set(ref_counts) | set(test_counts))
    ref_total = max(1, sum(ref_counts.values()))
    test_total = max(1, sum(test_counts.values()))

    ref_dist = np.asarray(
        [ref_counts.get(label, 0) / ref_total for label in labels], dtype=float
    )
    test_dist = np.asarray(
        [test_counts.get(label, 0) / test_total for label in labels],
        dtype=float,
    )
    return float(jensenshannon(ref_dist, test_dist, base=2))


def _conditional_distribution_gaps(
    reference_rows: list[BeliefProfileRow],
    held_out_rows: list[BeliefProfileRow],
    *,
    node_ids: list[str],
    bin_scheme: str,
    bin_edges: np.ndarray | None = None,
    min_train_per_bin: int,
    min_test_per_bin: int,
) -> dict[str, float | int]:
    """Compute within-initial-bin distribution gaps and coverage metrics."""
    ref_by_bin = _rows_by_initial_bin(
        reference_rows,
        bin_scheme=bin_scheme,
        bin_edges=bin_edges,
    )
    test_by_bin = _rows_by_initial_bin(
        held_out_rows,
        bin_scheme=bin_scheme,
        bin_edges=bin_edges,
    )

    evaluable_weights: list[float] = []
    target_scores: list[float] = []
    node_scores: list[float] = []
    evaluable_test_count = 0

    for bin_label, test_rows in test_by_bin.items():
        train_rows = ref_by_bin.get(bin_label, [])
        if len(train_rows) < min_train_per_bin:
            continue
        if len(test_rows) < min_test_per_bin:
            continue

        target_w1, node_w1 = _unconditional_distribution_gaps(
            train_rows,
            test_rows,
            node_ids=node_ids,
        )
        weight = float(len(test_rows))
        evaluable_weights.append(weight)
        target_scores.append(target_w1)
        node_scores.append(node_w1)
        evaluable_test_count += len(test_rows)

    total_test = len(held_out_rows)
    if not evaluable_weights:
        return {
            "conditional_target_delta_wasserstein": math.nan,
            "conditional_mean_node_delta_wasserstein": math.nan,
            "conditional_human_like_score": math.nan,
            "conditional_coverage": 0.0,
            "conditional_evaluable_bins": 0,
            "conditional_test_bins": int(len(test_by_bin)),
        }

    weights = np.asarray(evaluable_weights, dtype=float)
    weight_sum = float(np.sum(weights))
    target_arr = np.asarray(target_scores, dtype=float)
    node_arr = np.asarray(node_scores, dtype=float)

    conditional_target = float(np.sum(target_arr * weights) / weight_sum)
    conditional_node = float(np.sum(node_arr * weights) / weight_sum)
    conditional_score = float(math.exp(-(conditional_target + conditional_node)))

    return {
        "conditional_target_delta_wasserstein": conditional_target,
        "conditional_mean_node_delta_wasserstein": conditional_node,
        "conditional_human_like_score": conditional_score,
        "conditional_coverage": (
            float(evaluable_test_count) / float(total_test) if total_test > 0 else 0.0
        ),
        "conditional_evaluable_bins": int(len(evaluable_weights)),
        "conditional_test_bins": int(len(test_by_bin)),
    }


def _leave_one_unit_out_rows(
    *,
    unit_to_rows: dict[str, list[BeliefProfileRow]],
    node_ids: list[str],
    bin_scheme: str,
    bin_edges: np.ndarray | None = None,
    min_train_per_bin: int,
    min_test_per_bin: int,
) -> list[dict[str, Any]]:
    """Run leave-one-unit-out evaluation for human ceiling metrics."""
    unit_ids = sorted(unit_to_rows)
    if len(unit_ids) < 2:
        return []

    rows: list[dict[str, Any]] = []
    for held_out_unit in unit_ids:
        held_out_rows = unit_to_rows[held_out_unit]
        reference_rows = [
            row
            for unit_id in unit_ids
            if unit_id != held_out_unit
            for row in unit_to_rows[unit_id]
        ]
        if not held_out_rows or not reference_rows:
            continue

        uncond_target_w1, uncond_node_w1 = _unconditional_distribution_gaps(
            reference_rows,
            held_out_rows,
            node_ids=node_ids,
        )
        occupancy_js = _bin_occupancy_js(
            reference_rows,
            held_out_rows,
            bin_scheme=bin_scheme,
            bin_edges=bin_edges,
        )
        conditional = _conditional_distribution_gaps(
            reference_rows,
            held_out_rows,
            node_ids=node_ids,
            bin_scheme=bin_scheme,
            bin_edges=bin_edges,
            min_train_per_bin=min_train_per_bin,
            min_test_per_bin=min_test_per_bin,
        )

        out_row: dict[str, Any] = {
            "held_out_unit": held_out_unit,
            "n_units_reference": int(len(unit_ids) - 1),
            "n_rounds_reference": int(len(reference_rows)),
            "n_rounds_held_out": int(len(held_out_rows)),
            "bin_occupancy_js_distance": occupancy_js,
            "unconditional_target_delta_wasserstein": uncond_target_w1,
            "unconditional_mean_node_delta_wasserstein": uncond_node_w1,
            "unconditional_human_like_score": float(
                math.exp(-(uncond_target_w1 + uncond_node_w1))
            ),
        }
        out_row.update(conditional)
        rows.append(out_row)

    return rows


def _summarize_metric_rows(
    *,
    rows: list[dict[str, Any]],
    method: str,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    """Summarize metric distributions with mean and quantile intervals."""
    if not rows:
        return []

    summary_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        finite_values = values[np.isfinite(values)]
        summary_rows.append(
            {
                "method": method,
                "metric": metric,
                "n_evals": int(finite_values.size),
                "mean": (
                    float(np.mean(finite_values)) if finite_values.size else math.nan
                ),
                "median": (
                    float(np.median(finite_values)) if finite_values.size else math.nan
                ),
                "std": float(np.std(finite_values)) if finite_values.size else math.nan,
                "q025": _quantile_or_nan(finite_values, 0.025),
                "q05": _quantile_or_nan(finite_values, 0.05),
                "q95": _quantile_or_nan(finite_values, 0.95),
                "q975": _quantile_or_nan(finite_values, 0.975),
            }
        )
    return summary_rows


def main() -> None:
    """Run simulator-vs-human node-belief profile analysis."""
    args = parse_args()
    min_date = parse_min_date(args.min_date)
    trajectories = load_serial_trajectories(args.results_dir, min_date=min_date)
    selector_kwargs = selector_kwargs_from_args(args)
    corpora = build_belief_profile_corpora(
        trajectories,
        human_source=args.human_source,
        proposition_match=args.proposition_match,
        include_vanilla_llm_target=args.include_vanilla_llm_target,
        selector_kwargs=selector_kwargs,
    )

    profile_records: list[_ProfileRecord] = []
    for corpus_name, corpus_rows in corpora:
        for row in corpus_rows:
            profile = profile_row_from_trajectory(corpus_name, row)
            if profile is None:
                continue
            profile_records.append(
                _ProfileRecord(
                    profile=profile,
                    row_key=_row_key_from_trajectory(row),
                    human_unit_id=_human_unit_id_from_trajectory(row),
                )
            )

    profile_rows = [record.profile for record in profile_records]
    human_rows = [row for row in profile_rows if row.corpus == "human_reference"]
    if not human_rows:
        raise ValueError(
            "No human node-belief profiles found. Ensure pre/post node beliefs are "
            "collected and exported in round JSONL files."
        )

    human_node_id_sets = [set(row.pre_nodes.keys()) for row in human_rows]
    common_node_ids = sorted(set.intersection(*human_node_id_sets), key=belief_sort_key)
    if not common_node_ids:
        raise ValueError("No common node ids found in human node-belief profiles.")

    filtered_records: list[_ProfileRecord] = []
    for record in profile_records:
        row = record.profile
        if not all(node_id in row.pre_nodes for node_id in common_node_ids):
            continue
        if not all(node_id in row.post_nodes for node_id in common_node_ids):
            continue
        filtered_records.append(record)

    filtered_profiles = [record.profile for record in filtered_records]
    human_filtered_records = [
        record
        for record in filtered_records
        if record.profile.corpus == "human_reference"
    ]
    human_filtered = [record.profile for record in human_filtered_records]

    human_pre_matrix = np.asarray(
        [
            [row.pre_nodes[node_id] for node_id in common_node_ids]
            for row in human_filtered
        ],
        dtype=float,
    )
    thresholds = np.asarray(
        [
            [
                float(np.quantile(human_pre_matrix[:, idx], 0.33)),
                float(np.quantile(human_pre_matrix[:, idx], 0.67)),
            ]
            for idx in range(human_pre_matrix.shape[1])
        ],
        dtype=float,
    )

    round_csv_rows: list[dict[str, Any]] = []
    grouped_by_corpus_and_bin: dict[tuple[str, str], list[BeliefProfileRow]] = {}
    for row in filtered_profiles:
        pre_vec = np.asarray(
            [row.pre_nodes[node_id] for node_id in common_node_ids],
            dtype=float,
        )
        post_vec = np.asarray(
            [row.post_nodes[node_id] for node_id in common_node_ids],
            dtype=float,
        )
        node_delta = post_vec - pre_vec
        bin_label = assign_bin_label(pre_vec, thresholds)
        grouped_by_corpus_and_bin.setdefault((row.corpus, bin_label), []).append(row)

        payload: dict[str, Any] = {
            "corpus": row.corpus,
            "bin": bin_label,
            "proposition": row.proposition,
            "supports_proposition": row.supports_proposition,
            "pre_target": row.pre_target,
            "post_target": row.post_target,
            "target_delta": row.target_delta,
        }
        for idx, node_id in enumerate(common_node_ids, start=1):
            payload[f"node_{idx}_id"] = node_id
            payload[f"node_{idx}_pre"] = float(pre_vec[idx - 1])
            payload[f"node_{idx}_post"] = float(post_vec[idx - 1])
            payload[f"node_{idx}_delta"] = float(node_delta[idx - 1])
        round_csv_rows.append(payload)

    round_csv_path = Path(f"{args.output_prefix}_round_profiles.csv")
    if round_csv_rows:
        round_fieldnames = list(round_csv_rows[0].keys())
        _write_csv(round_csv_path, round_csv_rows, round_fieldnames)

    human_bin_counts: dict[str, int] = {}
    for row in round_csv_rows:
        if row["corpus"] == "human_reference":
            label = str(row["bin"])
            human_bin_counts[label] = human_bin_counts.get(label, 0) + 1
    total_human = int(sum(human_bin_counts.values()))
    human_bin_weights = {
        key: (count / total_human if total_human > 0 else 0.0)
        for key, count in human_bin_counts.items()
    }

    bin_summary_rows: list[dict[str, Any]] = []
    for corpus_name in BELIEF_PROFILE_CORPUS_ORDER:
        bins_for_corpus = sorted(
            label
            for (corpus, label) in grouped_by_corpus_and_bin
            if corpus == corpus_name
        )
        for bin_label in bins_for_corpus:
            rows_for_bin = grouped_by_corpus_and_bin[(corpus_name, bin_label)]
            target_deltas = np.asarray(
                [row.target_delta for row in rows_for_bin], dtype=float
            )
            node_delta_matrix = np.asarray(
                [
                    [
                        row.post_nodes[node_id] - row.pre_nodes[node_id]
                        for node_id in common_node_ids
                    ]
                    for row in rows_for_bin
                ],
                dtype=float,
            )
            summary_row: dict[str, Any] = {
                "corpus": corpus_name,
                "bin": bin_label,
                "n_rounds": int(len(rows_for_bin)),
                "mean_target_delta": float(np.mean(target_deltas)),
                "mean_abs_target_delta": float(np.mean(np.abs(target_deltas))),
            }
            for idx, node_id in enumerate(common_node_ids, start=1):
                summary_row[f"node_{idx}_id"] = node_id
                summary_row[f"node_{idx}_mean_delta"] = float(
                    np.mean(node_delta_matrix[:, idx - 1])
                )
            bin_summary_rows.append(summary_row)

    bin_summary_path = Path(f"{args.output_prefix}_bin_summary.csv")
    if bin_summary_rows:
        bin_summary_fieldnames = list(bin_summary_rows[0].keys())
        _write_csv(bin_summary_path, bin_summary_rows, bin_summary_fieldnames)

    human_by_bin = {
        row["bin"]: row
        for row in bin_summary_rows
        if row["corpus"] == "human_reference" and int(row["n_rounds"]) >= args.min_bin_n
    }

    corpus_names = sorted(
        {row.corpus for row in filtered_profiles}, key=_safe_corpus_sort_key
    )
    likeness_rows: list[dict[str, Any]] = []
    for corpus_name in corpus_names:
        corpus_rows = [row for row in round_csv_rows if row["corpus"] == corpus_name]
        if not corpus_rows:
            continue
        corpus_bin_counts: dict[str, int] = {}
        for row in corpus_rows:
            label = str(row["bin"])
            corpus_bin_counts[label] = corpus_bin_counts.get(label, 0) + 1

        all_bins = sorted(set(human_bin_counts) | set(corpus_bin_counts))
        human_dist = np.asarray(
            [
                human_bin_counts.get(label, 0) / max(total_human, 1)
                for label in all_bins
            ],
            dtype=float,
        )
        corpus_total = int(sum(corpus_bin_counts.values()))
        corpus_dist = np.asarray(
            [
                corpus_bin_counts.get(label, 0) / max(corpus_total, 1)
                for label in all_bins
            ],
            dtype=float,
        )
        occupancy_js = (
            float(jensenshannon(human_dist, corpus_dist, base=2))
            if total_human > 0 and corpus_total > 0
            else math.nan
        )

        corpus_by_bin = {
            row["bin"]: row
            for row in bin_summary_rows
            if row["corpus"] == corpus_name and int(row["n_rounds"]) >= args.min_bin_n
        }
        weighted_target_gap = 0.0
        weighted_node_gap = 0.0
        weight_sum = 0.0
        for bin_label, human_row in human_by_bin.items():
            corpus_row = corpus_by_bin.get(bin_label)
            if corpus_row is None:
                continue
            weight = float(human_bin_weights.get(bin_label, 0.0))
            if weight <= 0:
                continue
            target_gap = abs(
                float(corpus_row["mean_target_delta"])
                - float(human_row["mean_target_delta"])
            )
            node_gaps: list[float] = []
            for idx in range(1, len(common_node_ids) + 1):
                node_gaps.append(
                    abs(
                        float(corpus_row[f"node_{idx}_mean_delta"])
                        - float(human_row[f"node_{idx}_mean_delta"])
                    )
                )
            weighted_target_gap += weight * target_gap
            weighted_node_gap += weight * float(np.mean(node_gaps))
            weight_sum += weight
        if weight_sum > 0:
            weighted_target_gap /= weight_sum
            weighted_node_gap /= weight_sum
        else:
            weighted_target_gap = math.nan
            weighted_node_gap = math.nan

        corpus_pre = np.asarray(
            [
                [
                    float(row[f"node_{idx}_pre"])
                    for idx in range(1, len(common_node_ids) + 1)
                ]
                for row in corpus_rows
            ],
            dtype=float,
        )
        if corpus_name == "human_reference":
            mean_pre_wasserstein = 0.0
        else:
            mean_pre_wasserstein = float(
                np.mean(
                    [
                        wasserstein_distance(
                            human_pre_matrix[:, idx],
                            corpus_pre[:, idx],
                        )
                        for idx in range(len(common_node_ids))
                    ]
                )
            )

        if corpus_name == "human_reference":
            human_like_score = 1.0
        else:
            occ = 0.0 if math.isnan(occupancy_js) else occupancy_js
            tg = 0.0 if math.isnan(weighted_target_gap) else weighted_target_gap
            ng = 0.0 if math.isnan(weighted_node_gap) else weighted_node_gap
            ws = 0.0 if math.isnan(mean_pre_wasserstein) else mean_pre_wasserstein
            human_like_score = float(math.exp(-(2.0 * occ + tg + ng + ws)))

        likeness_rows.append(
            {
                "corpus": corpus_name,
                "n_rounds": int(corpus_total),
                "bin_occupancy_js_distance": occupancy_js,
                "weighted_target_delta_gap": weighted_target_gap,
                "weighted_node_delta_gap": weighted_node_gap,
                "mean_pre_node_wasserstein": mean_pre_wasserstein,
                "human_like_score": human_like_score,
            }
        )

    likeness_rows.sort(key=lambda row: _safe_corpus_sort_key(str(row["corpus"])))
    likeness_csv_path = Path(f"{args.output_prefix}_human_likeness.csv")
    if likeness_rows:
        likeness_fieldnames = list(likeness_rows[0].keys())
        _write_csv(likeness_csv_path, likeness_rows, likeness_fieldnames)

        table_rows = [
            {
                "corpus": row["corpus"],
                "n": row["n_rounds"],
                "score": round(float(row["human_like_score"]), 4),
                "bin_js": round(float(row["bin_occupancy_js_distance"]), 4),
                "target_gap": round(float(row["weighted_target_delta_gap"]), 4),
                "node_gap": round(float(row["weighted_node_delta_gap"]), 4),
            }
            for row in likeness_rows
        ]
        print_table(
            table_rows,
            columns=["corpus", "n", "score", "bin_js", "target_gap", "node_gap"],
            title="Node-Profile Human-Likeness",
            aligns={"n": "right", "score": "right", "bin_js": "right"},
        )

        plot_rows = [
            row for row in likeness_rows if str(row["corpus"]) != "human_reference"
        ]
        if not plot_rows:
            # Keep a temporary visual artifact when only human-reference
            # rows are available under narrow filters.
            plot_rows = list(likeness_rows)
        if plot_rows:
            labels = [
                COMPARISON_CORPUS_LABEL_MAP.get(str(row["corpus"]), str(row["corpus"]))
                for row in plot_rows
            ]
            values = [float(row["human_like_score"]) for row in plot_rows]
            colors = [
                COMPARISON_CORPUS_COLOR_MAP.get(str(row["corpus"]), "#888888")
                for row in plot_rows
            ]
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            ax.bar(labels, values, color=colors)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Human-Like Score")
            ax.tick_params(axis="x", rotation=15, labelsize=9)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            fig.tight_layout(pad=0.7)
            plot_path = Path(f"{args.output_prefix}_human_likeness.{args.plot_format}")
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=220)
            plt.close(fig)
            print(f"Saved figure: {plot_path.resolve()}")

    leave_one_out_path = Path(f"{args.output_prefix}_human_ceiling_leave_one_out.csv")
    ceiling_summary_path = Path(f"{args.output_prefix}_human_ceiling_summary.csv")
    leave_one_out_rows: list[dict[str, Any]] = []
    ceiling_summary_rows: list[dict[str, Any]] = []

    if args.human_ceiling:
        unit_to_rows, fallback_round_count = _human_unit_map(
            human_filtered_records,
            unit_mode=args.human_ceiling_unit,
        )
        bin_scheme = args.human_ceiling_bin_scheme
        bin_edges: np.ndarray | None = None
        if bin_scheme == "quantile":
            bin_edges = _initial_bin_edges(
                human_filtered,
                n_quantiles=args.human_ceiling_init_quantiles,
            )

        leave_one_out_rows = _leave_one_unit_out_rows(
            unit_to_rows=unit_to_rows,
            node_ids=common_node_ids,
            bin_scheme=bin_scheme,
            bin_edges=bin_edges,
            min_train_per_bin=args.human_ceiling_min_train_per_bin,
            min_test_per_bin=args.human_ceiling_min_test_per_bin,
        )
        if leave_one_out_rows:
            _write_csv(
                leave_one_out_path,
                leave_one_out_rows,
                list(leave_one_out_rows[0].keys()),
            )

        ceiling_metric_names = [
            "bin_occupancy_js_distance",
            "unconditional_target_delta_wasserstein",
            "unconditional_mean_node_delta_wasserstein",
            "unconditional_human_like_score",
            "conditional_target_delta_wasserstein",
            "conditional_mean_node_delta_wasserstein",
            "conditional_human_like_score",
            "conditional_coverage",
            "conditional_evaluable_bins",
            "conditional_test_bins",
        ]
        ceiling_summary_rows = _summarize_metric_rows(
            rows=leave_one_out_rows,
            method="leave_one_unit_out",
            metric_names=ceiling_metric_names,
        )
        if ceiling_summary_rows:
            _write_csv(
                ceiling_summary_path,
                ceiling_summary_rows,
                list(ceiling_summary_rows[0].keys()),
            )

        if fallback_round_count > 0 and args.human_ceiling_unit == "target-participant":
            print(
                "Human ceiling note:",
                f"{fallback_round_count} rows missing participant ids; "
                "fell back to round-level units for those rows.",
            )

        summary_lookup = {
            str(row["metric"]): float(row["mean"]) for row in ceiling_summary_rows
        }
        if summary_lookup:
            print_table(
                [
                    {
                        "metric": "unconditional_human_like_score",
                        "mean": round(
                            float(
                                summary_lookup.get(
                                    "unconditional_human_like_score",
                                    math.nan,
                                )
                            ),
                            4,
                        ),
                    },
                    {
                        "metric": "conditional_human_like_score",
                        "mean": round(
                            float(
                                summary_lookup.get(
                                    "conditional_human_like_score",
                                    math.nan,
                                )
                            ),
                            4,
                        ),
                    },
                    {
                        "metric": "conditional_coverage",
                        "mean": round(
                            float(summary_lookup.get("conditional_coverage", math.nan)),
                            4,
                        ),
                    },
                    {
                        "metric": "bin_occupancy_js_distance",
                        "mean": round(
                            float(
                                summary_lookup.get(
                                    "bin_occupancy_js_distance",
                                    math.nan,
                                )
                            ),
                            4,
                        ),
                    },
                ],
                columns=["metric", "mean"],
                title="Human Ceiling (Leave-One-Out)",
            )

    print(f"Completed outputs for prefix: {Path(args.output_prefix).resolve()}")


if __name__ == "__main__":
    main()
