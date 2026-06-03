"""
Compare simulator belief-update shapes against human serial-question updates.

This script is designed for population-level "human-likeness" evaluation.
It compares per-message belief deltas from:
1) a structure-conditioned LLM target (`llm_target_use_bayes_structure=True`)
2) the full simulated target (`roles.simulated_target`)
3) optional vanilla LLM target (`roles.llm_target` with no Bayes structure)
against a human reference corpus.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import wasserstein_distance

from simulation.human_likeness import (
    RoundTrajectory,
    load_serial_trajectories,
    parse_min_date,
    select_simulator,
)
from simulation.human_likeness_eval.atomizer_metrics import (
    _atomizer_alignment_rows_for_corpus,
    _atomizer_alignment_summary_row,
    _atomizer_proposition_bias_rows,
    _atomizer_proposition_bias_summary_rows,
)
from simulation.human_likeness_eval.corpus_variants import (
    _corpora_with_instruction_variants_for_fan,
    _corpora_with_persona_variants_for_fan,
    _corpora_with_policy_variants_for_fan,
    _corpora_with_simulator_model_variants_for_fan,
    _corpus_sort_key,
    _filter_corpora_by_fan_policy_model,
    _filter_corpora_for_fan,
    _split_persona_variant_corpus,
    _split_policy_instruction_variant_corpus,
    _split_policy_variant_corpus,
    _split_simulator_model_variant_corpus,
)
from simulation.human_likeness_eval.trajectory_metrics import (
    _belief_bin_sort_key,
    _belief_trajectory_values,
    _bootstrap_primary,
    _common_turns_with_min_n,
    _corpus_secondary_stats,
    _corpus_summary_row,
    _flatten_updates,
    _flatten_updates_for_turns,
    _histogram_edges_for_arrays,
    _initial_belief_bin_from_value,
    _length_matched_pooled_w1,
    _movement_summary_row,
    _pooled_jsd_from_arrays,
    _pooled_w1,
    _prop_weighted_w1,
    _proposition_stance_delta_rows,
    _proposition_stance_gap_vs_baseline_rows,
    _round_dynamics_row,
    _turn_index_jsd,
    _updates_by_turn,
)
from simulation.human_trajectory_clusters import (
    HumanTrajectoryClusterModel,
    classify_updates,
    load_human_trajectory_cluster_model,
)

from .simulator_common import (
    add_common_human_simulator_filter_args,
    add_include_vanilla_llm_target_arg,
    select_matched_human_structure_full_vanilla_from_args,
    selector_kwargs_from_args,
)
from .simulator_plot_style import (
    COMPARISON_CORPUS_COLOR_MAP,
    COMPARISON_CORPUS_LABEL_MAP,
    COMPARISON_CORPUS_ORDER_WITH_NO_RHETORIC,
    CORE_COMPARISON_CORPUS_ORDER,
)
from .stats import bootstrap_mean_ci, bootstrap_statistic_ci
from .tables import print_table
from .utils import safe_slug

# Use a non-interactive backend for CLI stability in headless environments.
plt.switch_backend("Agg")

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_RESULTS_DIR = Path("results")
ROUND_DYNAMICS_COLUMNS = [
    "corpus",
    "trajectory_index",
    "n_turns",
    "total_delta",
    "abs_total_delta",
    "raw_belief_delta",
    "initial_belief",
    "final_belief",
    "supports_proposition",
    "stance",
    "has_up_step",
    "has_down_step",
    "both_directions",
    "sign_changes",
    "max_up_step",
    "max_down_step",
    "abs_max_step",
    "source_path",
    "source_line_index",
    "source_round_index",
    "proposition",
]
CELL_TOTAL_DELTA_COLUMNS = [
    "corpus",
    "init_belief_bin",
    "n_rounds",
    "mean_total_delta",
    "mean_total_delta_ci_low",
    "mean_total_delta_ci_high",
    "mean_total_delta_sem",
    "median_total_delta",
    "mean_abs_total_delta",
    "toward_round_rate",
    "away_round_rate",
    "near_zero_round_rate",
]
PROPOSITION_STANCE_DELTA_COLUMNS = [
    "corpus",
    "proposition",
    "stance",
    "n_rounds",
    "mean_total_delta",
    "median_total_delta",
    "mean_abs_total_delta",
    "toward_round_rate",
    "away_round_rate",
    "near_zero_round_rate",
]
PROPOSITION_STANCE_GAP_COLUMNS = [
    "baseline_corpus",
    "comparator_corpus",
    "proposition",
    "stance",
    "baseline_n_rounds",
    "comparator_n_rounds",
    "baseline_mean_total_delta",
    "comparator_mean_total_delta",
    "mean_total_delta_gap_baseline_minus_comparator",
]
CHARACTERISTIC_TRACE_COLUMNS = [
    "corpus",
    "label",
    *ROUND_DYNAMICS_COLUMNS[1:-1],
    "trace_path",
    "proposition",
]
TRAJECTORY_FAN_COLUMNS = [
    "corpus",
    "turn",
    "n",
    "mean",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
]
TRAJECTORY_FAN_NORMALIZED_COLUMNS = [
    "corpus",
    "x_norm",
    "n",
    "mean",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
]
ATOMIZER_ALIGNMENT_SUMMARY_COLUMNS = [
    "corpus",
    "rounds_total",
    "rounds_with_trace",
    "rounds_with_target_directed_atoms",
    "atoms_total",
    "atoms_target_directed",
    "atoms_target_directed_non_neutral",
    "mean_p_support_target_directed",
    "neutral_rate_target_directed",
    "aligned_rate_target_directed_non_neutral",
    "balanced_aligned_rate",
    "symmetry_gap_supports_true_minus_false",
    "supports_true_n",
    "supports_true_aligned_rate",
    "supports_false_n",
    "supports_false_aligned_rate",
]
ATOMIZER_ALIGNMENT_ATOM_COLUMNS = [
    "corpus",
    "trajectory_index",
    "turn_index",
    "atom_index",
    "proposition",
    "supports_proposition",
    "p_support",
    "is_neutral",
    "is_aligned",
    "text_span",
    "source_path",
    "source_line_index",
    "source_round_index",
]
ATOMIZER_ALIGNMENT_PROPOSITION_COLUMNS = [
    "corpus",
    "base_corpus",
    "policy_model",
    "proposition",
    "supports_proposition",
    "n_atoms",
    "mean_p_support",
    "mean_goal_alignment",
    "mean_signed_goal_alignment",
    "aligned_rate_non_neutral",
]
ATOMIZER_ALIGNMENT_PROPOSITION_SUMMARY_COLUMNS = [
    "corpus",
    "base_corpus",
    "policy_model",
    "supports_proposition",
    "n_propositions",
    "mean_of_prop_mean_goal_alignment",
    "std_of_prop_mean_goal_alignment",
    "min_prop_mean_goal_alignment",
    "max_prop_mean_goal_alignment",
]
BASE_CORPUS_COLOR_MAP: dict[str, str] = dict(COMPARISON_CORPUS_COLOR_MAP)
BASE_CORPUS_LABEL_MAP: dict[str, str] = dict(COMPARISON_CORPUS_LABEL_MAP)
FAN_CORPUS_CHOICES = (*COMPARISON_CORPUS_ORDER_WITH_NO_RHETORIC,)
DEFAULT_FAN_POLICY_SELECTOR = "all"


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate which simulator produces belief-update distributions "
            "closer to human serial-question data."
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
        help_text=(
            "Deprecated: ignored. Vanilla llm_target rounds are included "
            "automatically when available after filtering."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for primary-distance CIs.",
    )
    parser.add_argument(
        "--length-match-draws",
        type=int,
        default=500,
        help=(
            "Monte Carlo draws for pooled Wasserstein after matching simulator "
            "trajectory lengths to human round lengths."
        ),
    )
    parser.add_argument(
        "--plot-min-n-per-turn",
        type=int,
        default=10,
        help=(
            "Minimum samples required at a turn to include that turn in "
            "trajectory/per-turn plots."
        ),
    )
    parser.add_argument(
        "--plot-individual-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render one overlaid spread figure per corpus using "
            "persuader-relative belief trajectories."
        ),
    )
    parser.add_argument(
        "--individual-trajectory-max-per-corpus",
        type=int,
        default=0,
        help=(
            "Optional cap on trajectories drawn per corpus in the "
            "spread plot. Default 0 plots all rounds."
        ),
    )
    parser.add_argument(
        "--metric-min-n-per-turn",
        type=int,
        default=0,
        help=(
            "Optional minimum per-turn sample count used to compute an additional "
            "stable-turn pooled Wasserstein metric. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--jsd-bins",
        type=int,
        default=31,
        help=(
            "Number of fixed bins for pooled and turn-index Jensen-Shannon "
            "divergence metrics."
        ),
    )
    parser.add_argument(
        "--normalized-grid-points",
        type=int,
        default=21,
        help=(
            "Number of points on [0,1] for normalized-time trajectory fan "
            "summaries and plot."
        ),
    )
    parser.add_argument(
        "--include-turn-jsd-diagnostics",
        action="store_true",
        help=(
            "Print and export per-turn JSD diagnostics. Disabled by default "
            "because primary JSD is pooled over all turns."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for bootstrap sampling.",
    )
    parser.add_argument(
        "--movement-epsilon",
        type=float,
        default=0.01,
        help=(
            "Absolute epsilon for classifying a per-turn or round-total movement "
            "as near-zero."
        ),
    )
    parser.add_argument(
        "--characteristic-rounds-per-corpus",
        type=int,
        default=3,
        help=(
            "Number of characteristic round traces to export per corpus. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_DIR / "simulator_human_likeness",
        help="Prefix for CSV outputs (suffixes are appended).",
    )
    parser.add_argument(
        "--cluster-model-path",
        type=Path,
        default=None,
        help=(
            "Optional path to human trajectory cluster model JSON. "
            "Uses packaged default when omitted."
        ),
    )
    parser.add_argument(
        "--enable-cluster-addendum",
        action="store_true",
        help=(
            "Enable per-cluster similarity addendum and cluster-mix plots. "
            "Default is off."
        ),
    )
    parser.add_argument(
        "--fan-split-persona",
        action="store_true",
        help=(
            "Split fan-plot corpus lines by simulated-target persona when "
            "multiple personas are present in a corpus. Default is off."
        ),
    )
    parser.add_argument(
        "--fan-corpora",
        type=str,
        default="all",
        help=(
            "Comma-separated fan corpus keys to include. Use 'all' to keep "
            "default behavior. Choices: " + ", ".join(FAN_CORPUS_CHOICES)
        ),
    )
    parser.add_argument(
        "--fan-max-turns",
        type=int,
        default=None,
        help=(
            "Optional max number of persuader turns used by fan-plot outputs "
            "(trajectory fan CSV/PNG and normalized fan CSV/PNG)."
        ),
    )
    parser.add_argument(
        "--fan-policy-models",
        type=str,
        default=DEFAULT_FAN_POLICY_SELECTOR,
        help=(
            "Comma-separated policy model IDs to include in fan outputs. "
            "Use 'all' to keep every policy variant."
        ),
    )
    parser.add_argument(
        "--fan-show-mean-error-bars",
        action="store_true",
        help=(
            "Render fan plots with mean lines and IQR error bars "
            "(q25/q75) instead of median-only lines."
        ),
    )
    return parser.parse_args()


def _parse_fan_corpora_arg(raw_value: str) -> set[str]:
    """
    Parse and validate the --fan-corpora selector.

    Args:
        raw_value: Raw comma-separated value from CLI.

    Returns:
        Set of selected corpus keys. Empty set means all.

    Raises:
        ValueError: If an unknown corpus key is supplied.
    """
    normalized = raw_value.strip().lower()
    if normalized in {"", "all"}:
        return set()
    selected: set[str] = set()
    for token in raw_value.split(","):
        key = token.strip()
        if not key:
            continue
        if key not in FAN_CORPUS_CHOICES:
            valid = ", ".join(FAN_CORPUS_CHOICES)
            raise ValueError(f"Unknown fan corpus '{key}'. Valid values: {valid}")
        selected.add(key)
    return selected


def _truncate_rows_for_fan(
    rows: list[RoundTrajectory], max_turns: int | None
) -> list[RoundTrajectory]:
    """
    Truncate trajectory updates for fan outputs.

    Args:
        rows: Round trajectories.
        max_turns: Optional max number of updates to keep.

    Returns:
        Rows with updates truncated to first ``max_turns`` entries.
    """
    if max_turns is None:
        return rows
    bounded_max_turns = max(0, int(max_turns))
    if bounded_max_turns == 0:
        return []
    return [
        replace(row, updates=tuple(row.updates[:bounded_max_turns]))
        for row in rows
        if tuple(row.updates[:bounded_max_turns])
    ]


def _parse_fan_policy_models_arg(raw_value: str) -> set[str]:
    """
    Parse and normalize the --fan-policy-models selector.

    Args:
        raw_value: Raw CLI value.

    Returns:
        Set of policy model IDs to keep. Empty means all.
    """
    normalized = raw_value.strip().lower()
    if normalized in {"", "all"}:
        return set()
    selected: set[str] = set()
    for token in raw_value.split(","):
        value = token.strip()
        if value:
            selected.add(value)
    return selected


def _coerce_sim_target_effect_scale(
    row: RoundTrajectory,
    *,
    default: float = 1.0,
) -> float:
    """
    Read the simulated-target effect scale from one trajectory condition.

    Args:
        row: Round trajectory row.
        default: Fallback value when scale is absent.

    Returns:
        Positive effect scale value.
    """
    raw = row.condition.simulated_target_effect_scale
    if raw is None:
        return float(default)
    value = float(raw)
    if value <= 0.0:
        return float(default)
    return float(value)


def _coerce_sim_target_verbalize_beliefs(
    row: RoundTrajectory,
    *,
    default: bool = False,
) -> bool:
    """
    Read the simulated-target verbalize-beliefs flag from one trajectory condition.

    Args:
        row: Round trajectory row.
        default: Fallback value when unavailable.

    Returns:
        Boolean verbalize-beliefs flag.
    """
    raw = row.condition.simulated_target_verbalize_beliefs
    if isinstance(raw, bool):
        return raw
    return bool(default)


def _effect_scale_slug(value: float) -> str:
    """
    Convert an effect-scale value into a compact corpus-name slug.

    Args:
        value: Positive effect scale.

    Returns:
        File-safe slug such as ``1p5``.
    """
    return f"{float(value):g}".replace("-", "neg_").replace(".", "p")


def _effect_scale_from_slug(slug: str) -> str:
    """
    Convert an effect-scale slug back into display text.

    Args:
        slug: Slug produced by ``_effect_scale_slug``.

    Returns:
        Human-readable numeric text.
    """
    return str(slug).replace("neg_", "-").replace("p", ".")


def _full_no_rhetoric_variant_corpora(
    rows: list[RoundTrajectory],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """
    Split no-rhetoric rows into variant corpora by scale and verbalization.

    Args:
        rows: Full no-rhetoric simulator rows.

    Returns:
        List of ``(corpus_name, rows)`` tuples sorted by effect scale and
        verbalize-beliefs flag.
    """
    if not rows:
        return []
    grouped: dict[tuple[float, bool], list[RoundTrajectory]] = {}
    for row in rows:
        scale = _coerce_sim_target_effect_scale(row)
        verbalize = _coerce_sim_target_verbalize_beliefs(row)
        grouped.setdefault((scale, verbalize), []).append(row)
    if len(grouped) == 1 and (1.0, False) in grouped:
        return [("full_no_rhetoric_target", grouped[(1.0, False)])]

    scales = sorted({scale for scale, _ in grouped})
    verbalize_values = sorted({flag for _, flag in grouped})

    variants: list[tuple[str, list[RoundTrajectory]]] = []
    for scale in scales:
        for verbalize in verbalize_values:
            key = (scale, verbalize)
            if key not in grouped:
                continue
            parts = ["full_no_rhetoric_target"]
            include_scale = len(scales) > 1 or not np.isclose(scale, 1.0)
            include_verbalize = len(verbalize_values) > 1 or verbalize
            if include_scale:
                parts.append(f"scale_{_effect_scale_slug(scale)}")
            if include_verbalize:
                parts.append(f"verbal_{'on' if verbalize else 'off'}")
            variants.append(("_".join(parts), grouped[key]))
    return variants


def _corpus_display_label(corpus: str) -> str:
    """
    Build a human-readable legend label for a corpus key.

    Args:
        corpus: Corpus key.

    Returns:
        Display label.
    """
    (
        base_with_simulator_model_and_instruction_and_persona,
        policy_model,
    ) = _split_policy_variant_corpus(corpus)
    (
        base_with_instruction_and_persona,
        simulator_model,
    ) = _split_simulator_model_variant_corpus(
        base_with_simulator_model_and_instruction_and_persona
    )
    (
        base_with_persona,
        instruction_variant,
    ) = _split_policy_instruction_variant_corpus(base_with_instruction_and_persona)
    base_corpus, persona = _split_persona_variant_corpus(base_with_persona)
    instruction_label: str | None = None
    if instruction_variant == "on":
        instruction_label = "Instruction=On"
    elif instruction_variant == "off":
        instruction_label = "Instruction=Off"
    elif instruction_variant == "unknown":
        instruction_label = "Instruction=Unknown"
    simulator_label: str | None = None
    if simulator_model is not None:
        simulator_label = f"Sim={simulator_model}"
    if base_corpus in BASE_CORPUS_LABEL_MAP:
        label = BASE_CORPUS_LABEL_MAP[base_corpus]
        if persona is not None:
            label = f"{label} (Persona={persona})"
        if instruction_label is not None:
            label = f"{label} ({instruction_label})"
        if simulator_label is not None:
            label = f"{label} ({simulator_label})"
        if policy_model is not None:
            return f"{label} (Policy={policy_model})"
        return label
    full_no_rhetoric_match = re.match(
        (
            r"^full_no_rhetoric_target"
            r"(?:_scale_(?P<scale>[a-zA-Z0-9_]+))?"
            r"(?:_verbal_(?P<verbal>on|off))?$"
        ),
        base_corpus,
    )
    if full_no_rhetoric_match:
        qualifiers: list[str] = []
        scale_slug = full_no_rhetoric_match.group("scale")
        verbal_slug = full_no_rhetoric_match.group("verbal")
        if scale_slug:
            qualifiers.append(f"Scale={_effect_scale_from_slug(scale_slug)}")
        if verbal_slug:
            qualifiers.append(f"Verbalize={'On' if verbal_slug == 'on' else 'Off'}")
        if qualifiers:
            label = f"BN Target (No Rhetoric) ({', '.join(qualifiers)})"
        else:
            label = "BN Target (No Rhetoric)"
        if persona is not None:
            label = f"{label} (Persona={persona})"
        if instruction_label is not None:
            label = f"{label} ({instruction_label})"
        if simulator_label is not None:
            label = f"{label} ({simulator_label})"
        if policy_model is not None:
            return f"{label} (Policy={policy_model})"
        return label

    if persona is not None:
        label = f"{base_corpus} (Persona={persona})"
    else:
        label = base_corpus
    if instruction_label is not None:
        label = f"{label} ({instruction_label})"
    if simulator_label is not None:
        label = f"{label} ({simulator_label})"
    if policy_model is not None:
        return f"{label} (Policy={policy_model})"
    return label


def _corpus_color_map(corpora: list[str]) -> dict[str, str]:
    """
    Build a deterministic corpus-to-color mapping.

    Args:
        corpora: Corpus keys present in a plot.

    Returns:
        Mapping from corpus key to hex color.
    """
    color_map: dict[str, str] = dict(BASE_CORPUS_COLOR_MAP)
    variant_prefix = "full_no_rhetoric_target_"
    variant_corpora = sorted(
        c
        for c in corpora
        if c.startswith(variant_prefix) and c != "full_no_rhetoric_target"
    )
    if variant_corpora:
        cmap = plt.get_cmap("plasma")
        denominator = max(1, len(variant_corpora) - 1)
        for index, corpus in enumerate(variant_corpora):
            color_map[corpus] = matplotlib.colors.to_hex(
                cmap(float(index / denominator))
            )
    unknown_corpora = sorted(
        corpus
        for corpus in corpora
        if corpus not in color_map and corpus not in variant_corpora
    )
    if unknown_corpora:
        cmap = plt.get_cmap("tab20")
        denominator = max(1, len(unknown_corpora) - 1)
        for index, corpus in enumerate(unknown_corpora):
            color_map[corpus] = matplotlib.colors.to_hex(
                cmap(float(index / denominator))
            )
    return color_map


def _trajectory_values_by_turn(rows: list[RoundTrajectory]) -> dict[int, np.ndarray]:
    """
    Group cumulative persuader-relative movement by turn index.

    Turn index 0 is always zero movement from the initial state.

    Args:
        rows: Trajectory rows.

    Returns:
        Mapping turn index (0-based) -> cumulative movement values.
    """
    grouped: dict[int, list[float]] = {}
    for row in rows:
        cumulative = 0.0
        grouped.setdefault(0, []).append(0.0)
        for idx, value in enumerate(row.updates, start=1):
            cumulative += float(value)
            grouped.setdefault(idx, []).append(cumulative)
    return {key: np.asarray(values, dtype=float) for key, values in grouped.items()}


def _trajectory_fan_rows(
    *,
    corpus: str,
    rows: list[RoundTrajectory],
) -> list[dict[str, object]]:
    """
    Summarize cumulative-trajectory quantiles by turn for plotting/export.

    Args:
        corpus: Corpus label.
        rows: Trajectory rows.

    Returns:
        Per-turn quantile rows for the corpus.
    """
    by_turn = _trajectory_values_by_turn(rows)
    out_rows: list[dict[str, object]] = []
    for turn in sorted(by_turn):
        values = by_turn[turn]
        if values.size == 0:
            continue
        out_rows.append(
            {
                "corpus": corpus,
                "turn": int(turn),
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
            }
        )
    return out_rows


def _trajectory_fan_rows_normalized(
    *,
    corpus: str,
    rows: list[RoundTrajectory],
    grid_points: int,
) -> list[dict[str, object]]:
    """
    Summarize cumulative trajectories on a normalized time grid.

    Args:
        corpus: Corpus label.
        rows: Trajectory rows.
        grid_points: Number of interpolation points on [0,1].

    Returns:
        Quantile rows keyed by normalized position.
    """
    grid, matrix = _normalized_trajectory_matrix(rows=rows, grid_points=grid_points)
    if grid.size == 0 or matrix.size == 0:
        return []
    out_rows: list[dict[str, object]] = []
    for idx, x_val in enumerate(grid):
        col = matrix[:, idx]
        out_rows.append(
            {
                "corpus": corpus,
                "x_norm": float(x_val),
                "n": int(col.size),
                "mean": float(np.mean(col)),
                "q10": float(np.quantile(col, 0.10)),
                "q25": float(np.quantile(col, 0.25)),
                "median": float(np.quantile(col, 0.50)),
                "q75": float(np.quantile(col, 0.75)),
                "q90": float(np.quantile(col, 0.90)),
            }
        )
    return out_rows


def _normalized_trajectory_matrix(
    *,
    rows: list[RoundTrajectory],
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an interpolated cumulative-trajectory matrix on normalized time.

    Args:
        rows: Trajectory rows.
        grid_points: Number of interpolation points on [0,1].

    Returns:
        Tuple of (grid, matrix) where matrix shape is (n_rows, grid_points).
    """
    if grid_points < 2 or not rows:
        return np.asarray([], dtype=float), np.empty((0, 0), dtype=float)
    grid = np.linspace(0.0, 1.0, int(grid_points), dtype=float)
    samples: list[np.ndarray] = []
    for row in rows:
        updates = np.asarray(row.updates, dtype=float)
        cumulative = np.concatenate(
            [
                np.asarray([0.0], dtype=float),
                np.cumsum(updates, dtype=float),
            ]
        )
        if cumulative.size < 2:
            x = np.asarray([0.0, 1.0], dtype=float)
            y = np.asarray([0.0, 0.0], dtype=float)
        else:
            x = np.linspace(0.0, 1.0, cumulative.size, dtype=float)
            y = cumulative
        samples.append(np.interp(grid, x, y))
    return grid, np.vstack(samples)


def _plot_trajectory_fan(
    *,
    path: Path,
    fan_rows: list[dict[str, object]],
    min_n_per_turn: int,
    corpus_counts: dict[str, int] | None = None,
    show_mean_error_bars: bool = False,
) -> None:
    """
    Plot cumulative trajectory fan chart across corpora.

    Args:
        path: Destination image path.
        fan_rows: Rows produced by ``_trajectory_fan_rows`` across corpora.
        min_n_per_turn: Minimum count required to plot a turn.
        corpus_counts: Optional corpus -> round count for legend labels.
        show_mean_error_bars: When True, plot mean with IQR error bars.
    """
    if not fan_rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.25, 3.33))

    corpus_to_rows: dict[str, list[dict[str, object]]] = {}
    for row in fan_rows:
        corpus = str(row["corpus"])
        corpus_to_rows.setdefault(corpus, []).append(row)

    color_map = _corpus_color_map(list(corpus_to_rows))

    ordered_corpora = sorted(corpus_to_rows, key=_corpus_sort_key)
    policy_variants_by_base: dict[str, set[str]] = defaultdict(set)
    for corpus in ordered_corpora:
        base_corpus, policy_model = _split_policy_variant_corpus(corpus)
        if policy_model is not None:
            policy_variants_by_base[base_corpus].add(policy_model)
    plotted_turns: set[int] = set()
    for corpus in ordered_corpora:
        rows = [
            row
            for row in corpus_to_rows.get(corpus, [])
            if int(row.get("n", 0)) >= int(min_n_per_turn)
        ]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda item: int(item["turn"]))
        turns = np.asarray([int(item["turn"]) for item in rows_sorted], dtype=float)
        plotted_turns.update(int(turn) for turn in turns.tolist())
        median = np.asarray(
            [float(item["median"]) for item in rows_sorted], dtype=float
        )
        mean = np.asarray([float(item["mean"]) for item in rows_sorted], dtype=float)
        q25 = np.asarray([float(item["q25"]) for item in rows_sorted], dtype=float)
        q75 = np.asarray([float(item["q75"]) for item in rows_sorted], dtype=float)

        color = color_map.get(corpus, "#555555")
        base_corpus, policy_model = _split_policy_variant_corpus(corpus)
        if (
            policy_model is not None
            and len(policy_variants_by_base.get(base_corpus, set())) == 1
        ):
            label = _corpus_display_label(base_corpus)
        else:
            label = _corpus_display_label(corpus)
        if corpus_counts is not None and corpus in corpus_counts:
            label = f"{label} (N={int(corpus_counts[corpus])})"
        if show_mean_error_bars:
            yerr = np.vstack([mean - q25, q75 - mean])
            axis.errorbar(
                turns,
                mean,
                yerr=yerr,
                color=color,
                linewidth=1.8,
                elinewidth=1.0,
                capsize=2.0,
                label=label,
            )
        else:
            axis.plot(turns, median, color=color, linewidth=2.0, label=label)

    axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Persuader Turn")
    axis.set_ylabel("Cumulative\nBelief Movement")
    if plotted_turns:
        ticks = sorted(plotted_turns)
        axis.set_xticks(ticks)
        axis.set_xlim(float(min(ticks)) - 0.05, float(max(ticks)) + 0.15)
    axis.set_title("")
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, int(np.ceil(len(ordered_corpora) / 2.0))),
    )
    axis.grid(alpha=0.20, linestyle=":")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_trajectory_fan_normalized(
    *,
    path: Path,
    fan_rows: list[dict[str, object]],
    corpus_counts: dict[str, int] | None = None,
    show_mean_error_bars: bool = False,
) -> None:
    """
    Plot normalized-time cumulative trajectory fan chart.

    Args:
        path: Destination image path.
        fan_rows: Rows from ``_trajectory_fan_rows_normalized``.
        corpus_counts: Optional corpus -> round count for legend labels.
        show_mean_error_bars: When True, plot mean with IQR error bars.
    """
    if not fan_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.25, 3.33))
    corpus_to_rows: dict[str, list[dict[str, object]]] = {}
    for row in fan_rows:
        corpus_to_rows.setdefault(str(row["corpus"]), []).append(row)
    color_map = _corpus_color_map(list(corpus_to_rows))
    ordered_corpora = sorted(corpus_to_rows, key=_corpus_sort_key)
    policy_variants_by_base: dict[str, set[str]] = defaultdict(set)
    for corpus in ordered_corpora:
        base_corpus, policy_model = _split_policy_variant_corpus(corpus)
        if policy_model is not None:
            policy_variants_by_base[base_corpus].add(policy_model)
    for corpus in ordered_corpora:
        rows = corpus_to_rows.get(corpus, [])
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda item: float(item["x_norm"]))
        x = np.asarray([float(item["x_norm"]) for item in rows_sorted], dtype=float)
        median = np.asarray(
            [float(item["median"]) for item in rows_sorted], dtype=float
        )
        mean = np.asarray([float(item["mean"]) for item in rows_sorted], dtype=float)
        q25 = np.asarray([float(item["q25"]) for item in rows_sorted], dtype=float)
        q75 = np.asarray([float(item["q75"]) for item in rows_sorted], dtype=float)
        color = color_map.get(corpus, "#555555")
        base_corpus, policy_model = _split_policy_variant_corpus(corpus)
        if (
            policy_model is not None
            and len(policy_variants_by_base.get(base_corpus, set())) == 1
        ):
            label = _corpus_display_label(base_corpus)
        else:
            label = _corpus_display_label(corpus)
        if corpus_counts is not None and corpus in corpus_counts:
            label = f"{label} (N={int(corpus_counts[corpus])})"
        if show_mean_error_bars:
            yerr = np.vstack([mean - q25, q75 - mean])
            axis.errorbar(
                x,
                mean,
                yerr=yerr,
                color=color,
                linewidth=1.8,
                elinewidth=1.0,
                capsize=2.0,
                label=label,
            )
        else:
            axis.plot(x, median, color=color, linewidth=2.0, label=label)
    axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Normalized Conversation Progress")
    axis.set_ylabel("Cumulative\nBelief Movement")
    axis.set_title("")
    axis.set_xlim(0.0, 1.0)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, int(np.ceil(len(ordered_corpora) / 2.0))),
    )
    axis.grid(alpha=0.20, linestyle=":")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_per_turn_delta_boxes(
    *,
    path: Path,
    human_rows: list[RoundTrajectory],
    structure_rows: list[RoundTrajectory],
    full_rows: list[RoundTrajectory],
    vanilla_rows: list[RoundTrajectory],
    include_vanilla: bool,
    min_n_per_turn: int,
    corpus_counts: dict[str, int] | None = None,
) -> None:
    """
    Plot per-turn delta distributions as side-by-side boxplots.

    Args:
        path: Destination image path.
        human_rows: Human trajectories.
        structure_rows: Structure trajectories.
        full_rows: Full trajectories.
        vanilla_rows: Vanilla trajectories.
        include_vanilla: Whether to include vanilla corpus.
        min_n_per_turn: Minimum sample count required per corpus at a turn.
        corpus_counts: Optional corpus -> round count for legend labels.
    """
    by_turn_human = _updates_by_turn(human_rows)
    by_turn_structure = _updates_by_turn(structure_rows)
    by_turn_full = _updates_by_turn(full_rows)
    by_turn_vanilla = _updates_by_turn(vanilla_rows) if include_vanilla else {}

    turns = set(by_turn_human) & set(by_turn_structure) & set(by_turn_full)
    if include_vanilla:
        turns = turns & set(by_turn_vanilla)
    valid_turns: list[int] = []
    for turn in sorted(turns):
        if by_turn_human[turn].size < min_n_per_turn:
            continue
        if by_turn_structure[turn].size < min_n_per_turn:
            continue
        if by_turn_full[turn].size < min_n_per_turn:
            continue
        if include_vanilla and by_turn_vanilla[turn].size < min_n_per_turn:
            continue
        valid_turns.append(turn)
    if not valid_turns:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    color_map: dict[str, str] = {
        "human": COMPARISON_CORPUS_COLOR_MAP["human_reference"],
        "vanilla": COMPARISON_CORPUS_COLOR_MAP["vanilla_llm_target"],
        "structure": COMPARISON_CORPUS_COLOR_MAP["structure_target"],
        "full": COMPARISON_CORPUS_COLOR_MAP["full_simulated_target"],
    }
    order = ["human", "structure", "full"]
    if include_vanilla:
        order = ["human", "vanilla", "structure", "full"]
    width = 0.18 if include_vanilla else 0.22
    offsets = np.linspace(-(len(order) - 1) / 2.0, (len(order) - 1) / 2.0, len(order))
    offset_map = {name: float(offsets[idx]) * width for idx, name in enumerate(order)}

    for name in order:
        if name == "human":
            values_by_turn = by_turn_human
        elif name == "structure":
            values_by_turn = by_turn_structure
        elif name == "full":
            values_by_turn = by_turn_full
        else:
            values_by_turn = by_turn_vanilla
        positions = [turn + offset_map[name] for turn in valid_turns]
        samples = [values_by_turn[turn] for turn in valid_turns]
        axis.boxplot(
            samples,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            medianprops={"color": "#ffffff", "linewidth": 1.5},
            boxprops={"facecolor": color_map[name], "alpha": 0.65, "linewidth": 1.0},
            whiskerprops={"color": color_map[name], "linewidth": 1.0},
            capprops={"color": color_map[name], "linewidth": 1.0},
        )

    axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Persuader Turn")
    axis.set_ylabel("Per-Turn Persuader-Relative Belief Delta")
    axis.set_title("Per-Turn Delta Distributions")
    axis.set_xticks(valid_turns)
    axis.set_xticklabels([str(turn) for turn in valid_turns])
    axis.set_xlim(min(valid_turns) - 0.6, max(valid_turns) + 0.6)
    axis.grid(alpha=0.20, linestyle=":")
    legend_labels = {
        "human": COMPARISON_CORPUS_LABEL_MAP["human_reference"],
        "vanilla": COMPARISON_CORPUS_LABEL_MAP["vanilla_llm_target"],
        "structure": COMPARISON_CORPUS_LABEL_MAP["structure_target"],
        "full": COMPARISON_CORPUS_LABEL_MAP["full_simulated_target"],
    }
    legend_handles = [
        Patch(
            facecolor=color_map[name],
            edgecolor=color_map[name],
            alpha=0.65,
            label=(
                f"{legend_labels[name]} (N={int(corpus_counts[name])})"
                if corpus_counts is not None and name in corpus_counts
                else legend_labels[name]
            ),
        )
        for name in order
    ]
    axis.legend(handles=legend_handles, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _normalized_trajectory_shape_w1(
    human_rows: list[RoundTrajectory],
    sim_rows: list[RoundTrajectory],
    *,
    grid_points: int,
    min_x: float = 0.05,
) -> float:
    """
    Compare cumulative-trajectory shape via normalized-time Wasserstein profile.

    At each normalized time point, compute Wasserstein distance between the
    human and simulator cumulative-position distributions, then average over
    time (excluding the trivial x=0 origin by default).

    Args:
        human_rows: Human trajectories.
        sim_rows: Simulator trajectories.
        grid_points: Number of normalized grid points in [0,1].
        min_x: Minimum normalized time included in the average.

    Returns:
        Mean per-grid Wasserstein distance, or NaN when unavailable.
    """
    human_grid, human_matrix = _normalized_trajectory_matrix(
        rows=human_rows,
        grid_points=grid_points,
    )
    sim_grid, sim_matrix = _normalized_trajectory_matrix(
        rows=sim_rows,
        grid_points=grid_points,
    )
    if (
        human_grid.size == 0
        or sim_grid.size == 0
        or human_matrix.size == 0
        or sim_matrix.size == 0
    ):
        return float("nan")
    if human_grid.size != sim_grid.size or not np.allclose(human_grid, sim_grid):
        return float("nan")
    distances: list[float] = []
    for idx, x_val in enumerate(human_grid):
        if float(x_val) < float(min_x):
            continue
        distances.append(
            float(wasserstein_distance(human_matrix[:, idx], sim_matrix[:, idx]))
        )
    if not distances:
        return float("nan")
    return float(np.mean(np.asarray(distances, dtype=float)))


def _cell_total_delta_rows(
    *,
    corpora: list[tuple[str, list[RoundTrajectory]]],
    epsilon: float,
    bootstrap_n: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """
    Aggregate total-delta movement by corpus and initial-belief bin.

    Args:
        corpora: Corpus rows to summarize.
        epsilon: Near-zero threshold for rates.
        bootstrap_n: Number of bootstrap resamples for CI.
        rng: Random generator used for bootstrap CI.

    Returns:
        Aggregated rows by ``corpus`` x ``init_belief_bin``.
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for corpus, rows in corpora:
        for row in rows:
            initial_raw = row.round_obj.target_initial_belief
            if isinstance(initial_raw, (int, float)):
                init_bin = _initial_belief_bin_from_value(float(initial_raw))
            else:
                init_bin = "unknown"
            total_delta = float(np.sum(np.asarray(row.updates, dtype=float)))
            grouped[(corpus, init_bin)].append(total_delta)

    rows_out: list[dict[str, object]] = []
    for corpus, init_bin in sorted(
        grouped,
        key=lambda item: (
            _corpus_sort_key(item[0]),
            _belief_bin_sort_key(str(item[1])),
        ),
    ):
        totals = np.asarray(grouped[(corpus, init_bin)], dtype=float)
        mean_total, ci_low, ci_high = bootstrap_mean_ci(
            totals,
            n_boot=max(100, int(bootstrap_n)),
            ci=0.95,
            rng=rng,
        )
        if totals.size > 1:
            sem = float(np.std(totals, ddof=1) / np.sqrt(float(totals.size)))
        else:
            sem = 0.0
        rows_out.append(
            {
                "corpus": corpus,
                "init_belief_bin": init_bin,
                "n_rounds": int(totals.size),
                "mean_total_delta": float(mean_total),
                "mean_total_delta_ci_low": float(ci_low),
                "mean_total_delta_ci_high": float(ci_high),
                "mean_total_delta_sem": float(sem),
                "median_total_delta": float(np.median(totals)),
                "mean_abs_total_delta": float(np.mean(np.abs(totals))),
                "toward_round_rate": float(np.mean(totals > epsilon)),
                "away_round_rate": float(np.mean(totals < -epsilon)),
                "near_zero_round_rate": float(np.mean(np.abs(totals) <= epsilon)),
            }
        )
    return rows_out


def _plot_cell_total_delta_bars(
    *,
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Plot mean total delta for each corpus x initial-belief-bin cell.

    Args:
        path: Output path for the PNG figure.
        rows: Rows from ``_cell_total_delta_rows``.
    """
    if not rows:
        return
    ordered = sorted(
        rows,
        key=lambda item: (
            _corpus_sort_key(str(item.get("corpus", ""))),
            _belief_bin_sort_key(str(item.get("init_belief_bin", "unknown"))),
        ),
    )

    corpora = [str(item["corpus"]) for item in ordered]
    color_map = _corpus_color_map(sorted(set(corpora), key=_corpus_sort_key))
    x = np.arange(len(ordered), dtype=float)
    y = np.asarray([float(item["mean_total_delta"]) for item in ordered], dtype=float)
    ci_low = np.asarray(
        [float(item.get("mean_total_delta_ci_low", np.nan)) for item in ordered],
        dtype=float,
    )
    ci_high = np.asarray(
        [float(item.get("mean_total_delta_ci_high", np.nan)) for item in ordered],
        dtype=float,
    )
    yerr_lower = np.maximum(0.0, y - ci_low)
    yerr_upper = np.maximum(0.0, ci_high - y)
    finite_err = np.isfinite(yerr_lower) & np.isfinite(yerr_upper)
    yerr = np.zeros((2, len(ordered)), dtype=float)
    yerr[0, finite_err] = yerr_lower[finite_err]
    yerr[1, finite_err] = yerr_upper[finite_err]
    colors = [color_map.get(str(item["corpus"]), "#777777") for item in ordered]
    labels = [
        (
            f"{_corpus_display_label(str(item['corpus']))}\n"
            f"{str(item['init_belief_bin'])} (n={int(item['n_rounds'])})"
        )
        for item in ordered
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(10.0, min(36.0, 0.45 * len(ordered) + 4.0))
    fig, axis = plt.subplots(figsize=(fig_width, 5.4))
    axis.bar(
        x,
        y,
        color=colors,
        alpha=0.88,
        width=0.8,
        yerr=yerr,
        capsize=3,
        error_kw={"elinewidth": 1.1, "ecolor": "#222222", "alpha": 0.9},
    )
    axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=28, ha="right")
    axis.set_ylabel("Mean Total Delta (Toward Persuader Is Positive)")
    axis.set_title("Total Delta by Corpus and Initial Belief Bin")
    axis.grid(alpha=0.20, linestyle=":", axis="y")

    legend_corpora = sorted(set(corpora), key=_corpus_sort_key)
    handles = [
        Patch(
            color=color_map.get(corpus, "#777777"), label=_corpus_display_label(corpus)
        )
        for corpus in legend_corpora
    ]
    axis.legend(handles=handles, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_atomizer_alignment(
    *,
    path: Path,
    summary_rows: list[dict[str, object]],
) -> None:
    """
    Plot corpus-level atomizer alignment rates.

    Args:
        path: Output figure path.
        summary_rows: Rows from `_atomizer_alignment_summary_row`.
    """
    plot_rows = [
        row
        for row in summary_rows
        if int(row.get("atoms_target_directed_non_neutral", 0)) > 0
    ]
    if not plot_rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    corpora = [str(row["corpus"]) for row in plot_rows]
    labels = [_corpus_display_label(corpus) for corpus in corpora]
    x = np.arange(len(plot_rows), dtype=float)
    width = 0.24

    all_rates = np.asarray(
        [float(row["aligned_rate_target_directed_non_neutral"]) for row in plot_rows],
        dtype=float,
    )
    supports_true_rates = np.asarray(
        [float(row["supports_true_aligned_rate"]) for row in plot_rows],
        dtype=float,
    )
    supports_false_rates = np.asarray(
        [float(row["supports_false_aligned_rate"]) for row in plot_rows],
        dtype=float,
    )

    fig, axis = plt.subplots(figsize=(10.0, 5.2))
    axis.bar(
        x - width,
        all_rates,
        width=width,
        color="#4c78a8",
        alpha=0.9,
        label="All non-neutral target-directed atoms",
    )
    axis.bar(
        x,
        supports_true_rates,
        width=width,
        color="#59a14f",
        alpha=0.9,
        label="When persuader supports proposition",
    )
    axis.bar(
        x + width,
        supports_false_rates,
        width=width,
        color="#e15759",
        alpha=0.9,
        label="When persuader opposes proposition",
    )
    axis.axhline(0.5, color="#888888", linestyle=":", linewidth=1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Aligned Rate")
    axis.set_xlabel("Corpus")
    axis.set_title("Atomizer Direction Alignment on Target-Directed Atoms")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.grid(alpha=0.2, linestyle=":", axis="y")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_atomizer_proposition_bias(
    *,
    path: Path,
    proposition_rows: list[dict[str, object]],
) -> None:
    """
    Plot proposition-level target-direction evidence spread by corpus.

    Args:
        path: Output figure path.
        proposition_rows: Rows from `_atomizer_proposition_bias_rows`.
    """
    plot_rows = [
        row
        for row in proposition_rows
        if np.isfinite(float(row.get("mean_goal_alignment", float("nan"))))
    ]
    if not plot_rows:
        return

    stance_values = [
        stance
        for stance in [True, False]
        if any(row.get("supports_proposition") is stance for row in plot_rows)
    ]
    if not stance_values:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(stance_values),
        figsize=(max(7.6, 5.2 * len(stance_values)), 5.2),
        sharey=True,
    )
    axes_array = np.asarray(axes, dtype=object).reshape(-1)

    for axis, supports in zip(axes_array, stance_values):
        stance_rows = [
            row for row in plot_rows if row.get("supports_proposition") is supports
        ]
        corpora = sorted(
            {str(row["corpus"]) for row in stance_rows},
            key=_corpus_sort_key,
        )
        if not corpora:
            axis.set_axis_off()
            continue

        color_map = _corpus_color_map(corpora)
        data: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []
        for corpus in corpora:
            values = np.asarray(
                [
                    float(row["mean_goal_alignment"])
                    for row in stance_rows
                    if str(row["corpus"]) == corpus
                ],
                dtype=float,
            )
            if values.size == 0:
                continue
            data.append(values)
            labels.append(f"{_corpus_display_label(corpus)}\n(n={int(values.size)})")
            colors.append(color_map.get(corpus, "#888888"))

        if not data:
            axis.set_axis_off()
            continue

        boxplot = axis.boxplot(
            data,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
        for median in boxplot["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.3)
        axis.axhline(0.5, color="#888888", linestyle=":", linewidth=1.0)
        axis.set_xticks(np.arange(1, len(labels) + 1, dtype=float))
        axis.set_xticklabels(labels, rotation=22, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2, linestyle=":", axis="y")
        axis.set_title(
            (
                "Persuader Supports Proposition"
                if supports
                else "Persuader Opposes Proposition"
            )
        )

    axes_array[0].set_ylabel("Proposition Mean Goal-Alignment Signal")
    fig.suptitle("Atomizer Target-Direction Signal by Proposition")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_atomizer_alignment_outputs(
    *,
    prefix: Path,
    corpora: list[tuple[str, list[RoundTrajectory]]],
) -> None:
    """
    Compute, print, and export atomizer-alignment diagnostics.

    Args:
        prefix: Output filename prefix.
        corpora: Corpus-name/rows pairs to evaluate.
    """
    summary_rows: list[dict[str, object]] = []
    atom_rows: list[dict[str, object]] = []
    for corpus, rows in corpora:
        corpus_atom_rows, counts = _atomizer_alignment_rows_for_corpus(
            corpus=corpus,
            rows=rows,
        )
        atom_rows.extend(corpus_atom_rows)
        summary_rows.append(
            _atomizer_alignment_summary_row(
                corpus=corpus,
                atom_rows=corpus_atom_rows,
                counts=counts,
            )
        )

    if summary_rows:
        print_table(
            summary_rows,
            title="\nAtomizer Target-Direction Alignment",
            columns=ATOMIZER_ALIGNMENT_SUMMARY_COLUMNS,
            aligns={
                "rounds_total": "right",
                "rounds_with_trace": "right",
                "rounds_with_target_directed_atoms": "right",
                "atoms_total": "right",
                "atoms_target_directed": "right",
                "atoms_target_directed_non_neutral": "right",
                "mean_p_support_target_directed": "right",
                "neutral_rate_target_directed": "right",
                "aligned_rate_target_directed_non_neutral": "right",
                "balanced_aligned_rate": "right",
                "symmetry_gap_supports_true_minus_false": "right",
                "supports_true_n": "right",
                "supports_true_aligned_rate": "right",
                "supports_false_n": "right",
                "supports_false_aligned_rate": "right",
            },
            formatters={
                "mean_p_support_target_directed": lambda value: f"{value:.3f}",
                "neutral_rate_target_directed": lambda value: f"{value:.3f}",
                "aligned_rate_target_directed_non_neutral": (
                    lambda value: f"{value:.3f}"
                ),
                "balanced_aligned_rate": lambda value: f"{value:.3f}",
                "symmetry_gap_supports_true_minus_false": (
                    lambda value: f"{value:+.3f}"
                ),
                "supports_true_aligned_rate": lambda value: f"{value:.3f}",
                "supports_false_aligned_rate": lambda value: f"{value:.3f}",
            },
        )

    _write_csv(
        prefix.with_name(prefix.name + "_atomizer_alignment_summary.csv"),
        summary_rows,
        ATOMIZER_ALIGNMENT_SUMMARY_COLUMNS,
    )
    _write_csv(
        prefix.with_name(prefix.name + "_atomizer_alignment_atoms.csv"),
        atom_rows,
        ATOMIZER_ALIGNMENT_ATOM_COLUMNS,
    )
    _plot_atomizer_alignment(
        path=prefix.with_name(prefix.name + "_atomizer_alignment.png"),
        summary_rows=summary_rows,
    )
    proposition_rows = _atomizer_proposition_bias_rows(atom_rows)
    proposition_summary_rows = _atomizer_proposition_bias_summary_rows(proposition_rows)
    if proposition_summary_rows:
        print_table(
            proposition_summary_rows,
            title="\nAtomizer Proposition-Level Signal Spread",
            columns=ATOMIZER_ALIGNMENT_PROPOSITION_SUMMARY_COLUMNS,
            aligns={
                "n_propositions": "right",
                "mean_of_prop_mean_goal_alignment": "right",
                "std_of_prop_mean_goal_alignment": "right",
                "min_prop_mean_goal_alignment": "right",
                "max_prop_mean_goal_alignment": "right",
            },
            formatters={
                "mean_of_prop_mean_goal_alignment": lambda value: f"{value:.3f}",
                "std_of_prop_mean_goal_alignment": lambda value: f"{value:.3f}",
                "min_prop_mean_goal_alignment": lambda value: f"{value:.3f}",
                "max_prop_mean_goal_alignment": lambda value: f"{value:.3f}",
            },
        )
    _write_csv(
        prefix.with_name(prefix.name + "_atomizer_alignment_by_proposition.csv"),
        proposition_rows,
        ATOMIZER_ALIGNMENT_PROPOSITION_COLUMNS,
    )
    _write_csv(
        prefix.with_name(
            prefix.name + "_atomizer_alignment_by_proposition_summary.csv"
        ),
        proposition_summary_rows,
        ATOMIZER_ALIGNMENT_PROPOSITION_SUMMARY_COLUMNS,
    )
    _plot_atomizer_proposition_bias(
        path=prefix.with_name(prefix.name + "_atomizer_alignment_by_proposition.png"),
        proposition_rows=proposition_rows,
    )


def _plot_individual_belief_trajectories(
    *,
    output_dir: Path,
    corpus: str,
    rows: list[RoundTrajectory],
    max_per_corpus: int,
    epsilon: float,
) -> list[dict[str, object]]:
    """
    Render one overlaid belief-trajectory spread figure for a corpus.

    Args:
        output_dir: Root output directory for trajectory figures.
        corpus: Corpus label.
        rows: Trajectory rows.
        max_per_corpus: Optional cap on number of trajectories drawn.
        epsilon: Near-zero movement threshold.

    Returns:
        One metadata row describing the generated figure.
    """
    if not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_per_corpus > 0:
        selected_rows = rows[:max_per_corpus]
    else:
        selected_rows = rows

    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    by_turn: dict[int, list[float]] = {}
    toward_rounds = 0
    away_rounds = 0
    near_zero_rounds = 0
    plotted = 0

    for row in selected_rows:
        beliefs = _belief_trajectory_values(row)
        if beliefs.size < 2:
            continue
        total_delta = float(np.sum(np.asarray(row.updates, dtype=float)))
        if total_delta > epsilon:
            color = "#2ca02c"
            toward_rounds += 1
        elif total_delta < -epsilon:
            color = "#d62728"
            away_rounds += 1
        else:
            color = "#7f7f7f"
            near_zero_rounds += 1
        turns = np.arange(beliefs.size, dtype=float)
        axis.plot(turns, beliefs, color=color, linewidth=0.9, alpha=0.14)
        for turn_index, value in enumerate(beliefs):
            by_turn.setdefault(int(turn_index), []).append(float(value))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return []

    median_turns = sorted(by_turn)
    median_values = np.asarray(
        [
            float(np.median(np.asarray(by_turn[turn_index], dtype=float)))
            for turn_index in median_turns
        ],
        dtype=float,
    )
    axis.plot(
        np.asarray(median_turns, dtype=float),
        median_values,
        color="#111111",
        linewidth=2.6,
        alpha=0.95,
        label="Median belief by turn",
    )
    axis.axhline(0.5, color="#999999", linewidth=1.0, linestyle=":")
    axis.set_xlabel("Target Serial Question Index (0=Initial)")
    axis.set_ylabel("Persuader-Relative Target Belief")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.2, linestyle=":")
    axis.set_title(
        (
            f"{corpus}: Individual Belief Trajectory Spread "
            f"(N={plotted}, toward={toward_rounds}, away={away_rounds}, "
            f"near_zero={near_zero_rounds})"
        ),
        fontsize=10,
    )
    axis.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    figure_path = output_dir / f"{corpus}_trajectory_spread.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return [
        {
            "corpus": corpus,
            "rounds_available": int(len(rows)),
            "rounds_plotted": int(plotted),
            "toward_rounds": int(toward_rounds),
            "away_rounds": int(away_rounds),
            "near_zero_rounds": int(near_zero_rounds),
            "figure_path": str(figure_path),
        }
    ]


def _select_characteristic_round_rows(
    *,
    corpus: str,
    rows: list[RoundTrajectory],
    per_corpus: int,
    epsilon: float,
) -> list[dict[str, object]]:
    """
    Select a diverse set of characteristic rounds from one corpus.

    Args:
        corpus: Corpus label.
        rows: Candidate rows.
        per_corpus: Number of rows to select.
        epsilon: Near-zero movement threshold.

    Returns:
        Selected diagnostics rows with an added ``label`` key.
    """
    if per_corpus <= 0 or not rows:
        return []

    diagnostics = [
        _round_dynamics_row(
            corpus=corpus,
            trajectory_index=index,
            row=row,
            epsilon=epsilon,
        )
        for index, row in enumerate(rows)
    ]
    by_index = {
        int(item["trajectory_index"]): item
        for item in diagnostics
        if isinstance(item.get("trajectory_index"), int)
    }

    labels_with_sorted = [
        (
            "most_toward",
            sorted(
                diagnostics,
                key=lambda item: (
                    -float(item["total_delta"]),
                    -float(item["max_up_step"]),
                ),
            ),
        ),
        (
            "most_away",
            sorted(
                diagnostics,
                key=lambda item: (
                    float(item["total_delta"]),
                    float(item["max_down_step"]),
                ),
            ),
        ),
        (
            "least_movement",
            sorted(
                diagnostics,
                key=lambda item: (
                    float(item["abs_total_delta"]),
                    float(item["abs_max_step"]),
                ),
            ),
        ),
        (
            "most_oscillatory",
            sorted(
                diagnostics,
                key=lambda item: (
                    -float(item["sign_changes"]),
                    -float(item["abs_total_delta"]),
                ),
            ),
        ),
    ]

    selected: list[dict[str, object]] = []
    used_indices: set[int] = set()
    for label, ordered in labels_with_sorted:
        for item in ordered:
            index = int(item["trajectory_index"])
            if index in used_indices:
                continue
            chosen = dict(item)
            chosen["label"] = label
            selected.append(chosen)
            used_indices.add(index)
            break
        if len(selected) >= per_corpus:
            break

    if len(selected) < per_corpus:
        remaining = sorted(
            by_index.values(),
            key=lambda item: -float(item["abs_total_delta"]),
        )
        for item in remaining:
            index = int(item["trajectory_index"])
            if index in used_indices:
                continue
            chosen = dict(item)
            chosen["label"] = "high_magnitude_fill"
            selected.append(chosen)
            used_indices.add(index)
            if len(selected) >= per_corpus:
                break
    return selected[:per_corpus]


def _write_characteristic_round_traces(
    *,
    prefix: Path,
    corpora: list[tuple[str, list[RoundTrajectory]]],
    per_corpus: int,
    epsilon: float,
) -> list[dict[str, object]]:
    """
    Export representative round traces to text files.

    Args:
        prefix: Output filename prefix.
        corpora: Corpus-name/rows pairs.
        per_corpus: Number of traces per corpus.
        epsilon: Near-zero movement threshold.

    Returns:
        Summary rows for exported trace files.
    """
    if per_corpus <= 0:
        return []
    root_dir = prefix.with_name(prefix.name + "_characteristic_rounds")
    root_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for corpus, rows in corpora:
        selected = _select_characteristic_round_rows(
            corpus=corpus,
            rows=rows,
            per_corpus=per_corpus,
            epsilon=epsilon,
        )
        if not selected:
            continue
        corpus_dir = root_dir / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)

        for rank, item in enumerate(selected, start=1):
            trajectory_index = int(item["trajectory_index"])
            if trajectory_index < 0 or trajectory_index >= len(rows):
                continue
            row = rows[trajectory_index]
            beliefs = _belief_trajectory_values(row)
            beliefs_text = (
                ", ".join(f"{float(value):.3f}" for value in beliefs)
                if beliefs.size > 0
                else "N/A"
            )
            updates_text = ", ".join(f"{float(value):+.3f}" for value in row.updates)
            file_name = (
                f"{rank:02d}_{safe_slug(str(item['label']), max_chars=28)}_"
                f"{trajectory_index:05d}_{safe_slug(row.proposition, max_chars=44)}.txt"
            )
            trace_path = corpus_dir / file_name
            header_lines = [
                f"corpus: {corpus}",
                f"label: {item['label']}",
                f"trajectory_index: {trajectory_index}",
                f"n_turns: {len(row.updates)}",
                f"total_delta: {float(item['total_delta']):+.6f}",
                f"raw_belief_delta: {float(item['raw_belief_delta']):+.6f}",
                f"has_up_step: {int(item['has_up_step'])}",
                f"has_down_step: {int(item['has_down_step'])}",
                f"sign_changes: {int(item['sign_changes'])}",
                f"source_path: {row.source_path}",
                f"source_line_index: {row.source_line_index}",
                f"source_round_index: {row.source_round_index}",
                "belief_trace_raw: " + beliefs_text,
                "updates_persuader_relative: " + updates_text,
                "",
                "round_to_string:",
                "",
                str(row.round_obj),
                "",
            ]
            trace_path.write_text("\n".join(header_lines), encoding="utf-8")

            summary = dict(item)
            summary["trace_path"] = str(trace_path)
            summary_rows.append(summary)

    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """
    Write rows to CSV.

    Args:
        path: Output CSV path.
        rows: Rows to write.
        columns: CSV column order.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rows_grouped_by_cluster(
    rows: list[RoundTrajectory],
    *,
    cluster_model: HumanTrajectoryClusterModel,
) -> dict[int, list[RoundTrajectory]]:
    """
    Group trajectory rows by human-cluster classifier assignment.

    Args:
        rows: Rows to classify.
        cluster_model: Loaded cluster model.

    Returns:
        Mapping from cluster id to rows assigned to that cluster.
    """
    grouped: dict[int, list[RoundTrajectory]] = {
        cluster_id: [] for cluster_id in cluster_model.cluster_ids
    }
    for row in rows:
        prediction = classify_updates(row.updates, model=cluster_model)
        grouped[prediction.cluster_id].append(row)
    return grouped


def _cluster_proportion_rows(
    grouped_rows: dict[str, dict[int, list[RoundTrajectory]]],
    *,
    cluster_model: HumanTrajectoryClusterModel,
) -> list[dict[str, object]]:
    """
    Build long-form cluster proportion rows for corpus comparison.

    Args:
        grouped_rows: Mapping corpus -> (cluster id -> rows).
        cluster_model: Loaded cluster model.

    Returns:
        One row per corpus x cluster with counts and proportions.
    """
    rows: list[dict[str, object]] = []
    for corpus, by_cluster in grouped_rows.items():
        total = int(sum(len(items) for items in by_cluster.values()))
        for cluster_id, cluster_name in cluster_model.cluster_items:
            count = int(len(by_cluster.get(cluster_id, [])))
            proportion = float(count / total) if total > 0 else float("nan")
            rows.append(
                {
                    "corpus": corpus,
                    "cluster_id": int(cluster_id),
                    "cluster_name": cluster_name,
                    "count": count,
                    "total": total,
                    "proportion": proportion,
                }
            )
    return rows


def _plot_cluster_proportions(
    *,
    path: Path,
    proportion_rows: list[dict[str, object]],
    cluster_model: HumanTrajectoryClusterModel,
) -> None:
    """
    Plot grouped cluster proportions by corpus.

    Args:
        path: Output figure path.
        proportion_rows: Rows from `_cluster_proportion_rows`.
        cluster_model: Loaded cluster model.
    """
    if not proportion_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    corpus_order = list(CORE_COMPARISON_CORPUS_ORDER)
    corpus_display = {
        corpus: COMPARISON_CORPUS_LABEL_MAP.get(corpus, corpus)
        for corpus in corpus_order
    }
    row_lookup: dict[tuple[str, int], float] = {}
    for row in proportion_rows:
        corpus = str(row.get("corpus", ""))
        cluster_id = int(row.get("cluster_id", -1))
        proportion = float(row.get("proportion", float("nan")))
        row_lookup[(corpus, cluster_id)] = proportion

    available_corpora = [
        corpus
        for corpus in corpus_order
        if any(
            (corpus, cluster_id) in row_lookup
            for cluster_id in cluster_model.cluster_ids
        )
    ]
    if not available_corpora:
        return

    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    x_positions = np.arange(len(cluster_model.cluster_ids), dtype=float)
    width = 0.75 / max(1, len(available_corpora))
    offsets = np.linspace(
        -0.5 * (len(available_corpora) - 1),
        0.5 * (len(available_corpora) - 1),
        len(available_corpora),
    )
    color_map = {
        corpus: COMPARISON_CORPUS_COLOR_MAP.get(corpus, "#888888")
        for corpus in corpus_order
    }
    for idx, corpus in enumerate(available_corpora):
        values = np.asarray(
            [
                row_lookup.get((corpus, cluster_id), 0.0)
                for cluster_id in cluster_model.cluster_ids
            ],
            dtype=float,
        )
        positions = x_positions + float(offsets[idx]) * width
        axis.bar(
            positions,
            values,
            width=width * 0.92,
            label=corpus_display.get(corpus, corpus),
            color=color_map.get(corpus, "#888888"),
            alpha=0.85,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([f"C{cluster_id}" for cluster_id in cluster_model.cluster_ids])
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Human trajectory cluster")
    axis.set_ylabel("Proportion of rounds")
    axis.set_title("Cluster proportion comparison by corpus")
    axis.grid(alpha=0.2, linestyle=":", axis="y")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _cluster_init_bin_rows(
    grouped_rows: dict[str, dict[int, list[RoundTrajectory]]],
    *,
    cluster_model: HumanTrajectoryClusterModel,
) -> list[dict[str, object]]:
    """
    Build corpus x cluster x initial-belief-bin composition rows.

    Args:
        grouped_rows: Mapping corpus -> (cluster id -> rows).
        cluster_model: Loaded cluster model.

    Returns:
        Long-form rows with raw counts and normalized proportions.
    """
    all_bins: set[str] = set()
    for by_cluster in grouped_rows.values():
        for cluster_rows in by_cluster.values():
            for row in cluster_rows:
                initial_raw = row.round_obj.target_initial_belief
                if isinstance(initial_raw, (int, float)):
                    all_bins.add(_initial_belief_bin_from_value(float(initial_raw)))
                else:
                    all_bins.add("unknown")
    ordered_bins = sorted(all_bins, key=_belief_bin_sort_key)

    rows: list[dict[str, object]] = []
    for corpus, by_cluster in grouped_rows.items():
        counts: dict[tuple[int, str], int] = defaultdict(int)
        cluster_totals: dict[int, int] = {
            cluster_id: 0 for cluster_id in cluster_model.cluster_ids
        }
        init_bin_totals: dict[str, int] = {init_bin: 0 for init_bin in ordered_bins}
        corpus_total = 0

        for cluster_id in cluster_model.cluster_ids:
            for row in by_cluster.get(cluster_id, []):
                initial_raw = row.round_obj.target_initial_belief
                if isinstance(initial_raw, (int, float)):
                    init_bin = _initial_belief_bin_from_value(float(initial_raw))
                else:
                    init_bin = "unknown"
                if init_bin not in init_bin_totals:
                    init_bin_totals[init_bin] = 0
                counts[(cluster_id, init_bin)] += 1
                cluster_totals[cluster_id] = cluster_totals.get(cluster_id, 0) + 1
                init_bin_totals[init_bin] = init_bin_totals.get(init_bin, 0) + 1
                corpus_total += 1

        for cluster_id, cluster_name in cluster_model.cluster_items:
            cluster_total = int(cluster_totals.get(cluster_id, 0))
            for init_bin in ordered_bins:
                count = int(counts.get((cluster_id, init_bin), 0))
                init_bin_total = int(init_bin_totals.get(init_bin, 0))
                rows.append(
                    {
                        "corpus": corpus,
                        "cluster_id": int(cluster_id),
                        "cluster_name": cluster_name,
                        "init_belief_bin": init_bin,
                        "count": count,
                        "cluster_total": cluster_total,
                        "init_bin_total": init_bin_total,
                        "corpus_total": int(corpus_total),
                        "prop_within_corpus": (
                            float(count / corpus_total)
                            if corpus_total > 0
                            else float("nan")
                        ),
                        "prop_within_cluster": (
                            float(count / cluster_total)
                            if cluster_total > 0
                            else float("nan")
                        ),
                        "prop_within_init_bin": (
                            float(count / init_bin_total)
                            if init_bin_total > 0
                            else float("nan")
                        ),
                    }
                )
    return rows


def _plot_cluster_init_bin_heatmaps(
    *,
    path: Path,
    rows: list[dict[str, object]],
    cluster_model: HumanTrajectoryClusterModel,
) -> None:
    """
    Plot corpus-specific heatmaps of P(cluster | initial-belief-bin).

    Args:
        path: Output figure path.
        rows: Rows from ``_cluster_init_bin_rows``.
        cluster_model: Loaded cluster model.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    corpora_present = {str(row.get("corpus", "")) for row in rows}
    corpora = [
        corpus for corpus in CORE_COMPARISON_CORPUS_ORDER if corpus in corpora_present
    ]
    corpora.extend(sorted(corpora_present - set(corpora), key=_corpus_sort_key))
    init_bins = sorted(
        {str(row.get("init_belief_bin", "unknown")) for row in rows},
        key=_belief_bin_sort_key,
    )
    cluster_ids = [int(cluster_id) for cluster_id in cluster_model.cluster_ids]

    n_corpora = len(corpora)
    n_cols = min(2, max(1, n_corpora))
    n_rows = int(np.ceil(float(n_corpora) / float(n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(7.0, 5.8 * n_cols), max(3.6, 3.8 * n_rows)),
        squeeze=False,
    )
    cmap = plt.get_cmap("Blues")
    image_artist = None

    for subplot_idx, corpus in enumerate(corpora):
        axis = axes[subplot_idx // n_cols][subplot_idx % n_cols]
        matrix = np.full((len(cluster_ids), len(init_bins)), np.nan, dtype=float)
        count_matrix = np.zeros((len(cluster_ids), len(init_bins)), dtype=int)
        lookup: dict[tuple[int, str], dict[str, object]] = {}
        for row in rows:
            if str(row.get("corpus", "")) != corpus:
                continue
            lookup[
                (
                    int(row.get("cluster_id", -1)),
                    str(row.get("init_belief_bin", "unknown")),
                )
            ] = row

        for row_idx, cluster_id in enumerate(cluster_ids):
            for col_idx, init_bin in enumerate(init_bins):
                record = lookup.get((cluster_id, init_bin))
                if record is None:
                    continue
                matrix[row_idx, col_idx] = float(
                    record.get("prop_within_init_bin", float("nan"))
                )
                count_matrix[row_idx, col_idx] = int(record.get("count", 0))

        image_artist = axis.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )

        for row_idx, cluster_id in enumerate(cluster_ids):
            for col_idx, init_bin in enumerate(init_bins):
                value = matrix[row_idx, col_idx]
                count = count_matrix[row_idx, col_idx]
                if not np.isfinite(value) or count <= 0:
                    continue
                text_color = "white" if value >= 0.55 else "black"
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}\n(n={count})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

        cluster_labels = [f"C{cluster_id}" for cluster_id in cluster_ids]
        axis.set_xticks(np.arange(len(init_bins), dtype=float))
        axis.set_xticklabels(init_bins, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(cluster_ids), dtype=float))
        axis.set_yticklabels(cluster_labels)
        axis.set_xlabel("Initial belief bin")
        axis.set_ylabel("Trajectory cluster")
        axis.set_title(_corpus_display_label(corpus))

    total_axes = n_rows * n_cols
    for subplot_idx in range(n_corpora, total_axes):
        axes[subplot_idx // n_cols][subplot_idx % n_cols].axis("off")

    if image_artist is not None:
        fig.colorbar(
            image_artist,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
            label="P(cluster | init_bin, corpus)",
        )
    fig.suptitle("Cluster Composition Conditioned On Initial Belief Bin", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """
    Run simulator-human likeness analysis and print/save summary tables.
    """
    args = parse_args()
    try:
        selected_fan_corpora = _parse_fan_corpora_arg(args.fan_corpora)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    selected_fan_policy_models = _parse_fan_policy_models_arg(args.fan_policy_models)
    fan_max_turns = int(args.fan_max_turns) if args.fan_max_turns is not None else None
    if fan_max_turns is not None and fan_max_turns < 1:
        raise SystemExit("--fan-max-turns must be >= 1 when provided.")
    min_date = parse_min_date(args.min_date)
    all_rows = load_serial_trajectories(args.results_dir, min_date=min_date)
    selector_kwargs = selector_kwargs_from_args(args)
    human_rows, structure_rows, full_rows, vanilla_rows = (
        select_matched_human_structure_full_vanilla_from_args(
            all_rows, args=args, include_vanilla=True
        )
    )
    full_no_rhet_rows = select_simulator(
        all_rows,
        simulator_type="full_no_rhetoric",
        **selector_kwargs,
    )
    if args.proposition_match != "none":
        human_props = {row.proposition for row in human_rows}
        full_no_rhet_rows = [
            row for row in full_no_rhet_rows if row.proposition in human_props
        ]
    full_no_rhetoric_corpora = _full_no_rhetoric_variant_corpora(full_no_rhet_rows)
    include_vanilla = bool(vanilla_rows)
    if not bool(args.include_vanilla_llm_target):
        print(
            "\nNote: --include-vanilla-llm-target is ignored; "
            "vanilla is included whenever rows are available."
        )

    corpus_rows = [_corpus_summary_row("human_reference", human_rows)]
    if include_vanilla:
        corpus_rows.append(_corpus_summary_row("vanilla_llm_target", vanilla_rows))
    corpus_rows.extend(
        [
            _corpus_summary_row("structure_target", structure_rows),
            _corpus_summary_row("full_simulated_target", full_rows),
        ]
    )
    if full_no_rhetoric_corpora:
        for corpus, corpus_rows_data in full_no_rhetoric_corpora:
            corpus_rows.append(_corpus_summary_row(corpus, corpus_rows_data))
    else:
        corpus_rows.append(_corpus_summary_row("full_no_rhetoric_target", []))
    print_table(
        corpus_rows,
        title="Corpus Summary",
        columns=[
            "corpus",
            "rounds",
            "updates",
            "unique_props",
            "mean_updates_per_round",
        ],
        aligns={"rounds": "right", "updates": "right", "unique_props": "right"},
        formatters={"mean_updates_per_round": lambda value: f"{value:.3f}"},
    )

    corpora_for_available_mode: list[tuple[str, list[RoundTrajectory]]] = [
        ("human_reference", human_rows),
    ]
    if include_vanilla:
        corpora_for_available_mode.append(("vanilla_llm_target", vanilla_rows))
    corpora_for_available_mode.extend(
        [
            ("structure_target", structure_rows),
            ("full_simulated_target", full_rows),
        ]
    )
    if full_no_rhetoric_corpora:
        corpora_for_available_mode.extend(full_no_rhetoric_corpora)
    else:
        corpora_for_available_mode.append(("full_no_rhetoric_target", []))
    corpora_for_available_mode = [
        (corpus, rows) for corpus, rows in corpora_for_available_mode if rows
    ]
    simulator_corpora_for_available_mode = [
        (corpus, rows)
        for corpus, rows in corpora_for_available_mode
        if corpus != "human_reference"
    ]
    simulator_corpora_for_available_mode_split = (
        _corpora_with_instruction_variants_for_fan(simulator_corpora_for_available_mode)
    )
    simulator_corpora_for_available_mode_split = (
        _corpora_with_simulator_model_variants_for_fan(
            simulator_corpora_for_available_mode_split
        )
    )
    simulator_corpora_for_available_mode_split = _corpora_with_policy_variants_for_fan(
        simulator_corpora_for_available_mode_split
    )

    prefix: Path = args.output_prefix
    _write_csv(
        prefix.with_name(prefix.name + "_corpus_summary.csv"),
        corpus_rows,
        ["corpus", "rounds", "updates", "unique_props", "mean_updates_per_round"],
    )

    if not human_rows or not structure_rows or not full_rows:
        available_labels = ", ".join(corpus for corpus, _ in corpora_for_available_mode)
        print(
            "\nReduced mode: running with available corpora only. "
            "Primary structure-vs-full metrics are skipped.\n"
            f"Available corpora: {available_labels}"
        )
        if not corpora_for_available_mode:
            return

        movement_epsilon = max(0.0, float(args.movement_epsilon))
        movement_summary_rows = [
            _movement_summary_row(corpus=corpus, rows=rows, epsilon=movement_epsilon)
            for corpus, rows in corpora_for_available_mode
        ]
        print_table(
            movement_summary_rows,
            title=(
                "\nMovement Diagnostics "
                f"(Epsilon={movement_epsilon:.3f}; Positive Is Toward Persuader)"
            ),
            columns=[
                "corpus",
                "rounds",
                "updates",
                "mean_total_delta",
                "median_total_delta",
                "mean_abs_total_delta",
                "toward_round_rate",
                "away_round_rate",
                "near_zero_round_rate",
                "any_up_step_rate",
                "any_down_step_rate",
                "both_directions_rate",
                "up_update_rate",
                "down_update_rate",
                "near_zero_update_rate",
            ],
            aligns={
                "rounds": "right",
                "updates": "right",
                "mean_total_delta": "right",
                "median_total_delta": "right",
                "mean_abs_total_delta": "right",
                "toward_round_rate": "right",
                "away_round_rate": "right",
                "near_zero_round_rate": "right",
                "any_up_step_rate": "right",
                "any_down_step_rate": "right",
                "both_directions_rate": "right",
                "up_update_rate": "right",
                "down_update_rate": "right",
                "near_zero_update_rate": "right",
            },
            formatters={
                "mean_total_delta": lambda value: f"{value:+.4f}",
                "median_total_delta": lambda value: f"{value:+.4f}",
                "mean_abs_total_delta": lambda value: f"{value:.4f}",
                "toward_round_rate": lambda value: f"{value:.3f}",
                "away_round_rate": lambda value: f"{value:.3f}",
                "near_zero_round_rate": lambda value: f"{value:.3f}",
                "any_up_step_rate": lambda value: f"{value:.3f}",
                "any_down_step_rate": lambda value: f"{value:.3f}",
                "both_directions_rate": lambda value: f"{value:.3f}",
                "up_update_rate": lambda value: f"{value:.3f}",
                "down_update_rate": lambda value: f"{value:.3f}",
                "near_zero_update_rate": lambda value: f"{value:.3f}",
            },
        )
        _write_csv(
            prefix.with_name(prefix.name + "_movement_summary.csv"),
            movement_summary_rows,
            [
                "corpus",
                "rounds",
                "updates",
                "mean_total_delta",
                "median_total_delta",
                "mean_abs_total_delta",
                "toward_round_rate",
                "away_round_rate",
                "near_zero_round_rate",
                "any_up_step_rate",
                "any_down_step_rate",
                "both_directions_rate",
                "up_update_rate",
                "down_update_rate",
                "near_zero_update_rate",
            ],
        )

        round_dynamics_rows: list[dict[str, object]] = []
        for corpus, rows in corpora_for_available_mode:
            for trajectory_index, row in enumerate(rows):
                round_dynamics_rows.append(
                    _round_dynamics_row(
                        corpus=corpus,
                        trajectory_index=trajectory_index,
                        row=row,
                        epsilon=movement_epsilon,
                    )
                )
        _write_csv(
            prefix.with_name(prefix.name + "_round_dynamics.csv"),
            round_dynamics_rows,
            ROUND_DYNAMICS_COLUMNS,
        )
        proposition_stance_rows = _proposition_stance_delta_rows(
            round_dynamics_rows=round_dynamics_rows,
            epsilon=movement_epsilon,
        )
        _write_csv(
            prefix.with_name(prefix.name + "_proposition_stance_deltas.csv"),
            proposition_stance_rows,
            PROPOSITION_STANCE_DELTA_COLUMNS,
        )
        proposition_stance_gap_rows = _proposition_stance_gap_vs_baseline_rows(
            proposition_rows=proposition_stance_rows,
            baseline_corpus="vanilla_llm_target",
        )
        _write_csv(
            prefix.with_name(prefix.name + "_proposition_stance_gaps_vs_vanilla.csv"),
            proposition_stance_gap_rows,
            PROPOSITION_STANCE_GAP_COLUMNS,
        )

        pairwise_rows: list[dict[str, object]] = []
        for (left_name, left_rows), (right_name, right_rows) in itertools.combinations(
            corpora_for_available_mode, 2
        ):
            pairwise_rows.append(
                {
                    "left_corpus": left_name,
                    "right_corpus": right_name,
                    "pooled_wasserstein": _pooled_w1(left_rows, right_rows),
                }
            )
        if pairwise_rows:
            print_table(
                pairwise_rows,
                title="\nPairwise Pooled Wasserstein (Lower Is More Similar)",
                columns=["left_corpus", "right_corpus", "pooled_wasserstein"],
                aligns={"pooled_wasserstein": "right"},
                formatters={"pooled_wasserstein": lambda value: f"{value:.4f}"},
            )
            _write_csv(
                prefix.with_name(prefix.name + "_pairwise_pooled_w1.csv"),
                pairwise_rows,
                ["left_corpus", "right_corpus", "pooled_wasserstein"],
            )

        _run_atomizer_alignment_outputs(
            prefix=prefix,
            corpora=simulator_corpora_for_available_mode_split,
        )

        individual_rows: list[dict[str, object]] = []
        if args.plot_individual_trajectories:
            individual_root = prefix.with_name(
                prefix.name + "_individual_trajectory_spread"
            )
            for corpus, rows in corpora_for_available_mode:
                individual_rows.extend(
                    _plot_individual_belief_trajectories(
                        output_dir=individual_root,
                        corpus=corpus,
                        rows=rows,
                        max_per_corpus=int(args.individual_trajectory_max_per_corpus),
                        epsilon=movement_epsilon,
                    )
                )
        _write_csv(
            prefix.with_name(prefix.name + "_individual_trajectory_spread.csv"),
            individual_rows,
            [
                "corpus",
                "rounds_available",
                "rounds_plotted",
                "toward_rounds",
                "away_rounds",
                "near_zero_rounds",
                "figure_path",
            ],
        )

        characteristic_trace_rows = _write_characteristic_round_traces(
            prefix=prefix,
            corpora=simulator_corpora_for_available_mode_split,
            per_corpus=max(0, int(args.characteristic_rounds_per_corpus)),
            epsilon=movement_epsilon,
        )
        _write_csv(
            prefix.with_name(prefix.name + "_characteristic_round_traces.csv"),
            characteristic_trace_rows,
            CHARACTERISTIC_TRACE_COLUMNS,
        )

        fan_source_corpora_for_available_mode = _filter_corpora_for_fan(
            corpora_for_available_mode,
            selected_fan_corpora,
        )
        if args.fan_split_persona:
            fan_base_corpora_for_available_mode = (
                _corpora_with_persona_variants_for_fan(
                    fan_source_corpora_for_available_mode
                )
            )
        else:
            fan_base_corpora_for_available_mode = fan_source_corpora_for_available_mode
        fan_base_corpora_for_available_mode = (
            _corpora_with_instruction_variants_for_fan(
                fan_base_corpora_for_available_mode
            )
        )
        fan_base_corpora_for_available_mode = (
            _corpora_with_simulator_model_variants_for_fan(
                fan_base_corpora_for_available_mode
            )
        )
        fan_corpora_for_available_mode = _corpora_with_policy_variants_for_fan(
            fan_base_corpora_for_available_mode
        )
        fan_corpora_for_available_mode = _filter_corpora_by_fan_policy_model(
            fan_corpora_for_available_mode,
            selected_fan_policy_models,
        )
        fan_corpora_for_available_mode = [
            (corpus, _truncate_rows_for_fan(rows, fan_max_turns))
            for corpus, rows in fan_corpora_for_available_mode
        ]
        fan_corpora_for_available_mode = [
            (corpus, rows) for corpus, rows in fan_corpora_for_available_mode if rows
        ]
        if not fan_corpora_for_available_mode:
            return
        fan_corpus_counts_available = {
            corpus: len(rows) for corpus, rows in fan_corpora_for_available_mode
        }
        cell_total_delta_rows = _cell_total_delta_rows(
            corpora=fan_corpora_for_available_mode,
            epsilon=movement_epsilon,
            bootstrap_n=max(100, int(args.bootstrap)),
            rng=np.random.default_rng(int(args.seed) + 7001),
        )
        _write_csv(
            prefix.with_name(prefix.name + "_cell_total_delta_by_init_bin.csv"),
            cell_total_delta_rows,
            CELL_TOTAL_DELTA_COLUMNS,
        )
        _plot_cell_total_delta_bars(
            path=prefix.with_name(prefix.name + "_cell_total_delta_by_init_bin.png"),
            rows=cell_total_delta_rows,
        )
        trajectory_fan_rows: list[dict[str, object]] = []
        for corpus, rows in fan_corpora_for_available_mode:
            trajectory_fan_rows.extend(_trajectory_fan_rows(corpus=corpus, rows=rows))
        _write_csv(
            prefix.with_name(prefix.name + "_trajectory_fan.csv"),
            trajectory_fan_rows,
            TRAJECTORY_FAN_COLUMNS,
        )
        _plot_trajectory_fan(
            path=prefix.with_name(prefix.name + "_trajectory_fan.png"),
            fan_rows=trajectory_fan_rows,
            min_n_per_turn=max(1, int(args.plot_min_n_per_turn)),
            corpus_counts=fan_corpus_counts_available,
            show_mean_error_bars=bool(args.fan_show_mean_error_bars),
        )

        trajectory_fan_norm_rows: list[dict[str, object]] = []
        grid_points = max(2, int(args.normalized_grid_points))
        for corpus, rows in fan_corpora_for_available_mode:
            trajectory_fan_norm_rows.extend(
                _trajectory_fan_rows_normalized(
                    corpus=corpus,
                    rows=rows,
                    grid_points=grid_points,
                )
            )
        _write_csv(
            prefix.with_name(prefix.name + "_trajectory_fan_normalized.csv"),
            trajectory_fan_norm_rows,
            TRAJECTORY_FAN_NORMALIZED_COLUMNS,
        )
        _plot_trajectory_fan_normalized(
            path=prefix.with_name(prefix.name + "_trajectory_fan_normalized.png"),
            fan_rows=trajectory_fan_norm_rows,
            corpus_counts=fan_corpus_counts_available,
            show_mean_error_bars=bool(args.fan_show_mean_error_bars),
        )
        return

    corpora_for_analysis: list[tuple[str, list[RoundTrajectory]]] = [
        ("human_reference", human_rows),
    ]
    if include_vanilla:
        corpora_for_analysis.append(("vanilla_llm_target", vanilla_rows))
    corpora_for_analysis.extend(
        [
            ("structure_target", structure_rows),
            ("full_simulated_target", full_rows),
            ("full_no_rhetoric_target", full_no_rhet_rows),
        ]
    )
    simulator_corpora_for_examples: list[tuple[str, list[RoundTrajectory]]] = []
    if include_vanilla:
        simulator_corpora_for_examples.append(("vanilla_llm_target", vanilla_rows))
    simulator_corpora_for_examples.extend(
        [
            ("structure_target", structure_rows),
            ("full_simulated_target", full_rows),
            ("full_no_rhetoric_target", full_no_rhet_rows),
        ]
    )
    simulator_corpora_for_examples_split = _corpora_with_instruction_variants_for_fan(
        simulator_corpora_for_examples
    )
    simulator_corpora_for_examples_split = (
        _corpora_with_simulator_model_variants_for_fan(
            simulator_corpora_for_examples_split
        )
    )
    simulator_corpora_for_examples_split = _corpora_with_policy_variants_for_fan(
        simulator_corpora_for_examples_split
    )

    cluster_model: HumanTrajectoryClusterModel | None = None
    if args.enable_cluster_addendum:
        try:
            cluster_model = load_human_trajectory_cluster_model(args.cluster_model_path)
        except (OSError, ValueError) as error:
            print(
                "\nCluster addendum could not be enabled because cluster model "
                "failed to load: "
                f"{error}"
            )

    human_all_updates = _flatten_updates(human_rows)
    structure_all_updates = _flatten_updates(structure_rows)
    full_all_updates = _flatten_updates(full_rows)
    if include_vanilla:
        vanilla_all_updates = _flatten_updates(vanilla_rows)
    else:
        vanilla_all_updates = np.asarray([], dtype=float)

    pooled_jsd_edges = _histogram_edges_for_arrays(
        arrays=[
            human_all_updates,
            structure_all_updates,
            full_all_updates,
            vanilla_all_updates,
        ],
        n_bins=max(5, int(args.jsd_bins)),
    )
    structure_pooled_jsd = _pooled_jsd_from_arrays(
        reference_values=human_all_updates,
        candidate_values=structure_all_updates,
        edges=pooled_jsd_edges,
    )
    full_pooled_jsd = _pooled_jsd_from_arrays(
        reference_values=human_all_updates,
        candidate_values=full_all_updates,
        edges=pooled_jsd_edges,
    )
    vanilla_pooled_jsd = _pooled_jsd_from_arrays(
        reference_values=human_all_updates,
        candidate_values=vanilla_all_updates,
        edges=pooled_jsd_edges,
    )

    jsd_min_n = max(1, int(args.metric_min_n_per_turn))
    jsd_turns = _common_turns_with_min_n(
        human_rows=human_rows,
        structure_rows=structure_rows,
        full_rows=full_rows,
        vanilla_rows=vanilla_rows,
        include_vanilla=include_vanilla,
        min_n=jsd_min_n,
    )
    if not jsd_turns:
        jsd_turns = (
            set(_updates_by_turn(human_rows))
            & set(_updates_by_turn(structure_rows))
            & set(_updates_by_turn(full_rows))
        )
        if include_vanilla:
            jsd_turns = jsd_turns & set(_updates_by_turn(vanilla_rows))

    structure_jsd = _turn_index_jsd(
        human_rows=human_rows,
        sim_rows=structure_rows,
        turns=jsd_turns,
        edges=pooled_jsd_edges,
    )
    full_jsd = _turn_index_jsd(
        human_rows=human_rows,
        sim_rows=full_rows,
        turns=jsd_turns,
        edges=pooled_jsd_edges,
    )
    if include_vanilla:
        vanilla_jsd = _turn_index_jsd(
            human_rows=human_rows,
            sim_rows=vanilla_rows,
            turns=jsd_turns,
            edges=pooled_jsd_edges,
        )
    else:
        vanilla_jsd = {}

    pooled_structure = _pooled_w1(human_rows, structure_rows)
    pooled_full = _pooled_w1(human_rows, full_rows)
    pooled_vanilla = (
        _pooled_w1(human_rows, vanilla_rows) if include_vanilla else float("nan")
    )
    trajectory_shape_structure = _normalized_trajectory_shape_w1(
        human_rows,
        structure_rows,
        grid_points=int(args.normalized_grid_points),
    )
    trajectory_shape_full = _normalized_trajectory_shape_w1(
        human_rows,
        full_rows,
        grid_points=int(args.normalized_grid_points),
    )
    trajectory_shape_vanilla = (
        _normalized_trajectory_shape_w1(
            human_rows,
            vanilla_rows,
            grid_points=int(args.normalized_grid_points),
        )
        if include_vanilla
        else float("nan")
    )
    prop_structure, prop_structure_cov = _prop_weighted_w1(human_rows, structure_rows)
    prop_full, prop_full_cov = _prop_weighted_w1(human_rows, full_rows)
    if include_vanilla:
        prop_vanilla, _ = _prop_weighted_w1(human_rows, vanilla_rows)
    else:
        prop_vanilla = float("nan")
    length_matched_structure = _length_matched_pooled_w1(
        human_rows,
        structure_rows,
        n_draws=args.length_match_draws,
        seed=args.seed + 101,
    )
    length_matched_full = _length_matched_pooled_w1(
        human_rows,
        full_rows,
        n_draws=args.length_match_draws,
        seed=args.seed + 202,
    )
    lm_structure_value = float(length_matched_structure.get("mean", float("nan")))
    lm_full_value = float(length_matched_full.get("mean", float("nan")))
    if include_vanilla:
        length_matched_vanilla = _length_matched_pooled_w1(
            human_rows,
            vanilla_rows,
            n_draws=args.length_match_draws,
            seed=args.seed + 303,
        )
        lm_vanilla_value = float(length_matched_vanilla.get("mean", float("nan")))
    else:
        length_matched_vanilla = {}
        lm_vanilla_value = float("nan")
    lm_diff_distribution = np.asarray([], dtype=float)
    if length_matched_structure and length_matched_full:
        lm_diff_distribution = np.asarray(
            length_matched_structure["distribution"], dtype=float
        ) - np.asarray(length_matched_full["distribution"], dtype=float)
    lm_diff_ci = (
        (
            float(np.quantile(lm_diff_distribution, 0.025)),
            float(np.quantile(lm_diff_distribution, 0.975)),
        )
        if lm_diff_distribution.size > 0
        else (float("nan"), float("nan"))
    )
    lm_p_structure_closer = (
        float(np.mean(lm_diff_distribution < 0.0))
        if lm_diff_distribution.size > 0
        else float("nan")
    )

    bootstrap = _bootstrap_primary(
        human_rows,
        structure_rows,
        full_rows,
        n_boot=args.bootstrap,
        seed=args.seed,
        bootstrap_statistic_ci_fn=bootstrap_statistic_ci,
    )

    primary_rows = [
        {
            "metric": ("pooled_jsd " f"(all_turns, bins={max(5, int(args.jsd_bins))})"),
            "structure": structure_pooled_jsd,
            "full": full_pooled_jsd,
            "structure_minus_full": structure_pooled_jsd - full_pooled_jsd,
            "winner": "structure" if structure_pooled_jsd < full_pooled_jsd else "full",
            "structure_ci": "",
            "full_ci": "",
            "diff_ci": "",
            "p_structure_closer": float("nan"),
        },
        {
            "metric": (
                "normalized_trajectory_shape_wasserstein "
                f"(grid={int(args.normalized_grid_points)})"
            ),
            "structure": trajectory_shape_structure,
            "full": trajectory_shape_full,
            "structure_minus_full": trajectory_shape_structure - trajectory_shape_full,
            "winner": (
                "structure"
                if trajectory_shape_structure < trajectory_shape_full
                else "full"
            ),
            "structure_ci": "",
            "full_ci": "",
            "diff_ci": "",
            "p_structure_closer": float("nan"),
        },
        {
            "metric": "pooled_wasserstein",
            "structure": pooled_structure,
            "full": pooled_full,
            "structure_minus_full": pooled_structure - pooled_full,
            "winner": "structure" if pooled_structure < pooled_full else "full",
            "structure_ci": (
                f"[{bootstrap['structure_ci_lo']:.4f}, {bootstrap['structure_ci_hi']:.4f}]"
                if bootstrap
                else ""
            ),
            "full_ci": (
                f"[{bootstrap['full_ci_lo']:.4f}, {bootstrap['full_ci_hi']:.4f}]"
                if bootstrap
                else ""
            ),
            "diff_ci": (
                f"[{bootstrap['diff_ci_lo']:.4f}, {bootstrap['diff_ci_hi']:.4f}]"
                if bootstrap
                else ""
            ),
            "p_structure_closer": bootstrap.get("p_structure_closer", float("nan")),
        },
        {
            "metric": "prop_weighted_wasserstein",
            "structure": prop_structure,
            "full": prop_full,
            "structure_minus_full": prop_structure - prop_full,
            "winner": "structure" if prop_structure < prop_full else "full",
            "structure_ci": "",
            "full_ci": "",
            "diff_ci": "",
            "p_structure_closer": float("nan"),
            "structure_cov": prop_structure_cov,
            "full_cov": prop_full_cov,
        },
        {
            "metric": "length_matched_pooled_wasserstein",
            "structure": lm_structure_value,
            "full": lm_full_value,
            "structure_minus_full": lm_structure_value - lm_full_value,
            "winner": (
                "structure"
                if (
                    np.isfinite(lm_structure_value)
                    and np.isfinite(lm_full_value)
                    and lm_structure_value < lm_full_value
                )
                else (
                    "full"
                    if np.isfinite(lm_structure_value) and np.isfinite(lm_full_value)
                    else ""
                )
            ),
            "structure_ci": (
                f"[{length_matched_structure['ci_lo']:.4f}, "
                f"{length_matched_structure['ci_hi']:.4f}]"
                if length_matched_structure
                else ""
            ),
            "full_ci": (
                f"[{length_matched_full['ci_lo']:.4f}, "
                f"{length_matched_full['ci_hi']:.4f}]"
                if length_matched_full
                else ""
            ),
            "diff_ci": (
                f"[{lm_diff_ci[0]:.4f}, {lm_diff_ci[1]:.4f}]"
                if np.isfinite(lm_diff_ci[0]) and np.isfinite(lm_diff_ci[1])
                else ""
            ),
            "p_structure_closer": lm_p_structure_closer,
            "structure_cov": float("nan"),
            "full_cov": float("nan"),
        },
    ]
    if args.metric_min_n_per_turn > 0:
        stable_turns = _common_turns_with_min_n(
            human_rows=human_rows,
            structure_rows=structure_rows,
            full_rows=full_rows,
            vanilla_rows=[],
            include_vanilla=False,
            min_n=int(args.metric_min_n_per_turn),
        )
        if stable_turns:
            human_stable = _flatten_updates_for_turns(
                human_rows,
                allowed_turns=stable_turns,
            )
            structure_stable = _flatten_updates_for_turns(
                structure_rows,
                allowed_turns=stable_turns,
            )
            full_stable = _flatten_updates_for_turns(
                full_rows,
                allowed_turns=stable_turns,
            )
            if (
                human_stable.size > 0
                and structure_stable.size > 0
                and full_stable.size > 0
            ):
                stable_jsd_edges = _histogram_edges_for_arrays(
                    arrays=[human_stable, structure_stable, full_stable],
                    n_bins=max(5, int(args.jsd_bins)),
                )
                structure_stable_jsd = _pooled_jsd_from_arrays(
                    reference_values=human_stable,
                    candidate_values=structure_stable,
                    edges=stable_jsd_edges,
                )
                full_stable_jsd = _pooled_jsd_from_arrays(
                    reference_values=human_stable,
                    candidate_values=full_stable,
                    edges=stable_jsd_edges,
                )
                primary_rows.append(
                    {
                        "metric": (
                            "stable_turn_pooled_jsd"
                            f" (n>={int(args.metric_min_n_per_turn)}, "
                            f"turns={min(stable_turns)}-{max(stable_turns)})"
                        ),
                        "structure": structure_stable_jsd,
                        "full": full_stable_jsd,
                        "structure_minus_full": structure_stable_jsd - full_stable_jsd,
                        "winner": (
                            "structure"
                            if structure_stable_jsd < full_stable_jsd
                            else "full"
                        ),
                        "structure_ci": "",
                        "full_ci": "",
                        "diff_ci": "",
                        "p_structure_closer": float("nan"),
                        "structure_cov": float("nan"),
                        "full_cov": float("nan"),
                    }
                )
                structure_stable_w1 = float(
                    wasserstein_distance(human_stable, structure_stable)
                )
                full_stable_w1 = float(wasserstein_distance(human_stable, full_stable))
                primary_rows.append(
                    {
                        "metric": (
                            "stable_turn_pooled_wasserstein"
                            f" (n>={int(args.metric_min_n_per_turn)}, "
                            f"turns={min(stable_turns)}-{max(stable_turns)})"
                        ),
                        "structure": structure_stable_w1,
                        "full": full_stable_w1,
                        "structure_minus_full": structure_stable_w1 - full_stable_w1,
                        "winner": (
                            "structure"
                            if structure_stable_w1 < full_stable_w1
                            else "full"
                        ),
                        "structure_ci": "",
                        "full_ci": "",
                        "diff_ci": "",
                        "p_structure_closer": float("nan"),
                        "structure_cov": float("nan"),
                        "full_cov": float("nan"),
                    }
                )

    print_table(
        primary_rows,
        title="\nPrimary Distances (Lower Is Better)",
        columns=[
            "metric",
            "structure",
            "full",
            "structure_minus_full",
            "winner",
            "structure_ci",
            "full_ci",
            "diff_ci",
            "p_structure_closer",
        ],
        aligns={
            "structure": "right",
            "full": "right",
            "structure_minus_full": "right",
            "p_structure_closer": "right",
        },
        formatters={
            "structure": lambda value: f"{value:.4f}",
            "full": lambda value: f"{value:.4f}",
            "structure_minus_full": lambda value: f"{value:+.4f}",
            "p_structure_closer": (
                lambda value: f"{value:.3f}" if np.isfinite(value) else ""
            ),
        },
    )

    if include_vanilla:
        vanilla_primary_rows = [
            {
                "metric": (
                    "pooled_jsd " f"(all_turns, bins={max(5, int(args.jsd_bins))})"
                ),
                "structure": structure_pooled_jsd,
                "full": full_pooled_jsd,
                "vanilla": vanilla_pooled_jsd,
                "winner": min(
                    (
                        ("structure", structure_pooled_jsd),
                        ("full", full_pooled_jsd),
                        ("vanilla", vanilla_pooled_jsd),
                    ),
                    key=lambda item: item[1],
                )[0],
            },
            {
                "metric": (
                    "normalized_trajectory_shape_wasserstein "
                    f"(grid={int(args.normalized_grid_points)})"
                ),
                "structure": trajectory_shape_structure,
                "full": trajectory_shape_full,
                "vanilla": trajectory_shape_vanilla,
                "winner": min(
                    (
                        ("structure", trajectory_shape_structure),
                        ("full", trajectory_shape_full),
                        ("vanilla", trajectory_shape_vanilla),
                    ),
                    key=lambda item: item[1],
                )[0],
            },
            {
                "metric": "pooled_wasserstein",
                "structure": pooled_structure,
                "full": pooled_full,
                "vanilla": pooled_vanilla,
                "winner": min(
                    (
                        ("structure", pooled_structure),
                        ("full", pooled_full),
                        ("vanilla", pooled_vanilla),
                    ),
                    key=lambda item: item[1],
                )[0],
            },
            {
                "metric": "prop_weighted_wasserstein",
                "structure": prop_structure,
                "full": prop_full,
                "vanilla": prop_vanilla,
                "winner": min(
                    (
                        ("structure", prop_structure),
                        ("full", prop_full),
                        ("vanilla", prop_vanilla),
                    ),
                    key=lambda item: item[1],
                )[0],
            },
            {
                "metric": "length_matched_pooled_wasserstein",
                "structure": lm_structure_value,
                "full": lm_full_value,
                "vanilla": lm_vanilla_value,
                "winner": min(
                    (
                        ("structure", lm_structure_value),
                        ("full", lm_full_value),
                        ("vanilla", lm_vanilla_value),
                    ),
                    key=lambda item: item[1],
                )[0],
            },
        ]
        print_table(
            vanilla_primary_rows,
            title="\nPrimary Distances To Human (With Vanilla Baseline)",
            columns=["metric", "structure", "full", "vanilla", "winner"],
            aligns={
                "structure": "right",
                "full": "right",
                "vanilla": "right",
            },
            formatters={
                "structure": lambda value: f"{value:.4f}",
                "full": lambda value: f"{value:.4f}",
                "vanilla": lambda value: f"{value:.4f}",
            },
        )
    else:
        vanilla_primary_rows: list[dict[str, object]] = []

    human_secondary = _corpus_secondary_stats(human_rows)
    structure_secondary = _corpus_secondary_stats(structure_rows)
    full_secondary = _corpus_secondary_stats(full_rows)
    vanilla_secondary = _corpus_secondary_stats(vanilla_rows) if include_vanilla else {}

    metric_names = [
        "mean_delta",
        "std_delta",
        "toward_persuader_rate",
        "first_mean",
        "rest_mean",
        "first_minus_rest",
    ]
    secondary_rows: list[dict[str, object]] = []
    for metric in metric_names:
        human_value = human_secondary[metric]
        structure_value = structure_secondary[metric]
        full_value = full_secondary[metric]
        structure_error = abs(structure_value - human_value)
        full_error = abs(full_value - human_value)
        secondary_rows.append(
            {
                "metric": metric,
                "human": human_value,
                "structure": structure_value,
                "full": full_value,
                "abs_err_structure": structure_error,
                "abs_err_full": full_error,
                "winner": "structure" if structure_error < full_error else "full",
            }
        )

    print_table(
        secondary_rows,
        title="\nSecondary Shape Metrics (Closer To Human Is Better)",
        columns=[
            "metric",
            "human",
            "structure",
            "full",
            "abs_err_structure",
            "abs_err_full",
            "winner",
        ],
        aligns={
            "human": "right",
            "structure": "right",
            "full": "right",
            "abs_err_structure": "right",
            "abs_err_full": "right",
        },
        formatters={
            "human": lambda value: f"{value:.4f}",
            "structure": lambda value: f"{value:.4f}",
            "full": lambda value: f"{value:.4f}",
            "abs_err_structure": lambda value: f"{value:.4f}",
            "abs_err_full": lambda value: f"{value:.4f}",
        },
    )

    if include_vanilla:
        secondary_with_vanilla_rows: list[dict[str, object]] = []
        for metric in metric_names:
            human_value = human_secondary[metric]
            structure_value = structure_secondary[metric]
            full_value = full_secondary[metric]
            vanilla_value = float(vanilla_secondary[metric])
            structure_error = abs(structure_value - human_value)
            full_error = abs(full_value - human_value)
            vanilla_error = abs(vanilla_value - human_value)
            secondary_with_vanilla_rows.append(
                {
                    "metric": metric,
                    "human": human_value,
                    "structure": structure_value,
                    "full": full_value,
                    "vanilla": vanilla_value,
                    "abs_err_structure": structure_error,
                    "abs_err_full": full_error,
                    "abs_err_vanilla": vanilla_error,
                    "winner": min(
                        (
                            ("structure", structure_error),
                            ("full", full_error),
                            ("vanilla", vanilla_error),
                        ),
                        key=lambda item: item[1],
                    )[0],
                }
            )
        print_table(
            secondary_with_vanilla_rows,
            title="\nSecondary Shape Metrics (With Vanilla Baseline)",
            columns=[
                "metric",
                "human",
                "structure",
                "full",
                "vanilla",
                "abs_err_structure",
                "abs_err_full",
                "abs_err_vanilla",
                "winner",
            ],
            aligns={
                "human": "right",
                "structure": "right",
                "full": "right",
                "vanilla": "right",
                "abs_err_structure": "right",
                "abs_err_full": "right",
                "abs_err_vanilla": "right",
            },
            formatters={
                "human": lambda value: f"{value:.4f}",
                "structure": lambda value: f"{value:.4f}",
                "full": lambda value: f"{value:.4f}",
                "vanilla": lambda value: f"{value:.4f}",
                "abs_err_structure": lambda value: f"{value:.4f}",
                "abs_err_full": lambda value: f"{value:.4f}",
                "abs_err_vanilla": lambda value: f"{value:.4f}",
            },
        )
    else:
        secondary_with_vanilla_rows = []

    by_turn_human = _updates_by_turn(human_rows)
    by_turn_structure = _updates_by_turn(structure_rows)
    by_turn_full = _updates_by_turn(full_rows)
    by_turn_vanilla = _updates_by_turn(vanilla_rows) if include_vanilla else {}
    max_turn = max(by_turn_human.keys()) if by_turn_human else 0
    per_turn_rows: list[dict[str, object]] = []
    for turn in range(1, max_turn + 1):
        human_turn = by_turn_human.get(turn)
        if human_turn is None or human_turn.size == 0:
            continue
        structure_turn = by_turn_structure.get(turn)
        full_turn = by_turn_full.get(turn)
        if structure_turn is None or full_turn is None:
            continue
        if structure_turn.size == 0 or full_turn.size == 0:
            continue
        structure_w1 = float(wasserstein_distance(human_turn, structure_turn))
        full_w1 = float(wasserstein_distance(human_turn, full_turn))
        vanilla_turn = by_turn_vanilla.get(turn)
        vanilla_w1 = float("nan")
        vanilla_n = 0
        if vanilla_turn is not None and vanilla_turn.size > 0:
            vanilla_w1 = float(wasserstein_distance(human_turn, vanilla_turn))
            vanilla_n = int(vanilla_turn.size)
        per_turn_rows.append(
            {
                "turn": turn,
                "human_n": int(human_turn.size),
                "structure_n": int(structure_turn.size),
                "full_n": int(full_turn.size),
                "vanilla_n": vanilla_n,
                "structure_w1": structure_w1,
                "full_w1": full_w1,
                "vanilla_w1": vanilla_w1,
                "winner": (
                    min(
                        [
                            ("structure", structure_w1),
                            ("full", full_w1),
                            (
                                "vanilla",
                                vanilla_w1,
                            ),
                        ],
                        key=lambda item: item[1],
                    )[0]
                    if np.isfinite(vanilla_w1)
                    else ("structure" if structure_w1 < full_w1 else "full")
                ),
            }
        )

    per_turn_columns = [
        "turn",
        "human_n",
        "structure_n",
        "full_n",
        "structure_w1",
        "full_w1",
        "winner",
    ]
    per_turn_aligns = {
        "turn": "right",
        "human_n": "right",
        "structure_n": "right",
        "full_n": "right",
        "structure_w1": "right",
        "full_w1": "right",
    }
    per_turn_formatters: dict[str, Callable[[object], str]] = {
        "structure_w1": lambda value: f"{value:.4f}",
        "full_w1": lambda value: f"{value:.4f}",
    }
    if include_vanilla:
        per_turn_columns = [
            "turn",
            "human_n",
            "structure_n",
            "full_n",
            "vanilla_n",
            "structure_w1",
            "full_w1",
            "vanilla_w1",
            "winner",
        ]
        per_turn_aligns["vanilla_n"] = "right"
        per_turn_aligns["vanilla_w1"] = "right"
        per_turn_formatters["vanilla_w1"] = lambda value: (
            f"{value:.4f}" if np.isfinite(float(value)) else ""
        )

    print_table(
        per_turn_rows,
        title="\nPer-Turn Wasserstein",
        columns=per_turn_columns,
        aligns=per_turn_aligns,
        formatters=per_turn_formatters,
    )

    movement_epsilon = max(0.0, float(args.movement_epsilon))
    movement_summary_rows = [
        _movement_summary_row(
            corpus=corpus,
            rows=rows,
            epsilon=movement_epsilon,
        )
        for corpus, rows in corpora_for_analysis
    ]
    print_table(
        movement_summary_rows,
        title=(
            "\nMovement Diagnostics "
            f"(Epsilon={movement_epsilon:.3f}; Positive Is Toward Persuader)"
        ),
        columns=[
            "corpus",
            "rounds",
            "updates",
            "mean_total_delta",
            "median_total_delta",
            "mean_abs_total_delta",
            "toward_round_rate",
            "away_round_rate",
            "near_zero_round_rate",
            "any_up_step_rate",
            "any_down_step_rate",
            "both_directions_rate",
            "up_update_rate",
            "down_update_rate",
            "near_zero_update_rate",
        ],
        aligns={
            "rounds": "right",
            "updates": "right",
            "mean_total_delta": "right",
            "median_total_delta": "right",
            "mean_abs_total_delta": "right",
            "toward_round_rate": "right",
            "away_round_rate": "right",
            "near_zero_round_rate": "right",
            "any_up_step_rate": "right",
            "any_down_step_rate": "right",
            "both_directions_rate": "right",
            "up_update_rate": "right",
            "down_update_rate": "right",
            "near_zero_update_rate": "right",
        },
        formatters={
            "mean_total_delta": lambda value: f"{value:+.4f}",
            "median_total_delta": lambda value: f"{value:+.4f}",
            "mean_abs_total_delta": lambda value: f"{value:.4f}",
            "toward_round_rate": lambda value: f"{value:.3f}",
            "away_round_rate": lambda value: f"{value:.3f}",
            "near_zero_round_rate": lambda value: f"{value:.3f}",
            "any_up_step_rate": lambda value: f"{value:.3f}",
            "any_down_step_rate": lambda value: f"{value:.3f}",
            "both_directions_rate": lambda value: f"{value:.3f}",
            "up_update_rate": lambda value: f"{value:.3f}",
            "down_update_rate": lambda value: f"{value:.3f}",
            "near_zero_update_rate": lambda value: f"{value:.3f}",
        },
    )

    round_dynamics_rows: list[dict[str, object]] = []
    for corpus, rows in corpora_for_analysis:
        for trajectory_index, row in enumerate(rows):
            round_dynamics_rows.append(
                _round_dynamics_row(
                    corpus=corpus,
                    trajectory_index=trajectory_index,
                    row=row,
                    epsilon=movement_epsilon,
                )
            )

    per_turn_jsd_rows: list[dict[str, object]] = []
    per_turn_jsd_columns = ["turn", "human_n", "structure_jsd", "full_jsd", "winner"]
    if args.include_turn_jsd_diagnostics:
        structure_jsd_by_turn = {
            int(row["turn"]): float(row["jsd"])
            for row in structure_jsd.get("per_turn", [])
            if isinstance(row, dict)
        }
        full_jsd_by_turn = {
            int(row["turn"]): float(row["jsd"])
            for row in full_jsd.get("per_turn", [])
            if isinstance(row, dict)
        }
        vanilla_jsd_by_turn = (
            {
                int(row["turn"]): float(row["jsd"])
                for row in vanilla_jsd.get("per_turn", [])
                if isinstance(row, dict)
            }
            if include_vanilla
            else {}
        )
        per_turn_jsd_turns = set(structure_jsd_by_turn) & set(full_jsd_by_turn)
        if include_vanilla:
            per_turn_jsd_turns = per_turn_jsd_turns & set(vanilla_jsd_by_turn)
        for turn in sorted(per_turn_jsd_turns):
            human_vals = by_turn_human.get(turn)
            if human_vals is None:
                continue
            row: dict[str, object] = {
                "turn": int(turn),
                "human_n": int(human_vals.size),
                "structure_jsd": float(structure_jsd_by_turn[turn]),
                "full_jsd": float(full_jsd_by_turn[turn]),
                "winner": (
                    "structure"
                    if float(structure_jsd_by_turn[turn])
                    < float(full_jsd_by_turn[turn])
                    else "full"
                ),
            }
            if include_vanilla:
                vanilla_value = float(vanilla_jsd_by_turn[turn])
                row["vanilla_jsd"] = vanilla_value
                row["winner"] = min(
                    (
                        ("structure", float(row["structure_jsd"])),
                        ("full", float(row["full_jsd"])),
                        ("vanilla", vanilla_value),
                    ),
                    key=lambda item: item[1],
                )[0]
            per_turn_jsd_rows.append(row)

        per_turn_jsd_aligns = {
            "turn": "right",
            "human_n": "right",
            "structure_jsd": "right",
            "full_jsd": "right",
        }
        per_turn_jsd_formatters: dict[str, Callable[[object], str]] = {
            "structure_jsd": lambda value: f"{value:.4f}",
            "full_jsd": lambda value: f"{value:.4f}",
        }
        if include_vanilla:
            per_turn_jsd_columns = [
                "turn",
                "human_n",
                "structure_jsd",
                "full_jsd",
                "vanilla_jsd",
                "winner",
            ]
            per_turn_jsd_aligns["vanilla_jsd"] = "right"
            per_turn_jsd_formatters["vanilla_jsd"] = lambda value: f"{value:.4f}"
        print_table(
            per_turn_jsd_rows,
            title="\nPer-Turn JSD (Lower Is Better)",
            columns=per_turn_jsd_columns,
            aligns=per_turn_jsd_aligns,
            formatters=per_turn_jsd_formatters,
        )

    prefix: Path = args.output_prefix
    _write_csv(
        prefix.with_name(prefix.name + "_primary.csv"),
        primary_rows,
        [
            "metric",
            "structure",
            "full",
            "structure_minus_full",
            "winner",
            "structure_ci",
            "full_ci",
            "diff_ci",
            "p_structure_closer",
            "structure_cov",
            "full_cov",
        ],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_secondary.csv"),
        secondary_rows,
        [
            "metric",
            "human",
            "structure",
            "full",
            "abs_err_structure",
            "abs_err_full",
            "winner",
        ],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_per_turn.csv"),
        per_turn_rows,
        per_turn_columns,
    )
    if args.include_turn_jsd_diagnostics:
        _write_csv(
            prefix.with_name(prefix.name + "_per_turn_jsd.csv"),
            per_turn_jsd_rows,
            per_turn_jsd_columns,
        )
    _write_csv(
        prefix.with_name(prefix.name + "_movement_summary.csv"),
        movement_summary_rows,
        [
            "corpus",
            "rounds",
            "updates",
            "mean_total_delta",
            "median_total_delta",
            "mean_abs_total_delta",
            "toward_round_rate",
            "away_round_rate",
            "near_zero_round_rate",
            "any_up_step_rate",
            "any_down_step_rate",
            "both_directions_rate",
            "up_update_rate",
            "down_update_rate",
            "near_zero_update_rate",
        ],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_round_dynamics.csv"),
        round_dynamics_rows,
        ROUND_DYNAMICS_COLUMNS,
    )
    proposition_stance_rows = _proposition_stance_delta_rows(
        round_dynamics_rows=round_dynamics_rows,
        epsilon=movement_epsilon,
    )
    _write_csv(
        prefix.with_name(prefix.name + "_proposition_stance_deltas.csv"),
        proposition_stance_rows,
        PROPOSITION_STANCE_DELTA_COLUMNS,
    )
    proposition_stance_gap_rows = _proposition_stance_gap_vs_baseline_rows(
        proposition_rows=proposition_stance_rows,
        baseline_corpus="vanilla_llm_target",
    )
    _write_csv(
        prefix.with_name(prefix.name + "_proposition_stance_gaps_vs_vanilla.csv"),
        proposition_stance_gap_rows,
        PROPOSITION_STANCE_GAP_COLUMNS,
    )
    _run_atomizer_alignment_outputs(
        prefix=prefix,
        corpora=simulator_corpora_for_examples_split,
    )

    individual_rows: list[dict[str, object]] = []
    if args.plot_individual_trajectories:
        individual_root = prefix.with_name(
            prefix.name + "_individual_trajectory_spread"
        )
        for corpus, rows in corpora_for_analysis:
            individual_rows.extend(
                _plot_individual_belief_trajectories(
                    output_dir=individual_root,
                    corpus=corpus,
                    rows=rows,
                    max_per_corpus=int(args.individual_trajectory_max_per_corpus),
                    epsilon=movement_epsilon,
                )
            )
    _write_csv(
        prefix.with_name(prefix.name + "_individual_trajectory_spread.csv"),
        individual_rows,
        [
            "corpus",
            "rounds_available",
            "rounds_plotted",
            "toward_rounds",
            "away_rounds",
            "near_zero_rounds",
            "figure_path",
        ],
    )

    characteristic_trace_rows = _write_characteristic_round_traces(
        prefix=prefix,
        corpora=simulator_corpora_for_examples_split,
        per_corpus=max(0, int(args.characteristic_rounds_per_corpus)),
        epsilon=movement_epsilon,
    )
    if characteristic_trace_rows:
        print_table(
            characteristic_trace_rows,
            title="\nCharacteristic Simulator Round Traces",
            columns=[
                "corpus",
                "label",
                "trajectory_index",
                "n_turns",
                "total_delta",
                "raw_belief_delta",
                "sign_changes",
                "trace_path",
                "proposition",
            ],
            aligns={
                "trajectory_index": "right",
                "n_turns": "right",
                "total_delta": "right",
                "raw_belief_delta": "right",
                "sign_changes": "right",
            },
            formatters={
                "total_delta": lambda value: f"{value:+.4f}",
                "raw_belief_delta": lambda value: f"{value:+.4f}",
            },
        )
    _write_csv(
        prefix.with_name(prefix.name + "_characteristic_round_traces.csv"),
        characteristic_trace_rows,
        CHARACTERISTIC_TRACE_COLUMNS,
    )

    fan_base_corpora: list[tuple[str, list[RoundTrajectory]]] = [
        ("human_reference", human_rows),
    ]
    if include_vanilla:
        fan_base_corpora.append(("vanilla_llm_target", vanilla_rows))
    fan_base_corpora.extend(
        [
            ("structure_target", structure_rows),
            ("full_simulated_target", full_rows),
        ]
    )
    fan_base_corpora = _filter_corpora_for_fan(
        fan_base_corpora,
        selected_fan_corpora,
    )
    if args.fan_split_persona:
        fan_base_corpora = _corpora_with_persona_variants_for_fan(fan_base_corpora)
    fan_base_corpora = _corpora_with_instruction_variants_for_fan(fan_base_corpora)
    fan_base_corpora = _corpora_with_simulator_model_variants_for_fan(fan_base_corpora)
    fan_corpora = _corpora_with_policy_variants_for_fan(fan_base_corpora)
    fan_corpora = _filter_corpora_by_fan_policy_model(
        fan_corpora,
        selected_fan_policy_models,
    )
    fan_corpora = [
        (corpus, _truncate_rows_for_fan(rows, fan_max_turns))
        for corpus, rows in fan_corpora
    ]
    fan_corpora = [(corpus, rows) for corpus, rows in fan_corpora if rows]
    if not fan_corpora:
        raise SystemExit("No fan corpora remained after --fan-corpora/--fan-max-turns.")
    fan_corpus_counts = {corpus: len(rows) for corpus, rows in fan_corpora}
    cell_total_delta_rows = _cell_total_delta_rows(
        corpora=fan_corpora,
        epsilon=movement_epsilon,
        bootstrap_n=max(100, int(args.bootstrap)),
        rng=np.random.default_rng(int(args.seed) + 7001),
    )
    _write_csv(
        prefix.with_name(prefix.name + "_cell_total_delta_by_init_bin.csv"),
        cell_total_delta_rows,
        CELL_TOTAL_DELTA_COLUMNS,
    )
    _plot_cell_total_delta_bars(
        path=prefix.with_name(prefix.name + "_cell_total_delta_by_init_bin.png"),
        rows=cell_total_delta_rows,
    )
    trajectory_fan_rows: list[dict[str, object]] = []
    for corpus, rows in fan_corpora:
        trajectory_fan_rows.extend(_trajectory_fan_rows(corpus=corpus, rows=rows))
    _write_csv(
        prefix.with_name(prefix.name + "_trajectory_fan.csv"),
        trajectory_fan_rows,
        TRAJECTORY_FAN_COLUMNS,
    )
    _plot_trajectory_fan(
        path=prefix.with_name(prefix.name + "_trajectory_fan.png"),
        fan_rows=trajectory_fan_rows,
        min_n_per_turn=max(1, int(args.plot_min_n_per_turn)),
        corpus_counts=fan_corpus_counts,
        show_mean_error_bars=bool(args.fan_show_mean_error_bars),
    )
    _plot_per_turn_delta_boxes(
        path=prefix.with_name(prefix.name + "_per_turn_delta_box.png"),
        human_rows=human_rows,
        structure_rows=structure_rows,
        full_rows=full_rows,
        vanilla_rows=vanilla_rows,
        include_vanilla=include_vanilla,
        min_n_per_turn=max(1, int(args.plot_min_n_per_turn)),
    )
    trajectory_fan_norm_rows: list[dict[str, object]] = []
    grid_points = max(2, int(args.normalized_grid_points))
    for corpus, rows in fan_corpora:
        trajectory_fan_norm_rows.extend(
            _trajectory_fan_rows_normalized(
                corpus=corpus,
                rows=rows,
                grid_points=grid_points,
            )
        )
    _write_csv(
        prefix.with_name(prefix.name + "_trajectory_fan_normalized.csv"),
        trajectory_fan_norm_rows,
        TRAJECTORY_FAN_NORMALIZED_COLUMNS,
    )
    _plot_trajectory_fan_normalized(
        path=prefix.with_name(prefix.name + "_trajectory_fan_normalized.png"),
        fan_rows=trajectory_fan_norm_rows,
        corpus_counts=fan_corpus_counts,
        show_mean_error_bars=bool(args.fan_show_mean_error_bars),
    )

    if cluster_model is not None:
        grouped_human = _rows_grouped_by_cluster(
            human_rows,
            cluster_model=cluster_model,
        )
        grouped_structure = _rows_grouped_by_cluster(
            structure_rows,
            cluster_model=cluster_model,
        )
        grouped_full = _rows_grouped_by_cluster(
            full_rows,
            cluster_model=cluster_model,
        )
        grouped_vanilla = (
            _rows_grouped_by_cluster(vanilla_rows, cluster_model=cluster_model)
            if include_vanilla
            else {cluster_id: [] for cluster_id in cluster_model.cluster_ids}
        )

        grouped_for_mix: dict[str, dict[int, list[RoundTrajectory]]] = {
            "human_reference": grouped_human,
            "structure_target": grouped_structure,
            "full_simulated_target": grouped_full,
        }
        if include_vanilla:
            grouped_for_mix["vanilla_llm_target"] = grouped_vanilla

        cluster_mix_rows = _cluster_proportion_rows(
            grouped_for_mix,
            cluster_model=cluster_model,
        )
        _write_csv(
            prefix.with_name(prefix.name + "_cluster_proportions_long.csv"),
            cluster_mix_rows,
            [
                "corpus",
                "cluster_id",
                "cluster_name",
                "count",
                "total",
                "proportion",
            ],
        )
        _plot_cluster_proportions(
            path=prefix.with_name(prefix.name + "_cluster_proportions.png"),
            proportion_rows=cluster_mix_rows,
            cluster_model=cluster_model,
        )

        cluster_mix_wide_rows: list[dict[str, object]] = []
        for cluster_id, cluster_name in cluster_model.cluster_items:
            human_count = len(grouped_human.get(cluster_id, []))
            structure_count = len(grouped_structure.get(cluster_id, []))
            full_count = len(grouped_full.get(cluster_id, []))
            vanilla_count = len(grouped_vanilla.get(cluster_id, []))
            cluster_mix_wide_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_name": cluster_name,
                    "human_prop": (
                        float(human_count / len(human_rows))
                        if human_rows
                        else float("nan")
                    ),
                    "structure_prop": (
                        float(structure_count / len(structure_rows))
                        if structure_rows
                        else float("nan")
                    ),
                    "full_prop": (
                        float(full_count / len(full_rows))
                        if full_rows
                        else float("nan")
                    ),
                    "vanilla_prop": (
                        float(vanilla_count / len(vanilla_rows))
                        if include_vanilla
                        else float("nan")
                    ),
                    "human_n": int(human_count),
                    "structure_n": int(structure_count),
                    "full_n": int(full_count),
                    "vanilla_n": int(vanilla_count),
                }
            )

        print_table(
            cluster_mix_wide_rows,
            title="\nCluster Proportions By Corpus",
            columns=[
                "cluster_id",
                "cluster_name",
                "human_prop",
                "structure_prop",
                "full_prop",
                "vanilla_prop",
                "human_n",
                "structure_n",
                "full_n",
                "vanilla_n",
            ],
            aligns={
                "cluster_id": "right",
                "human_prop": "right",
                "structure_prop": "right",
                "full_prop": "right",
                "vanilla_prop": "right",
                "human_n": "right",
                "structure_n": "right",
                "full_n": "right",
                "vanilla_n": "right",
            },
            formatters={
                "human_prop": lambda value: f"{value:.3f}",
                "structure_prop": lambda value: f"{value:.3f}",
                "full_prop": lambda value: f"{value:.3f}",
                "vanilla_prop": (
                    lambda value: f"{value:.3f}" if np.isfinite(float(value)) else ""
                ),
            },
        )
        _write_csv(
            prefix.with_name(prefix.name + "_cluster_proportions.csv"),
            cluster_mix_wide_rows,
            [
                "cluster_id",
                "cluster_name",
                "human_prop",
                "structure_prop",
                "full_prop",
                "vanilla_prop",
                "human_n",
                "structure_n",
                "full_n",
                "vanilla_n",
            ],
        )
        cluster_init_bin_rows = _cluster_init_bin_rows(
            grouped_for_mix,
            cluster_model=cluster_model,
        )
        _write_csv(
            prefix.with_name(prefix.name + "_cluster_by_init_bin_long.csv"),
            cluster_init_bin_rows,
            [
                "corpus",
                "cluster_id",
                "cluster_name",
                "init_belief_bin",
                "count",
                "cluster_total",
                "init_bin_total",
                "corpus_total",
                "prop_within_corpus",
                "prop_within_cluster",
                "prop_within_init_bin",
            ],
        )
        _plot_cluster_init_bin_heatmaps(
            path=prefix.with_name(prefix.name + "_cluster_by_init_bin_heatmap.png"),
            rows=cluster_init_bin_rows,
            cluster_model=cluster_model,
        )

        cluster_primary_rows: list[dict[str, object]] = []
        for cluster_id, cluster_name in cluster_model.cluster_items:
            human_cluster_rows = grouped_human.get(cluster_id, [])
            structure_cluster_rows = grouped_structure.get(cluster_id, [])
            full_cluster_rows = grouped_full.get(cluster_id, [])
            vanilla_cluster_rows = grouped_vanilla.get(cluster_id, [])

            has_required = bool(
                human_cluster_rows and structure_cluster_rows and full_cluster_rows
            )
            if has_required:
                human_updates = _flatten_updates(human_cluster_rows)
                structure_updates = _flatten_updates(structure_cluster_rows)
                full_updates = _flatten_updates(full_cluster_rows)
                vanilla_updates = _flatten_updates(vanilla_cluster_rows)
                edges = _histogram_edges_for_arrays(
                    arrays=[
                        human_updates,
                        structure_updates,
                        full_updates,
                        vanilla_updates,
                    ],
                    n_bins=max(5, int(args.jsd_bins)),
                )
                structure_jsd_cluster = _pooled_jsd_from_arrays(
                    reference_values=human_updates,
                    candidate_values=structure_updates,
                    edges=edges,
                )
                full_jsd_cluster = _pooled_jsd_from_arrays(
                    reference_values=human_updates,
                    candidate_values=full_updates,
                    edges=edges,
                )
                vanilla_jsd_cluster = _pooled_jsd_from_arrays(
                    reference_values=human_updates,
                    candidate_values=vanilla_updates,
                    edges=edges,
                )

                structure_shape_cluster = _normalized_trajectory_shape_w1(
                    human_cluster_rows,
                    structure_cluster_rows,
                    grid_points=int(args.normalized_grid_points),
                )
                full_shape_cluster = _normalized_trajectory_shape_w1(
                    human_cluster_rows,
                    full_cluster_rows,
                    grid_points=int(args.normalized_grid_points),
                )
                vanilla_shape_cluster = _normalized_trajectory_shape_w1(
                    human_cluster_rows,
                    vanilla_cluster_rows,
                    grid_points=int(args.normalized_grid_points),
                )
                structure_w1_cluster = _pooled_w1(
                    human_cluster_rows, structure_cluster_rows
                )
                full_w1_cluster = _pooled_w1(human_cluster_rows, full_cluster_rows)
                vanilla_w1_cluster = _pooled_w1(
                    human_cluster_rows, vanilla_cluster_rows
                )
            else:
                structure_jsd_cluster = float("nan")
                full_jsd_cluster = float("nan")
                vanilla_jsd_cluster = float("nan")
                structure_shape_cluster = float("nan")
                full_shape_cluster = float("nan")
                vanilla_shape_cluster = float("nan")
                structure_w1_cluster = float("nan")
                full_w1_cluster = float("nan")
                vanilla_w1_cluster = float("nan")

            include_vanilla_cluster = bool(vanilla_cluster_rows)
            pooled_jsd_candidates: list[tuple[str, float]] = []
            shape_candidates: list[tuple[str, float]] = []
            pooled_w1_candidates: list[tuple[str, float]] = []
            if np.isfinite(structure_jsd_cluster):
                pooled_jsd_candidates.append(("structure", structure_jsd_cluster))
            if np.isfinite(full_jsd_cluster):
                pooled_jsd_candidates.append(("full", full_jsd_cluster))
            if np.isfinite(structure_shape_cluster):
                shape_candidates.append(("structure", structure_shape_cluster))
            if np.isfinite(full_shape_cluster):
                shape_candidates.append(("full", full_shape_cluster))
            if np.isfinite(structure_w1_cluster):
                pooled_w1_candidates.append(("structure", structure_w1_cluster))
            if np.isfinite(full_w1_cluster):
                pooled_w1_candidates.append(("full", full_w1_cluster))
            if include_vanilla_cluster:
                if np.isfinite(vanilla_jsd_cluster):
                    pooled_jsd_candidates.append(("vanilla", vanilla_jsd_cluster))
                if np.isfinite(vanilla_shape_cluster):
                    shape_candidates.append(("vanilla", vanilla_shape_cluster))
                if np.isfinite(vanilla_w1_cluster):
                    pooled_w1_candidates.append(("vanilla", vanilla_w1_cluster))

            cluster_primary_rows.extend(
                [
                    {
                        "cluster_id": int(cluster_id),
                        "cluster_name": cluster_name,
                        "metric": "pooled_jsd",
                        "human_n": int(len(human_cluster_rows)),
                        "structure": structure_jsd_cluster,
                        "full": full_jsd_cluster,
                        "vanilla": (
                            vanilla_jsd_cluster
                            if include_vanilla_cluster
                            else float("nan")
                        ),
                        "winner": (
                            min(pooled_jsd_candidates, key=lambda item: item[1])[0]
                            if pooled_jsd_candidates
                            else ""
                        ),
                    },
                    {
                        "cluster_id": int(cluster_id),
                        "cluster_name": cluster_name,
                        "metric": "normalized_trajectory_shape_wasserstein",
                        "human_n": int(len(human_cluster_rows)),
                        "structure": structure_shape_cluster,
                        "full": full_shape_cluster,
                        "vanilla": (
                            vanilla_shape_cluster
                            if include_vanilla_cluster
                            else float("nan")
                        ),
                        "winner": (
                            min(shape_candidates, key=lambda item: item[1])[0]
                            if shape_candidates
                            else ""
                        ),
                    },
                    {
                        "cluster_id": int(cluster_id),
                        "cluster_name": cluster_name,
                        "metric": "pooled_wasserstein",
                        "human_n": int(len(human_cluster_rows)),
                        "structure": structure_w1_cluster,
                        "full": full_w1_cluster,
                        "vanilla": (
                            vanilla_w1_cluster
                            if include_vanilla_cluster
                            else float("nan")
                        ),
                        "winner": (
                            min(pooled_w1_candidates, key=lambda item: item[1])[0]
                            if pooled_w1_candidates
                            else ""
                        ),
                    },
                ]
            )

            cluster_slug = safe_slug(cluster_name, max_chars=64, default="cluster")
            cluster_prefix = prefix.with_name(
                prefix.name + f"_cluster_{int(cluster_id)}_{cluster_slug}"
            )
            cluster_corpus_counts: dict[str, int] = {
                "human_reference": int(len(human_cluster_rows)),
                "structure_target": int(len(structure_cluster_rows)),
                "full_simulated_target": int(len(full_cluster_rows)),
                "human": int(len(human_cluster_rows)),
                "structure": int(len(structure_cluster_rows)),
                "full": int(len(full_cluster_rows)),
            }
            if include_vanilla_cluster:
                cluster_corpus_counts["vanilla_llm_target"] = int(
                    len(vanilla_cluster_rows)
                )
                cluster_corpus_counts["vanilla"] = int(len(vanilla_cluster_rows))

            cluster_fan_rows: list[dict[str, object]] = []
            cluster_fan_rows.extend(
                _trajectory_fan_rows(
                    corpus="human_reference",
                    rows=human_cluster_rows,
                )
            )
            cluster_fan_rows.extend(
                _trajectory_fan_rows(
                    corpus="structure_target",
                    rows=structure_cluster_rows,
                )
            )
            cluster_fan_rows.extend(
                _trajectory_fan_rows(
                    corpus="full_simulated_target",
                    rows=full_cluster_rows,
                )
            )
            if include_vanilla_cluster:
                cluster_fan_rows.extend(
                    _trajectory_fan_rows(
                        corpus="vanilla_llm_target",
                        rows=vanilla_cluster_rows,
                    )
                )
            _write_csv(
                cluster_prefix.with_name(cluster_prefix.name + "_trajectory_fan.csv"),
                cluster_fan_rows,
                TRAJECTORY_FAN_COLUMNS,
            )
            _plot_trajectory_fan(
                path=cluster_prefix.with_name(
                    cluster_prefix.name + "_trajectory_fan.png"
                ),
                fan_rows=cluster_fan_rows,
                min_n_per_turn=max(1, int(args.plot_min_n_per_turn)),
                corpus_counts=cluster_corpus_counts,
                show_mean_error_bars=bool(args.fan_show_mean_error_bars),
            )
            _plot_per_turn_delta_boxes(
                path=cluster_prefix.with_name(
                    cluster_prefix.name + "_per_turn_delta_box.png"
                ),
                human_rows=human_cluster_rows,
                structure_rows=structure_cluster_rows,
                full_rows=full_cluster_rows,
                vanilla_rows=vanilla_cluster_rows,
                include_vanilla=include_vanilla_cluster,
                min_n_per_turn=max(1, int(args.plot_min_n_per_turn)),
                corpus_counts=cluster_corpus_counts,
            )
            cluster_norm_rows: list[dict[str, object]] = []
            cluster_norm_rows.extend(
                _trajectory_fan_rows_normalized(
                    corpus="human_reference",
                    rows=human_cluster_rows,
                    grid_points=grid_points,
                )
            )
            cluster_norm_rows.extend(
                _trajectory_fan_rows_normalized(
                    corpus="structure_target",
                    rows=structure_cluster_rows,
                    grid_points=grid_points,
                )
            )
            cluster_norm_rows.extend(
                _trajectory_fan_rows_normalized(
                    corpus="full_simulated_target",
                    rows=full_cluster_rows,
                    grid_points=grid_points,
                )
            )
            if include_vanilla_cluster:
                cluster_norm_rows.extend(
                    _trajectory_fan_rows_normalized(
                        corpus="vanilla_llm_target",
                        rows=vanilla_cluster_rows,
                        grid_points=grid_points,
                    )
                )
            _write_csv(
                cluster_prefix.with_name(
                    cluster_prefix.name + "_trajectory_fan_normalized.csv"
                ),
                cluster_norm_rows,
                TRAJECTORY_FAN_NORMALIZED_COLUMNS,
            )
            _plot_trajectory_fan_normalized(
                path=cluster_prefix.with_name(
                    cluster_prefix.name + "_trajectory_fan_normalized.png"
                ),
                fan_rows=cluster_norm_rows,
                corpus_counts=cluster_corpus_counts,
                show_mean_error_bars=bool(args.fan_show_mean_error_bars),
            )

        if cluster_primary_rows:
            print_table(
                cluster_primary_rows,
                title="\nPer-Cluster Primary Distances (Lower Is Better)",
                columns=[
                    "cluster_id",
                    "cluster_name",
                    "metric",
                    "human_n",
                    "structure",
                    "full",
                    "vanilla",
                    "winner",
                ],
                aligns={
                    "cluster_id": "right",
                    "human_n": "right",
                    "structure": "right",
                    "full": "right",
                    "vanilla": "right",
                },
                formatters={
                    "structure": lambda value: f"{value:.4f}",
                    "full": lambda value: f"{value:.4f}",
                    "vanilla": (
                        lambda value: (
                            f"{value:.4f}" if np.isfinite(float(value)) else ""
                        )
                    ),
                },
            )
            _write_csv(
                prefix.with_name(prefix.name + "_cluster_primary.csv"),
                cluster_primary_rows,
                [
                    "cluster_id",
                    "cluster_name",
                    "metric",
                    "human_n",
                    "structure",
                    "full",
                    "vanilla",
                    "winner",
                ],
            )

    if include_vanilla:
        _write_csv(
            prefix.with_name(prefix.name + "_primary_with_vanilla.csv"),
            vanilla_primary_rows,
            ["metric", "structure", "full", "vanilla", "winner"],
        )
        _write_csv(
            prefix.with_name(prefix.name + "_secondary_with_vanilla.csv"),
            secondary_with_vanilla_rows,
            [
                "metric",
                "human",
                "structure",
                "full",
                "vanilla",
                "abs_err_structure",
                "abs_err_full",
                "abs_err_vanilla",
                "winner",
            ],
        )


if __name__ == "__main__":
    main()
