"""Compute and plot naive-policy movement penalties from proposition rows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np

from simulation.human_likeness_eval.corpus_variants import _split_policy_variant_corpus

from .simulator_plot_cli import add_initial_belief_match_tolerance_argument
from .simulator_plot_style import (
    COMPARISON_CORPUS_COLOR_MAP,
    CORE_COMPARISON_CORPUS_ORDER,
    PAPER_RESULTS_FIGURE_SIZE_INCHES,
    comparison_corpus_tick_label,
)
from .utils import resolve_repo_path, safe_float_or_nan, safe_int_or_none

DEFAULT_INPUT_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_proposition_stance_deltas_by_policy.csv"
)
DEFAULT_ROUND_DYNAMICS_INPUT_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_round_dynamics_by_policy.csv"
)
DEFAULT_OUTPUT_CSV = Path("analysis/data/rl_human_match_sim_compare_naive_penalty.csv")
DEFAULT_OUTPUT_PDF = Path("analysis/figures/results_naive_penalty.pdf")

# Use a non-interactive backend for CLI stability in headless environments.
plt.switch_backend("Agg")


@dataclass(frozen=True)
class PolicyPropRow:
    """Store one policy-specific proposition/stance aggregate row.

    Attributes:
        base_corpus: Base corpus key without policy suffix.
        policy_model: Policy model id from corpus suffix.
        proposition: Proposition text.
        stance: Persuader stance label.
        n_rounds: Aggregate round count.
        mean_abs_total_delta: Mean absolute total movement for the aggregate cell.
    """

    base_corpus: str
    policy_model: str
    proposition: str
    stance: str
    n_rounds: int
    mean_abs_total_delta: float


@dataclass(frozen=True)
class PolicyRoundRow:
    """Store one round-level policy row with initial belief.

    Attributes:
        base_corpus: Base corpus key without policy suffix.
        policy_model: Policy model id from corpus suffix.
        proposition: Proposition text.
        stance: Persuader stance label.
        initial_belief: Initial belief used for the round.
        abs_total_delta: Absolute total movement for the round.
    """

    base_corpus: str
    policy_model: str
    proposition: str
    stance: str
    initial_belief: float
    abs_total_delta: float


@dataclass(frozen=True)
class NaivePenaltyRow:
    """Store one corpus-level naive-penalty summary.

    Attributes:
        corpus: Base corpus key.
        paired_cells: Number of proposition/stance cells shared by both policies.
        paired_round_weight: Total weight used in weighted means.
        baseline_weighted_mean_abs_delta: Weighted mean absolute movement for baseline.
        naive_weighted_mean_abs_delta: Weighted mean absolute movement for naive policy.
        naive_excess_abs_delta: Difference ``naive - baseline`` in weighted means.
    """

    corpus: str
    paired_cells: int
    paired_round_weight: int
    baseline_weighted_mean_abs_delta: float
    naive_weighted_mean_abs_delta: float
    naive_excess_abs_delta: float
    naive_excess_abs_delta_abs: float
    naive_excess_abs_delta_penalty: float
    naive_excess_abs_delta_ci_low: float
    naive_excess_abs_delta_ci_high: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for naive-penalty analysis.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how much more a simulator moves under naive persuaders "
            "relative to a non-naive policy model."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input proposition stance CSV with policy-suffixed corpus keys.",
    )
    parser.add_argument(
        "--round-dynamics-by-policy-csv",
        type=Path,
        default=DEFAULT_ROUND_DYNAMICS_INPUT_CSV,
        help="Round-level by-policy dynamics CSV used for initial-belief matching.",
    )
    parser.add_argument(
        "--non-naive-policy-model",
        type=str,
        default=None,
        help="Non-naive policy model id used for naive-vs-non-naive comparison.",
    )
    parser.add_argument(
        "--reference-policy-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--baseline-policy-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--naive-policy-model",
        type=str,
        default="naive",
        help="Naive policy model id.",
    )
    parser.add_argument(
        "--min-rounds-per-cell",
        type=int,
        default=3,
        help="Minimum n_rounds required for each compared proposition/stance cell.",
    )
    add_initial_belief_match_tolerance_argument(
        parser,
        help_text=(
            "Optional one-to-one initial-belief matching tolerance. When set, "
            "compute per-cell policy means from matched round pairs with "
            "|initial_belief_non_naive - initial_belief_naive| <= tolerance."
        ),
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=4000,
        help="Number of bootstrap draws for 95% confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=17,
        help="Random seed for bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for corpus-level naive-penalty rows.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path for the naive-penalty figure.",
    )
    return parser.parse_args()


def _resolve_non_naive_policy_model(args: argparse.Namespace) -> str:
    """Resolve the non-naive policy model id from CLI arguments.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Effective non-naive policy model id.
    """
    non_naive = str(args.non_naive_policy_model or "").strip()
    if non_naive:
        return non_naive
    legacy_reference = str(args.reference_policy_model or "").strip()
    if legacy_reference:
        return legacy_reference
    legacy = str(args.baseline_policy_model or "").strip()
    if legacy:
        return legacy
    raise ValueError("A non-naive policy model is required.")


def _parse_policy_row_fields(raw: dict[str, str]) -> tuple[str, str, str, str] | None:
    """Parse shared corpus/proposition/stance fields for policy CSV rows.

    Args:
        raw: One CSV row mapping.

    Returns:
        Tuple ``(base_corpus, policy_model, proposition, stance)`` or ``None``.
    """
    corpus = str(raw.get("corpus") or "").strip()
    if not corpus:
        return None
    base_corpus, policy_model = _split_policy_variant_corpus(corpus)
    if base_corpus == "human_reference" or policy_model is None:
        return None
    if base_corpus not in CORE_COMPARISON_CORPUS_ORDER:
        return None
    proposition = str(raw.get("proposition") or "").strip()
    stance = str(raw.get("stance") or "").strip()
    if not proposition or not stance:
        return None
    return base_corpus, policy_model, proposition, stance


def _iter_valid_policy_rows(
    input_csv: Path,
) -> Iterator[tuple[tuple[str, str, str, str], dict[str, str]]]:
    """Yield rows with shared policy/corpus/proposition/stance fields pre-validated.

    Args:
        input_csv: Input CSV path.

    Yields:
        Tuples ``(parsed_fields, raw_row)`` where parsed_fields is
        ``(base_corpus, policy_model, proposition, stance)``.
    """
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            parsed = _parse_policy_row_fields(raw)
            if parsed is not None:
                yield parsed, raw


def _load_rows(input_csv: Path, *, min_rounds_per_cell: int) -> list[PolicyPropRow]:
    """Load and filter policy-specific proposition rows.

    Args:
        input_csv: Input CSV path.
        min_rounds_per_cell: Minimum rounds required per row.

    Returns:
        Filtered policy proposition rows.
    """
    rows: list[PolicyPropRow] = []
    for parsed, raw in _iter_valid_policy_rows(input_csv):
        base_corpus, policy_model, proposition, stance = parsed
        n_rounds = safe_int_or_none(raw.get("n_rounds"))
        if n_rounds is None or n_rounds < min_rounds_per_cell:
            continue
        mean_abs = safe_float_or_nan(raw.get("mean_abs_total_delta"))
        if not np.isfinite(mean_abs):
            continue
        rows.append(
            PolicyPropRow(
                base_corpus=base_corpus,
                policy_model=policy_model,
                proposition=proposition,
                stance=stance,
                n_rounds=int(n_rounds),
                mean_abs_total_delta=float(mean_abs),
            )
        )
    return rows


def _load_round_rows(input_csv: Path) -> list[PolicyRoundRow]:
    """Load round-level policy rows with initial belief and absolute movement.

    Args:
        input_csv: Round-level by-policy input CSV path.

    Returns:
        Parsed round-level policy rows.
    """
    rows: list[PolicyRoundRow] = []
    for parsed, raw in _iter_valid_policy_rows(input_csv):
        base_corpus, policy_model, proposition, stance = parsed
        initial_belief = safe_float_or_nan(raw.get("initial_belief"))
        abs_total_delta = safe_float_or_nan(raw.get("abs_total_delta"))
        if not np.isfinite(initial_belief) or not np.isfinite(abs_total_delta):
            continue
        rows.append(
            PolicyRoundRow(
                base_corpus=base_corpus,
                policy_model=policy_model,
                proposition=proposition,
                stance=stance,
                initial_belief=float(initial_belief),
                abs_total_delta=float(abs_total_delta),
            )
        )
    return rows


def _matched_abs_means_for_cell(
    *,
    baseline_rows: list[PolicyRoundRow],
    naive_rows: list[PolicyRoundRow],
    tolerance: float,
) -> tuple[float, float, int] | None:
    """Build one-to-one initial-belief matches for one proposition/stance cell.

    Args:
        baseline_rows: Non-naive rows for one cell.
        naive_rows: Naive rows for one cell.
        tolerance: Maximum absolute initial-belief distance allowed per match.

    Returns:
        Tuple ``(baseline_mean_abs, naive_mean_abs, matched_count)`` or None.
    """
    if not baseline_rows or not naive_rows:
        return None
    baseline_order = sorted(
        range(len(baseline_rows)),
        key=lambda index: float(baseline_rows[index].initial_belief),
    )
    naive_used = [False] * len(naive_rows)
    baseline_matches: list[float] = []
    naive_matches: list[float] = []
    for base_index in baseline_order:
        baseline_row = baseline_rows[base_index]
        baseline_initial = float(baseline_row.initial_belief)
        best_naive_index: int | None = None
        best_distance: float | None = None
        for naive_index, naive_row in enumerate(naive_rows):
            if naive_used[naive_index]:
                continue
            distance = abs(baseline_initial - float(naive_row.initial_belief))
            if distance > tolerance:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_naive_index = naive_index
        if best_naive_index is None:
            continue
        naive_used[best_naive_index] = True
        baseline_matches.append(float(baseline_row.abs_total_delta))
        naive_matches.append(float(naive_rows[best_naive_index].abs_total_delta))
    matched_count = len(baseline_matches)
    if matched_count <= 0:
        return None
    return (
        float(np.mean(np.asarray(baseline_matches, dtype=float))),
        float(np.mean(np.asarray(naive_matches, dtype=float))),
        int(matched_count),
    )


def _load_rows_with_initial_belief_matching(
    *,
    input_csv: Path,
    min_rounds_per_cell: int,
    baseline_policy_model: str,
    naive_policy_model: str,
    tolerance: float,
) -> list[PolicyPropRow]:
    """Build policy proposition rows from round-level initial-belief matching.

    Args:
        input_csv: Round-level by-policy CSV path.
        min_rounds_per_cell: Minimum rows per policy required before matching.
        baseline_policy_model: Non-naive policy model id.
        naive_policy_model: Naive policy model id.
        tolerance: Maximum allowed initial-belief distance per matched pair.

    Returns:
        Policy proposition rows using matched-pair means and counts.
    """
    round_rows = _load_round_rows(input_csv=input_csv)
    grouped: dict[tuple[str, str, str, str], list[PolicyRoundRow]] = {}
    for row in round_rows:
        grouped.setdefault(
            (row.base_corpus, row.proposition, row.stance, row.policy_model), []
        ).append(row)

    output: list[PolicyPropRow] = []
    cells = sorted(
        {
            (base_corpus, proposition, stance)
            for base_corpus, proposition, stance, _policy in grouped
            if base_corpus in CORE_COMPARISON_CORPUS_ORDER
        }
    )
    for base_corpus, proposition, stance in cells:
        baseline_rows = grouped.get(
            (base_corpus, proposition, stance, baseline_policy_model), []
        )
        naive_rows = grouped.get(
            (base_corpus, proposition, stance, naive_policy_model), []
        )
        if (
            len(baseline_rows) < min_rounds_per_cell
            or len(naive_rows) < min_rounds_per_cell
        ):
            continue
        matched = _matched_abs_means_for_cell(
            baseline_rows=baseline_rows,
            naive_rows=naive_rows,
            tolerance=tolerance,
        )
        if matched is None:
            continue
        baseline_mean, naive_mean, matched_count = matched
        output.append(
            PolicyPropRow(
                base_corpus=base_corpus,
                policy_model=baseline_policy_model,
                proposition=proposition,
                stance=stance,
                n_rounds=int(matched_count),
                mean_abs_total_delta=float(baseline_mean),
            )
        )
        output.append(
            PolicyPropRow(
                base_corpus=base_corpus,
                policy_model=naive_policy_model,
                proposition=proposition,
                stance=stance,
                n_rounds=int(matched_count),
                mean_abs_total_delta=float(naive_mean),
            )
        )
    return output


def _compute_penalty_rows(
    rows: list[PolicyPropRow],
    *,
    baseline_policy_model: str,
    naive_policy_model: str,
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> tuple[list[NaivePenaltyRow], list[tuple[PolicyPropRow, PolicyPropRow, int]]]:
    """Compute per-corpus naive penalties using shared proposition cells.

    Args:
        rows: Loaded policy proposition rows.
        baseline_policy_model: Baseline policy model id.
        naive_policy_model: Naive policy model id.

    Returns:
        Tuple of sorted corpus-level rows and all shared paired rows.
    """
    keyed: dict[tuple[str, str, str, str], PolicyPropRow] = {}
    for row in rows:
        keyed[(row.base_corpus, row.proposition, row.stance, row.policy_model)] = row

    output: list[NaivePenaltyRow] = []
    all_pairs: list[tuple[PolicyPropRow, PolicyPropRow, int]] = []
    for base_corpus in CORE_COMPARISON_CORPUS_ORDER:
        if base_corpus == "human_reference":
            continue
        paired_rows = _paired_rows_for_corpus(
            keyed=keyed,
            base_corpus=base_corpus,
            baseline_policy_model=baseline_policy_model,
            naive_policy_model=naive_policy_model,
        )
        summary_row = _corpus_penalty_row(
            base_corpus=base_corpus,
            paired_rows=paired_rows,
            bootstrap_draws=bootstrap_draws,
            rng=rng,
        )
        if summary_row is not None:
            output.append(summary_row)
            all_pairs.extend(paired_rows)
    return output, all_pairs


def _corpus_penalty_row(
    *,
    base_corpus: str,
    paired_rows: list[tuple[PolicyPropRow, PolicyPropRow, int]],
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> NaivePenaltyRow | None:
    """Compute one corpus-level naive penalty row.

    Args:
        base_corpus: Base corpus key.
        paired_rows: Shared proposition/stance row pairs.
        bootstrap_draws: Number of bootstrap draws.
        rng: Random generator.

    Returns:
        Naive penalty summary row, or None when no shared cells exist.
    """
    paired_cells = len(paired_rows)
    baseline_value, naive_value, total_weight = _weighted_means_from_pairs(paired_rows)
    if total_weight <= 0:
        return None
    excess_value = float(naive_value - baseline_value)
    ci_low, ci_high = _bootstrap_ci_from_pairs(
        paired_rows=paired_rows,
        bootstrap_draws=bootstrap_draws,
        rng=rng,
    )
    return NaivePenaltyRow(
        corpus=base_corpus,
        paired_cells=int(paired_cells),
        paired_round_weight=int(total_weight),
        baseline_weighted_mean_abs_delta=baseline_value,
        naive_weighted_mean_abs_delta=naive_value,
        naive_excess_abs_delta=excess_value,
        naive_excess_abs_delta_abs=abs(excess_value),
        naive_excess_abs_delta_penalty=max(0.0, excess_value),
        naive_excess_abs_delta_ci_low=ci_low,
        naive_excess_abs_delta_ci_high=ci_high,
    )


def _weighted_means_from_pairs(
    paired_rows: list[tuple[PolicyPropRow, PolicyPropRow, int]],
) -> tuple[float, float, int]:
    """Compute weighted baseline/naive means for paired proposition rows.

    Args:
        paired_rows: Shared proposition/stance row tuples.

    Returns:
        Tuple of ``(baseline_mean, naive_mean, total_weight)``.
    """
    total_weight = int(sum(weight for _, _, weight in paired_rows))
    if total_weight <= 0:
        return float("nan"), float("nan"), 0
    baseline_weighted = sum(
        float(weight) * float(row.mean_abs_total_delta)
        for row, _, weight in paired_rows
    )
    naive_weighted = sum(
        float(weight) * float(row.mean_abs_total_delta)
        for _, row, weight in paired_rows
    )
    return (
        float(baseline_weighted / float(total_weight)),
        float(naive_weighted / float(total_weight)),
        int(total_weight),
    )


def _bootstrap_ci_from_pairs(
    *,
    paired_rows: list[tuple[PolicyPropRow, PolicyPropRow, int]],
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a 95% CI for signed naive excess movement.

    Args:
        paired_rows: Shared proposition/stance row tuples.
        bootstrap_draws: Number of bootstrap draws.
        rng: Random generator.

    Returns:
        Tuple of ``(ci_low, ci_high)``.
    """
    if not paired_rows:
        return float("nan"), float("nan")
    draws = max(1, int(bootstrap_draws))
    baseline_values = np.asarray(
        [float(row.mean_abs_total_delta) for row, _, _ in paired_rows],
        dtype=float,
    )
    naive_values = np.asarray(
        [float(row.mean_abs_total_delta) for _, row, _ in paired_rows],
        dtype=float,
    )
    weights = np.asarray([int(weight) for _, _, weight in paired_rows], dtype=float)
    n_pairs = int(baseline_values.size)
    if n_pairs <= 0:
        return float("nan"), float("nan")

    boot = np.empty(draws, dtype=float)
    for draw_index in range(draws):
        sample_idx = rng.integers(0, n_pairs, size=n_pairs)
        sample_weights = weights[sample_idx]
        total_weight = float(np.sum(sample_weights))
        if total_weight <= 0:
            boot[draw_index] = float("nan")
            continue
        baseline_mean = float(
            np.sum(sample_weights * baseline_values[sample_idx]) / total_weight
        )
        naive_mean = float(
            np.sum(sample_weights * naive_values[sample_idx]) / total_weight
        )
        boot[draw_index] = float(naive_mean - baseline_mean)

    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))


def _paired_rows_for_corpus(
    *,
    keyed: dict[tuple[str, str, str, str], PolicyPropRow],
    base_corpus: str,
    baseline_policy_model: str,
    naive_policy_model: str,
) -> list[tuple[PolicyPropRow, PolicyPropRow, int]]:
    """Collect shared proposition/stance rows for one corpus.

    Args:
        keyed: Mapping of policy proposition rows.
        base_corpus: Base corpus key.
        baseline_policy_model: Baseline policy model id.
        naive_policy_model: Naive policy model id.

    Returns:
        Tuples of ``(baseline_row, naive_row, weight)``.
    """
    pairs: list[tuple[PolicyPropRow, PolicyPropRow, int]] = []
    cell_keys = sorted(
        {
            (prop, stance)
            for corpus, prop, stance, policy in keyed
            if corpus == base_corpus
            and policy in {baseline_policy_model, naive_policy_model}
        }
    )
    for proposition, stance in cell_keys:
        baseline_row = keyed.get(
            (base_corpus, proposition, stance, baseline_policy_model)
        )
        naive_row = keyed.get((base_corpus, proposition, stance, naive_policy_model))
        if baseline_row is None or naive_row is None:
            continue
        weight = min(int(baseline_row.n_rounds), int(naive_row.n_rounds))
        if weight <= 0:
            continue
        pairs.append((baseline_row, naive_row, int(weight)))
    return pairs


def _overall_row(
    rows: list[NaivePenaltyRow],
    *,
    overall_pairs: list[tuple[PolicyPropRow, PolicyPropRow, int]],
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> NaivePenaltyRow | None:
    """Aggregate one overall naive-penalty row across corpora.

    Args:
        rows: Corpus-level naive-penalty rows.
        overall_pairs: Shared proposition/stance row pairs across corpora.
        bootstrap_draws: Number of bootstrap draws.
        rng: Random generator.

    Returns:
        Overall weighted row, or None when rows are empty.
    """
    if not rows:
        return None
    baseline_value, naive_value, total_weight = _weighted_means_from_pairs(
        overall_pairs
    )
    if total_weight <= 0:
        return None
    excess_value = float(naive_value - baseline_value)
    ci_low, ci_high = _bootstrap_ci_from_pairs(
        paired_rows=overall_pairs,
        bootstrap_draws=bootstrap_draws,
        rng=rng,
    )
    return NaivePenaltyRow(
        corpus="overall",
        paired_cells=int(sum(int(row.paired_cells) for row in rows)),
        paired_round_weight=total_weight,
        baseline_weighted_mean_abs_delta=baseline_value,
        naive_weighted_mean_abs_delta=naive_value,
        naive_excess_abs_delta=excess_value,
        naive_excess_abs_delta_abs=abs(excess_value),
        naive_excess_abs_delta_penalty=max(0.0, excess_value),
        naive_excess_abs_delta_ci_low=ci_low,
        naive_excess_abs_delta_ci_high=ci_high,
    )


def _write_output_csv(
    path: Path,
    rows: list[NaivePenaltyRow],
    *,
    non_naive_policy_model: str,
    naive_policy_model: str,
) -> None:
    """Write naive-penalty summary rows to CSV.

    Args:
        path: Output CSV path.
        rows: Rows to write.
        non_naive_policy_model: Non-naive policy model id.
        naive_policy_model: Naive policy model id.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "corpus",
                "paired_cells",
                "paired_round_weight",
                "non_naive_policy_model",
                "naive_policy_model",
                "non_naive_weighted_mean_abs_delta",
                "naive_weighted_mean_abs_delta",
                "naive_excess_abs_delta",
                "naive_excess_abs_delta_abs",
                "naive_excess_abs_delta_penalty",
                "naive_excess_abs_delta_ci_low",
                "naive_excess_abs_delta_ci_high",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "corpus": row.corpus,
                    "paired_cells": row.paired_cells,
                    "paired_round_weight": row.paired_round_weight,
                    "non_naive_policy_model": non_naive_policy_model,
                    "naive_policy_model": naive_policy_model,
                    "non_naive_weighted_mean_abs_delta": row.baseline_weighted_mean_abs_delta,
                    "naive_weighted_mean_abs_delta": row.naive_weighted_mean_abs_delta,
                    "naive_excess_abs_delta": row.naive_excess_abs_delta,
                    "naive_excess_abs_delta_abs": row.naive_excess_abs_delta_abs,
                    "naive_excess_abs_delta_penalty": row.naive_excess_abs_delta_penalty,
                    "naive_excess_abs_delta_ci_low": row.naive_excess_abs_delta_ci_low,
                    "naive_excess_abs_delta_ci_high": row.naive_excess_abs_delta_ci_high,
                }
            )


def _render_plot(path: Path, rows: list[NaivePenaltyRow]) -> None:
    """Render a bar chart for naive-penalty values.

    Args:
        path: Output PDF path.
        rows: Corpus-level rows plus overall row.

    Returns:
        None.
    """
    corpus_rank = {
        corpus: index for index, corpus in enumerate(CORE_COMPARISON_CORPUS_ORDER)
    }
    corpus_rows = sorted(
        [row for row in rows if row.corpus != "overall"],
        key=lambda row: (
            float(row.naive_excess_abs_delta),
            corpus_rank.get(str(row.corpus), len(corpus_rank)),
        ),
    )
    if not corpus_rows:
        raise ValueError("No corpus-level rows available for plotting.")

    labels = [comparison_corpus_tick_label(row.corpus) for row in corpus_rows]
    values = [float(row.naive_excess_abs_delta) for row in corpus_rows]
    ci_lows = [float(row.naive_excess_abs_delta_ci_low) for row in corpus_rows]
    ci_highs = [float(row.naive_excess_abs_delta_ci_high) for row in corpus_rows]
    colors = [
        COMPARISON_CORPUS_COLOR_MAP.get(row.corpus, "#888888") for row in corpus_rows
    ]

    fig, axis = plt.subplots(figsize=PAPER_RESULTS_FIGURE_SIZE_INCHES)
    xpos = np.arange(len(corpus_rows), dtype=float)
    axis.bar(
        xpos,
        values,
        color=colors,
        width=0.62,
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.92,
        zorder=2,
    )
    lower_err = np.asarray(
        [
            max(0.0, value - ci_low) if np.isfinite(ci_low) else 0.0
            for value, ci_low in zip(values, ci_lows)
        ],
        dtype=float,
    )
    upper_err = np.asarray(
        [
            max(0.0, ci_high - value) if np.isfinite(ci_high) else 0.0
            for value, ci_high in zip(values, ci_highs)
        ],
        dtype=float,
    )
    axis.errorbar(
        xpos,
        values,
        yerr=np.vstack([lower_err, upper_err]),
        fmt="none",
        ecolor="#333333",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=3,
    )
    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_xticks(xpos)
    axis.set_xticklabels(labels, rotation=0, ha="center")
    axis.tick_params(axis="x", labelsize=6.8, pad=1.2)
    axis.set_ylabel(r"Naive Excess ($\leftarrow$)", fontsize=10)
    axis.grid(axis="y", linestyle=":", alpha=0.25)
    fig.subplots_adjust(left=0.26, bottom=0.18, right=0.98, top=0.96)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """Execute naive-penalty computation and plotting workflow."""
    args = parse_args()
    reference_file = Path(__file__).resolve()
    input_csv = resolve_repo_path(args.input_csv, reference_file=reference_file)
    round_input_csv = resolve_repo_path(
        args.round_dynamics_by_policy_csv,
        reference_file=reference_file,
    )
    output_csv = resolve_repo_path(args.output_csv, reference_file=reference_file)
    output_pdf = resolve_repo_path(args.output_pdf, reference_file=reference_file)

    bootstrap_draws = max(1, int(args.bootstrap_draws))
    rng = np.random.default_rng(int(args.bootstrap_seed))
    non_naive_policy_model = _resolve_non_naive_policy_model(args)
    min_rounds_per_cell = max(1, int(args.min_rounds_per_cell))
    tolerance_raw = args.initial_belief_match_tolerance

    if tolerance_raw is None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        rows = _load_rows(
            input_csv=input_csv,
            min_rounds_per_cell=min_rounds_per_cell,
        )
        if not rows:
            raise ValueError("No eligible policy rows found in the input CSV.")
    else:
        tolerance = float(tolerance_raw)
        if tolerance <= 0.0:
            raise ValueError("initial_belief_match_tolerance must be > 0.")
        if not round_input_csv.exists():
            raise FileNotFoundError(
                f"Round-dynamics by-policy CSV not found: {round_input_csv}"
            )
        rows = _load_rows_with_initial_belief_matching(
            input_csv=round_input_csv,
            min_rounds_per_cell=min_rounds_per_cell,
            baseline_policy_model=non_naive_policy_model,
            naive_policy_model=str(args.naive_policy_model),
            tolerance=tolerance,
        )
        if not rows:
            raise ValueError(
                "No eligible matched policy rows were found after initial-belief "
                "matching."
            )

    penalty_rows, overall_pairs = _compute_penalty_rows(
        rows,
        baseline_policy_model=non_naive_policy_model,
        naive_policy_model=str(args.naive_policy_model),
        bootstrap_draws=bootstrap_draws,
        rng=rng,
    )
    overall = _overall_row(
        penalty_rows,
        overall_pairs=overall_pairs,
        bootstrap_draws=bootstrap_draws,
        rng=rng,
    )
    output_rows = list(penalty_rows)
    if overall is not None:
        output_rows.append(overall)
    if not penalty_rows:
        raise ValueError(
            "No naive-vs-non-naive proposition/stance cells were shared across corpora."
        )

    _write_output_csv(
        output_csv,
        output_rows,
        non_naive_policy_model=non_naive_policy_model,
        naive_policy_model=str(args.naive_policy_model),
    )
    _render_plot(output_pdf, output_rows)
    print(f"Wrote naive-penalty CSV: {output_csv}")
    print(f"Wrote naive-penalty figure: {output_pdf}")


if __name__ == "__main__":
    main()
