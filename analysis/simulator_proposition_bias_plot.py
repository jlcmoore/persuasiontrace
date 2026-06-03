"""Render robust proposition susceptibility spreads from movement summaries."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from .simulator_plot_cli import add_initial_belief_match_tolerance_argument
from .simulator_plot_style import (
    COMPARISON_CORPUS_COLOR_MAP,
    CORE_COMPARISON_CORPUS_ORDER,
    PAPER_RESULTS_FIGURE_SIZE_INCHES,
    comparison_corpus_tick_label,
)
from .utils import resolve_repo_path, safe_float_or_nan, safe_int_or_none

DEFAULT_INPUT_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_proposition_stance_deltas.csv"
)
DEFAULT_ROUND_DYNAMICS_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_round_dynamics.csv"
)
DEFAULT_OUTPUT_PDF = Path("analysis/figures/results_proposition_bias.pdf")
DEFAULT_MIN_ROUNDS = 2
DEFAULT_BOOTSTRAP_SAMPLES = 4000
DEFAULT_SEED = 17

CORPUS_ORDER = [
    "vanilla_llm_target",
    "structure_target",
    "full_simulated_target",
]
CORPUS_COLORS = dict(COMPARISON_CORPUS_COLOR_MAP)

# Use a non-interactive backend for CLI stability in headless environments.
plt.switch_backend("Agg")


@dataclass(frozen=True)
class PropositionDeltaRow:
    """Store one proposition-level movement summary row.

    Attributes:
        corpus: Corpus key.
        proposition: Proposition text.
        n_rounds: Number of rounds aggregated for this proposition row.
        mean_total_delta: Mean total target movement in persuader direction.
    """

    corpus: str
    proposition: str
    n_rounds: int
    mean_total_delta: float


@dataclass(frozen=True)
class CorpusRangeRow:
    """Store per-corpus proposition susceptibility spread statistics.

    Attributes:
        corpus: Corpus key.
        p10_delta: 10th percentile of collapsed proposition means.
        p90_delta: 90th percentile of collapsed proposition means.
        spread_delta: Difference ``p90_delta - p10_delta``.
        spread_ci_low: Lower 95% bootstrap confidence bound for spread.
        spread_ci_high: Upper 95% bootstrap confidence bound for spread.
        total_rounds: Total rounds contributing to collapsed proposition means.
    """

    corpus: str
    p10_delta: float
    p90_delta: float
    spread_delta: float
    spread_ci_low: float
    spread_ci_high: float
    total_rounds: int


@dataclass(frozen=True)
class RoundDeltaRow:
    """Store one round-level movement row used for initial-belief controls.

    Attributes:
        corpus: Corpus key.
        proposition: Proposition text.
        initial_belief: Initial target belief.
        total_delta: Signed total movement in persuader direction.
    """

    corpus: str
    proposition: str
    initial_belief: float
    total_delta: float


class _CorpusPropositionRow(Protocol):
    """Protocol for rows that expose ``corpus`` and ``proposition`` fields."""

    corpus: str
    proposition: str


RowT = TypeVar("RowT", bound=_CorpusPropositionRow)


def _group_by_corpus_proposition(rows: list[RowT]) -> dict[tuple[str, str], list[RowT]]:
    """Group row objects by ``(corpus, proposition)``.

    Args:
        rows: Row objects with corpus and proposition attributes.

    Returns:
        Mapping from ``(corpus, proposition)`` to row lists.
    """
    grouped: dict[tuple[str, str], list[RowT]] = {}
    for row in rows:
        grouped.setdefault((row.corpus, row.proposition), []).append(row)
    return grouped


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for proposition susceptibility spread plotting.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot robust proposition susceptibility spreads by corpus, "
            "collapsing across persuader stance."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input *_proposition_stance_deltas.csv file.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path for the proposition-bias figure.",
    )
    parser.add_argument(
        "--round-dynamics-csv",
        type=Path,
        default=DEFAULT_ROUND_DYNAMICS_CSV,
        help="Round-level dynamics CSV used for initial-belief equalization.",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=DEFAULT_MIN_ROUNDS,
        help="Require n_rounds >= this threshold for each proposition/stance row.",
    )
    add_initial_belief_match_tolerance_argument(
        parser,
        help_text=(
            "Optional initial-belief equalization tolerance. When set (for "
            "example 0.05), propositions are compared after reweighting to a "
            "shared uniform distribution over initial-belief bins of width "
            "2*tolerance."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples used for 95% spread confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for bootstrap resampling.",
    )
    return parser.parse_args()


def _load_proposition_rows(
    *,
    input_csv: Path,
    min_rounds: int,
) -> list[PropositionDeltaRow]:
    """Load proposition movement rows from CSV.

    Args:
        input_csv: Input delta CSV path.
        min_rounds: Minimum per-row round count required.

    Returns:
        Filtered proposition-level rows.
    """
    rows: list[PropositionDeltaRow] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            corpus = str(row.get("corpus") or "")
            proposition = str(row.get("proposition") or "").strip()
            if corpus not in CORPUS_ORDER:
                continue
            if not proposition:
                continue

            n_rounds = safe_int_or_none(row.get("n_rounds"))
            if n_rounds is None or n_rounds < min_rounds:
                continue

            value = safe_float_or_nan(row.get("mean_total_delta"))
            if not np.isfinite(value):
                continue

            rows.append(
                PropositionDeltaRow(
                    corpus=corpus,
                    proposition=proposition,
                    n_rounds=int(n_rounds),
                    mean_total_delta=float(value),
                )
            )
    return rows


def _load_round_rows(
    *,
    input_csv: Path,
    allowed_propositions: dict[str, set[str]] | None = None,
) -> list[RoundDeltaRow]:
    """Load round-level rows used by initial-belief equalization mode.

    Args:
        input_csv: Round-dynamics CSV path.
        allowed_propositions: Optional corpus-specific proposition allowlist.

    Returns:
        Filtered round-level rows.
    """
    rows: list[RoundDeltaRow] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            corpus = str(row.get("corpus") or "").strip()
            proposition = str(row.get("proposition") or "").strip()
            if corpus not in CORPUS_ORDER or not proposition:
                continue
            if (
                allowed_propositions is not None
                and proposition not in allowed_propositions.get(corpus, set())
            ):
                continue
            initial_belief = safe_float_or_nan(row.get("initial_belief"))
            total_delta = safe_float_or_nan(row.get("total_delta"))
            if not np.isfinite(initial_belief) or not np.isfinite(total_delta):
                continue
            rows.append(
                RoundDeltaRow(
                    corpus=corpus,
                    proposition=proposition,
                    initial_belief=float(initial_belief),
                    total_delta=float(total_delta),
                )
            )
    return rows


def _belief_bin_index(*, initial_belief: float, bin_width: float) -> int:
    """Convert initial belief into a stable integer bin index.

    Args:
        initial_belief: Initial target belief.
        bin_width: Belief bin width.

    Returns:
        Integer bin index.
    """
    clipped = float(min(max(initial_belief, 0.0), 1.0 - 1e-9))
    return int(clipped / bin_width)


def _collapse_with_initial_belief_equalization(
    *,
    rows: list[RoundDeltaRow],
    min_rounds: int,
    tolerance: float,
) -> dict[str, list[tuple[float, int]]]:
    """Collapse proposition means under shared initial-belief distributions.

    Args:
        rows: Round-level rows.
        min_rounds: Minimum rounds required per proposition.
        tolerance: Matching tolerance; bin width is ``2*tolerance``.

    Returns:
        Mapping from corpus key to equalized proposition means and effective rounds.
    """
    bin_width = float(tolerance * 2.0)
    grouped = _group_by_corpus_proposition(rows)

    by_corpus: dict[str, list[tuple[float, int]]] = {}
    for corpus in CORPUS_ORDER:
        prop_groups = {
            proposition: entries
            for (entry_corpus, proposition), entries in grouped.items()
            if entry_corpus == corpus and len(entries) >= min_rounds
        }
        if not prop_groups:
            continue

        bins_by_prop: dict[str, dict[int, list[float]]] = {}
        for proposition, entries in prop_groups.items():
            by_bin: dict[int, list[float]] = {}
            for entry in entries:
                index = _belief_bin_index(
                    initial_belief=float(entry.initial_belief),
                    bin_width=bin_width,
                )
                by_bin.setdefault(index, []).append(float(entry.total_delta))
            bins_by_prop[proposition] = by_bin

        shared_bins = set.intersection(
            *(set(by_bin.keys()) for by_bin in bins_by_prop.values())
        )
        if not shared_bins:
            continue
        ordered_shared_bins = sorted(shared_bins)

        collapsed_entries: list[tuple[float, int]] = []
        for proposition, by_bin in bins_by_prop.items():
            bin_means = np.asarray(
                [
                    float(np.mean(np.asarray(by_bin[index], dtype=float)))
                    for index in ordered_shared_bins
                ],
                dtype=float,
            )
            effective_rounds = int(
                sum(len(by_bin[index]) for index in ordered_shared_bins)
            )
            if bin_means.size == 0 or effective_rounds <= 0:
                continue
            collapsed_entries.append((float(np.mean(bin_means)), effective_rounds))
        if collapsed_entries:
            by_corpus[corpus] = collapsed_entries
    return by_corpus


def _collapse_across_stance(
    rows: list[PropositionDeltaRow],
) -> dict[str, list[tuple[float, int]]]:
    """Collapse proposition rows across stance using round-count weighting.

    Args:
        rows: Proposition rows that may include both stances.

    Returns:
        Mapping from corpus key to collapsed proposition means.
    """
    grouped = _group_by_corpus_proposition(rows)

    by_corpus: dict[str, list[tuple[float, int]]] = {}
    for (corpus, _proposition), prop_rows in grouped.items():
        total_weight = float(sum(max(1, item.n_rounds) for item in prop_rows))
        if total_weight <= 0:
            continue
        total_rounds = int(sum(max(1, item.n_rounds) for item in prop_rows))
        weighted_mean = float(
            sum(item.mean_total_delta * max(1, item.n_rounds) for item in prop_rows)
            / total_weight
        )
        by_corpus.setdefault(corpus, []).append((weighted_mean, total_rounds))
    return by_corpus


def _corpus_range_rows(
    by_corpus: dict[str, list[tuple[float, int]]],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[CorpusRangeRow]:
    """Compute robust percentile spread summaries for each corpus.

    Args:
        by_corpus: Collapsed proposition means by corpus.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
        rng: Shared random generator for bootstrap draws.

    Returns:
        Ordered corpus range rows.
    """
    out: list[CorpusRangeRow] = []
    for corpus in CORPUS_ORDER:
        entries = by_corpus.get(corpus, [])
        values = np.asarray([value for value, _ in entries], dtype=float)
        if values.size == 0:
            continue
        p10_delta = float(np.quantile(values, 0.10))
        p90_delta = float(np.quantile(values, 0.90))
        spread_delta = float(p90_delta - p10_delta)
        spread_ci_low, spread_ci_high = _bootstrap_spread_ci(
            values,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        total_rounds = int(sum(round_count for _, round_count in entries))
        out.append(
            CorpusRangeRow(
                corpus=corpus,
                p10_delta=p10_delta,
                p90_delta=p90_delta,
                spread_delta=spread_delta,
                spread_ci_low=spread_ci_low,
                spread_ci_high=spread_ci_high,
                total_rounds=total_rounds,
            )
        )
    return out


def _bootstrap_spread_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for proposition spread.

    Args:
        values: Per-proposition collapsed deltas for one corpus.
        n_bootstrap: Number of bootstrap resamples.
        rng: Random generator for reproducibility.

    Returns:
        Tuple ``(ci_low, ci_high)`` for the spread statistic.
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float("nan"), float("nan")

    point = float(np.quantile(finite_values, 0.90) - np.quantile(finite_values, 0.10))
    if int(n_bootstrap) <= 0 or finite_values.size == 1:
        return point, point

    draws = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sample = rng.choice(
            finite_values,
            size=finite_values.size,
            replace=True,
        )
        draws[index] = float(np.quantile(sample, 0.90) - np.quantile(sample, 0.10))

    centered_draws = draws - point
    ci_low, ci_high = point + np.quantile(centered_draws, [0.025, 0.975])
    ci_low = float(min(ci_low, point))
    ci_high = float(max(ci_high, point))
    return ci_low, ci_high


def _intervals_totally_overlap(
    ci_lows: np.ndarray,
    ci_highs: np.ndarray,
) -> bool:
    """Check whether all confidence intervals share one common overlap region.

    Args:
        ci_lows: Array of lower confidence bounds.
        ci_highs: Array of upper confidence bounds.

    Returns:
        ``True`` when all finite intervals overlap at one common value.
    """
    finite = np.isfinite(ci_lows) & np.isfinite(ci_highs) & (ci_highs >= ci_lows)
    if int(np.count_nonzero(finite)) < 2:
        return False
    return bool(float(np.max(ci_lows[finite])) <= float(np.min(ci_highs[finite])))


def _render_plot(*, range_rows: list[CorpusRangeRow], output_pdf: Path) -> None:
    """Render the proposition susceptibility-spread figure.

    Args:
        range_rows: Per-corpus range summary rows.
        output_pdf: Output PDF path.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    corpus_rank = {
        corpus: index for index, corpus in enumerate(CORE_COMPARISON_CORPUS_ORDER)
    }
    ranked_rows = sorted(
        range_rows,
        key=lambda row: (
            float(row.spread_delta),
            corpus_rank.get(str(row.corpus), len(corpus_rank)),
        ),
    )

    fig, axis = plt.subplots(figsize=PAPER_RESULTS_FIGURE_SIZE_INCHES)
    x_positions = np.arange(len(ranked_rows), dtype=float)
    labels = [comparison_corpus_tick_label(row.corpus) for row in ranked_rows]
    spreads = np.asarray([float(row.spread_delta) for row in ranked_rows], dtype=float)
    spread_ci_lows = np.asarray(
        [float(row.spread_ci_low) for row in ranked_rows],
        dtype=float,
    )
    spread_ci_highs = np.asarray(
        [float(row.spread_ci_high) for row in ranked_rows],
        dtype=float,
    )
    colors = [CORPUS_COLORS.get(row.corpus, "#888888") for row in ranked_rows]

    bars = axis.bar(
        x_positions,
        spreads,
        color=colors,
        width=0.62,
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.92,
        zorder=2,
    )

    show_ci = not _intervals_totally_overlap(spread_ci_lows, spread_ci_highs)
    max_spread = float(np.max(spreads)) if spreads.size else 0.0
    max_ci = (
        float(np.nanmax(spread_ci_highs))
        if show_ci and np.any(np.isfinite(spread_ci_highs))
        else max_spread
    )
    y_top = max(0.08, max(max_spread, max_ci) * 1.25)
    axis.set_ylim(0.0, y_top)
    if show_ci:
        lower_err = np.maximum(0.0, spreads - spread_ci_lows)
        upper_err = np.maximum(0.0, spread_ci_highs - spreads)
        axis.errorbar(
            x_positions,
            spreads,
            yerr=np.vstack([lower_err, upper_err]),
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=3,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.tick_params(axis="x", labelsize=6.8, pad=1.2)
    axis.grid(axis="y", linestyle=":", alpha=0.24, zorder=0)
    axis.set_ylabel(r"Spread ($\leftarrow$)", fontsize=10)
    axis.margins(x=0.06)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#444444")
    fig.subplots_adjust(left=0.26, bottom=0.18, right=0.98, top=0.96)
    fig.savefig(output_pdf, dpi=220)
    plt.close(fig)


def main() -> None:
    """Execute proposition susceptibility spread plotting workflow."""
    args = parse_args()
    reference_file = Path(__file__).resolve()
    input_csv = resolve_repo_path(args.input_csv, reference_file=reference_file)
    round_dynamics_csv = resolve_repo_path(
        args.round_dynamics_csv,
        reference_file=reference_file,
    )
    output_pdf = resolve_repo_path(args.output_pdf, reference_file=reference_file)
    min_rounds = int(args.min_rounds)
    tolerance_raw = args.initial_belief_match_tolerance

    if tolerance_raw is None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        rows = _load_proposition_rows(
            input_csv=input_csv,
            min_rounds=min_rounds,
        )
        if not rows:
            raise ValueError(
                "No proposition rows passed filters; no figure was generated."
            )
        by_corpus = _collapse_across_stance(rows)
    else:
        tolerance = float(tolerance_raw)
        if tolerance <= 0.0:
            raise ValueError("initial_belief_match_tolerance must be > 0.")
        if not round_dynamics_csv.exists():
            raise FileNotFoundError(
                f"Round-dynamics CSV not found: {round_dynamics_csv}"
            )
        allowed_propositions: dict[str, set[str]] | None = None
        if input_csv.exists():
            baseline_props = _load_proposition_rows(
                input_csv=input_csv,
                min_rounds=min_rounds,
            )
            allowed_propositions = {}
            for row in baseline_props:
                allowed_propositions.setdefault(row.corpus, set()).add(row.proposition)
        round_rows = _load_round_rows(
            input_csv=round_dynamics_csv,
            allowed_propositions=allowed_propositions,
        )
        if not round_rows:
            raise ValueError(
                "No round-level rows passed filters; no figure was generated."
            )
        by_corpus = _collapse_with_initial_belief_equalization(
            rows=round_rows,
            min_rounds=min_rounds,
            tolerance=tolerance,
        )

    bootstrap_samples = max(0, int(args.bootstrap_samples))
    rng = np.random.default_rng(int(args.seed))
    range_rows = _corpus_range_rows(
        by_corpus,
        n_bootstrap=bootstrap_samples,
        rng=rng,
    )
    if not range_rows:
        raise ValueError(
            "No corpus rows remained after proposition collapse/equalization."
        )

    _render_plot(range_rows=range_rows, output_pdf=output_pdf)
    print(f"Wrote proposition-bias figure: {output_pdf}")


if __name__ == "__main__":
    main()
