"""Export paper-ready qualitative examples for susceptibility and naive analyses.

This script automatically selects two qualitative examples:
1) A proposition pair within one simulator corpus where movement differs the most
   under similar initial beliefs (susceptibility story).
2) A proposition/stance cell where naive persuasion most exceeds non-naive
   persuasion for a selected corpus under matched initial beliefs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulation.human_likeness_eval.corpus_variants import _split_policy_variant_corpus

from .utils import resolve_repo_path, safe_float_or_nan

DEFAULT_ROUND_DYNAMICS_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_round_dynamics.csv"
)
DEFAULT_ROUND_DYNAMICS_BY_POLICY_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_round_dynamics_by_policy.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "analysis/data/rl_human_match_sim_compare_qualitative_examples.csv"
)
DEFAULT_OUTPUT_MD = Path(
    "analysis/data/rl_human_match_sim_compare_qualitative_examples.md"
)
DEFAULT_SUSCEPTIBILITY_CORPUS = "vanilla_llm_target"
DEFAULT_NAIVE_CORPUS = "vanilla_llm_target"
DEFAULT_NON_NAIVE_POLICY_MODEL = "gpt-5-2025-08-07"
DEFAULT_NAIVE_POLICY_MODEL = "naive"
DEFAULT_INITIAL_BELIEF_TOLERANCE = 0.05


@dataclass(frozen=True)
class RoundMovementRow:
    """One round-level row for corpus-only movement analysis.

    Attributes:
        corpus: Corpus key.
        proposition: Proposition text.
        initial_belief: Initial belief in [0, 1].
        final_belief: Final belief in [0, 1].
        total_delta: Signed total movement in persuader direction.
    """

    corpus: str
    proposition: str
    initial_belief: float
    final_belief: float
    total_delta: float


@dataclass(frozen=True)
class PolicyRoundMovementRow:
    """One round-level row for by-policy movement analysis.

    Attributes:
        base_corpus: Base corpus key without policy suffix.
        policy_model: Policy model id.
        proposition: Proposition text.
        stance: Persuader stance.
        initial_belief: Initial belief in [0, 1].
        final_belief: Final belief in [0, 1].
        abs_total_delta: Absolute total movement.
    """

    base_corpus: str
    policy_model: str
    proposition: str
    stance: str
    initial_belief: float
    final_belief: float
    abs_total_delta: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for qualitative-example export.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export one susceptibility proposition-pair example and one "
            "naive-vs-non-naive bad-case example."
        )
    )
    parser.add_argument(
        "--round-dynamics-csv",
        type=Path,
        default=DEFAULT_ROUND_DYNAMICS_CSV,
        help="Round-level movement CSV (without policy suffixes).",
    )
    parser.add_argument(
        "--round-dynamics-by-policy-csv",
        type=Path,
        default=DEFAULT_ROUND_DYNAMICS_BY_POLICY_CSV,
        help="Round-level movement CSV with policy-suffixed corpus keys.",
    )
    parser.add_argument(
        "--susceptibility-corpus",
        type=str,
        default=DEFAULT_SUSCEPTIBILITY_CORPUS,
        help="Corpus used for the susceptibility proposition-pair example.",
    )
    parser.add_argument(
        "--naive-corpus",
        type=str,
        default=DEFAULT_NAIVE_CORPUS,
        help="Base corpus used for the naive bad-case example.",
    )
    parser.add_argument(
        "--non-naive-policy-model",
        type=str,
        default=DEFAULT_NON_NAIVE_POLICY_MODEL,
        help="Policy model id used as the non-naive baseline.",
    )
    parser.add_argument(
        "--naive-policy-model",
        type=str,
        default=DEFAULT_NAIVE_POLICY_MODEL,
        help="Policy model id used as naive baseline.",
    )
    parser.add_argument(
        "--initial-belief-match-tolerance",
        type=float,
        default=DEFAULT_INITIAL_BELIEF_TOLERANCE,
        help=(
            "Maximum initial-belief distance used for matching; susceptibility "
            "uses bins of width 2*tolerance."
        ),
    )
    parser.add_argument(
        "--min-rounds-per-proposition",
        type=int,
        default=2,
        help="Minimum rounds required for a proposition within one belief bin.",
    )
    parser.add_argument(
        "--min-rounds-per-naive-cell",
        type=int,
        default=1,
        help="Minimum rounds required per policy before naive matching.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for qualitative examples.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Output Markdown path for qualitative examples.",
    )
    return parser.parse_args()


def _belief_bin_index(*, initial_belief: float, bin_width: float) -> int:
    """Map initial belief into a stable bin index.

    Args:
        initial_belief: Initial belief in [0, 1].
        bin_width: Width of one bin.

    Returns:
        Integer bin index.
    """
    clipped = float(min(max(initial_belief, 0.0), 1.0 - 1e-9))
    return int(clipped / bin_width)


def _load_round_rows(path: Path) -> list[RoundMovementRow]:
    """Load round-level rows for susceptibility example selection.

    Args:
        path: Round-dynamics CSV path.

    Returns:
        Parsed round-level rows.
    """
    rows: list[RoundMovementRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            corpus = str(raw.get("corpus") or "").strip()
            proposition = str(raw.get("proposition") or "").strip()
            initial_belief = safe_float_or_nan(raw.get("initial_belief"))
            final_belief = safe_float_or_nan(raw.get("final_belief"))
            total_delta = safe_float_or_nan(raw.get("total_delta"))
            if not corpus or not proposition:
                continue
            if (
                not np.isfinite(initial_belief)
                or not np.isfinite(final_belief)
                or not np.isfinite(total_delta)
            ):
                continue
            rows.append(
                RoundMovementRow(
                    corpus=corpus,
                    proposition=proposition,
                    initial_belief=float(initial_belief),
                    final_belief=float(final_belief),
                    total_delta=float(total_delta),
                )
            )
    return rows


def _load_policy_round_rows(path: Path) -> list[PolicyRoundMovementRow]:
    """Load by-policy round-level rows for naive example selection.

    Args:
        path: Round-dynamics by-policy CSV path.

    Returns:
        Parsed by-policy round rows.
    """
    rows: list[PolicyRoundMovementRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            corpus = str(raw.get("corpus") or "").strip()
            base_corpus, policy_model = _split_policy_variant_corpus(corpus)
            if policy_model is None or base_corpus == "human_reference":
                continue
            proposition = str(raw.get("proposition") or "").strip()
            stance = str(raw.get("stance") or "").strip()
            initial_belief = safe_float_or_nan(raw.get("initial_belief"))
            final_belief = safe_float_or_nan(raw.get("final_belief"))
            abs_total_delta = safe_float_or_nan(raw.get("abs_total_delta"))
            if not proposition or not stance:
                continue
            if (
                not np.isfinite(initial_belief)
                or not np.isfinite(final_belief)
                or not np.isfinite(abs_total_delta)
            ):
                continue
            rows.append(
                PolicyRoundMovementRow(
                    base_corpus=base_corpus,
                    policy_model=policy_model,
                    proposition=proposition,
                    stance=stance,
                    initial_belief=float(initial_belief),
                    final_belief=float(final_belief),
                    abs_total_delta=float(abs_total_delta),
                )
            )
    return rows


def _best_susceptibility_example(
    *,
    rows: list[RoundMovementRow],
    corpus: str,
    tolerance: float,
    min_rounds_per_prop: int,
) -> dict[str, object] | None:
    """Select the strongest proposition-pair susceptibility example.

    Args:
        rows: Round-level movement rows.
        corpus: Corpus key to analyze.
        tolerance: Belief matching tolerance.
        min_rounds_per_prop: Minimum rounds per proposition/bin.

    Returns:
        Example dictionary or None when unavailable.
    """
    if tolerance <= 0.0:
        return None
    bin_width = float(tolerance * 2.0)
    grouped: dict[tuple[int, str], list[RoundMovementRow]] = {}
    for row in rows:
        if row.corpus != corpus:
            continue
        bin_index = _belief_bin_index(
            initial_belief=float(row.initial_belief),
            bin_width=bin_width,
        )
        grouped.setdefault((bin_index, row.proposition), []).append(row)

    bin_indices = sorted({index for index, _ in grouped})
    best: dict[str, object] | None = None
    best_gap = float("-inf")
    best_support = -1
    for bin_index in bin_indices:
        prop_stats: list[tuple[str, float, int, float, float]] = []
        for (current_index, proposition), entries in grouped.items():
            if current_index != bin_index:
                continue
            if len(entries) < min_rounds_per_prop:
                continue
            mean_delta = float(
                np.mean(
                    np.asarray([entry.total_delta for entry in entries], dtype=float)
                )
            )
            mean_initial = float(
                np.mean(
                    np.asarray([entry.initial_belief for entry in entries], dtype=float)
                )
            )
            mean_final = float(
                np.mean(
                    np.asarray([entry.final_belief for entry in entries], dtype=float)
                )
            )
            prop_stats.append(
                (proposition, mean_delta, len(entries), mean_initial, mean_final)
            )
        if len(prop_stats) < 2:
            continue
        prop_stats.sort(key=lambda item: item[1])
        low_prop, low_mean, low_count, low_initial, low_final = prop_stats[0]
        high_prop, high_mean, high_count, high_initial, high_final = prop_stats[-1]
        gap_value = float(high_mean - low_mean)
        support_count = int(low_count + high_count)
        if gap_value > best_gap or (
            np.isclose(gap_value, best_gap) and support_count > best_support
        ):
            bin_low = float(bin_index * bin_width)
            bin_high = float(min(1.0, bin_low + bin_width))
            best_gap = gap_value
            best_support = support_count
            best = {
                "example_type": "susceptibility_prop_pair",
                "corpus": corpus,
                "initial_belief_bin_low": bin_low,
                "initial_belief_bin_high": bin_high,
                "proposition_a": high_prop,
                "proposition_b": low_prop,
                "mean_initial_belief_a": high_initial,
                "mean_initial_belief_b": low_initial,
                "mean_final_belief_a": high_final,
                "mean_final_belief_b": low_final,
                "mean_total_delta_a": high_mean,
                "mean_total_delta_b": low_mean,
                "delta_gap_signed": float(high_mean - low_mean),
                "delta_gap_abs": abs(float(high_mean - low_mean)),
                "n_rounds_a": int(high_count),
                "n_rounds_b": int(low_count),
            }
    return best


def _match_policy_rows(
    *,
    baseline_rows: list[PolicyRoundMovementRow],
    naive_rows: list[PolicyRoundMovementRow],
    tolerance: float,
) -> list[tuple[PolicyRoundMovementRow, PolicyRoundMovementRow]]:
    """Create one-to-one initial-belief matches across policies.

    Args:
        baseline_rows: Rows for non-naive policy.
        naive_rows: Rows for naive policy.
        tolerance: Maximum initial-belief distance for a match.

    Returns:
        Matched row pairs.
    """
    if not baseline_rows or not naive_rows:
        return []
    baseline_order = sorted(
        range(len(baseline_rows)),
        key=lambda index: float(baseline_rows[index].initial_belief),
    )
    naive_used = [False] * len(naive_rows)
    matched: list[tuple[PolicyRoundMovementRow, PolicyRoundMovementRow]] = []
    for baseline_index in baseline_order:
        baseline_row = baseline_rows[baseline_index]
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
        matched.append((baseline_row, naive_rows[best_naive_index]))
    return matched


def _best_naive_bad_case(
    *,
    rows: list[PolicyRoundMovementRow],
    base_corpus: str,
    baseline_policy_model: str,
    naive_policy_model: str,
    tolerance: float,
    min_rounds_per_cell: int,
) -> dict[str, object] | None:
    """Select the strongest naive-over-baseline example for one corpus.

    Args:
        rows: By-policy round rows.
        base_corpus: Base corpus key.
        baseline_policy_model: Non-naive policy model id.
        naive_policy_model: Naive policy model id.
        tolerance: Initial-belief matching tolerance.
        min_rounds_per_cell: Minimum rounds per policy before matching.

    Returns:
        Example dictionary or None when unavailable.
    """
    grouped: dict[tuple[str, str, str], list[PolicyRoundMovementRow]] = {}
    for row in rows:
        if row.base_corpus != base_corpus:
            continue
        if row.policy_model not in {baseline_policy_model, naive_policy_model}:
            continue
        grouped.setdefault((row.proposition, row.stance, row.policy_model), []).append(
            row
        )

    cells = sorted(
        {
            (proposition, stance)
            for proposition, stance, policy in grouped
            if policy in {baseline_policy_model, naive_policy_model}
        }
    )
    best: dict[str, object] | None = None
    best_excess = float("-inf")
    best_matches = -1
    for proposition, stance in cells:
        baseline_rows = grouped.get((proposition, stance, baseline_policy_model), [])
        naive_rows = grouped.get((proposition, stance, naive_policy_model), [])
        if (
            len(baseline_rows) < min_rounds_per_cell
            or len(naive_rows) < min_rounds_per_cell
        ):
            continue
        matched_pairs = _match_policy_rows(
            baseline_rows=baseline_rows,
            naive_rows=naive_rows,
            tolerance=tolerance,
        )
        if not matched_pairs:
            continue
        baseline_values = np.asarray(
            [pair[0].abs_total_delta for pair in matched_pairs],
            dtype=float,
        )
        naive_values = np.asarray(
            [pair[1].abs_total_delta for pair in matched_pairs],
            dtype=float,
        )
        baseline_initials = np.asarray(
            [pair[0].initial_belief for pair in matched_pairs],
            dtype=float,
        )
        naive_initials = np.asarray(
            [pair[1].initial_belief for pair in matched_pairs],
            dtype=float,
        )
        baseline_finals = np.asarray(
            [pair[0].final_belief for pair in matched_pairs],
            dtype=float,
        )
        naive_finals = np.asarray(
            [pair[1].final_belief for pair in matched_pairs],
            dtype=float,
        )
        baseline_mean = float(np.mean(baseline_values))
        naive_mean = float(np.mean(naive_values))
        excess = float(naive_mean - baseline_mean)
        representative_index = int(np.argmax(naive_values - baseline_values))
        representative_non_naive = matched_pairs[representative_index][0]
        representative_naive = matched_pairs[representative_index][1]
        denominator = max(abs(baseline_mean), 1e-9)
        relative_percent = float((excess / denominator) * 100.0)
        matched_count = len(matched_pairs)
        if excess > best_excess or (
            np.isclose(excess, best_excess) and matched_count > best_matches
        ):
            best_excess = excess
            best_matches = matched_count
            best = {
                "example_type": "naive_bad_case",
                "corpus": base_corpus,
                "proposition": proposition,
                "stance": stance,
                "matched_count": matched_count,
                "baseline_mean_abs_total_delta": baseline_mean,
                "naive_mean_abs_total_delta": naive_mean,
                "naive_excess_abs_total_delta": excess,
                "naive_excess_percent_of_baseline": relative_percent,
                "baseline_mean_initial_belief": float(np.mean(baseline_initials)),
                "naive_mean_initial_belief": float(np.mean(naive_initials)),
                "baseline_mean_final_belief": float(np.mean(baseline_finals)),
                "naive_mean_final_belief": float(np.mean(naive_finals)),
                "representative_initial_belief_non_naive": float(
                    representative_non_naive.initial_belief
                ),
                "representative_initial_belief_naive": float(
                    representative_naive.initial_belief
                ),
                "representative_final_belief_non_naive": float(
                    representative_non_naive.final_belief
                ),
                "representative_final_belief_naive": float(
                    representative_naive.final_belief
                ),
                "representative_abs_total_delta_non_naive": float(
                    representative_non_naive.abs_total_delta
                ),
                "representative_abs_total_delta_naive": float(
                    representative_naive.abs_total_delta
                ),
                "representative_naive_excess_abs_total_delta": float(
                    representative_naive.abs_total_delta
                    - representative_non_naive.abs_total_delta
                ),
            }
    return best


def _blank_row(example_type: str) -> dict[str, object]:
    """Create a placeholder row when an example is unavailable.

    Args:
        example_type: Example category label.

    Returns:
        Placeholder row dictionary.
    """
    return {
        "example_type": example_type,
        "status": "not_found",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write qualitative examples to CSV.

    Args:
        path: Output CSV path.
        rows: Row dictionaries to write.
    """
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(rows: list[dict[str, object]]) -> str:
    """Render human-readable markdown for qualitative examples.

    Args:
        rows: Qualitative example rows.

    Returns:
        Markdown content.
    """
    by_type = {str(row.get("example_type") or ""): row for row in rows}
    susceptibility = by_type.get("susceptibility_prop_pair", {})
    naive = by_type.get("naive_bad_case", {})

    lines = ["# Qualitative Examples", ""]
    lines.append("## Susceptibility Proposition Pair")
    if str(susceptibility.get("status") or "") == "not_found":
        lines.append("- No eligible proposition pair found.")
    else:
        lines.append(f"- Corpus: {susceptibility.get('corpus', '')}")
        lines.append(
            "- Initial-belief bin: "
            f"[{float(susceptibility.get('initial_belief_bin_low', np.nan)):.2f}, "
            f"{float(susceptibility.get('initial_belief_bin_high', np.nan)):.2f}]"
        )
        lines.append(
            f"- High-movement proposition: {susceptibility.get('proposition_a', '')}"
        )
        lines.append(
            f"- Low-movement proposition: {susceptibility.get('proposition_b', '')}"
        )
        lines.append(
            "- Mean total delta (high/low): "
            f"{float(susceptibility.get('mean_total_delta_a', np.nan)):+.4f} / "
            f"{float(susceptibility.get('mean_total_delta_b', np.nan)):+.4f}"
        )
        lines.append(
            "- Mean initial belief (high/low): "
            f"{float(susceptibility.get('mean_initial_belief_a', np.nan)):.4f} / "
            f"{float(susceptibility.get('mean_initial_belief_b', np.nan)):.4f}"
        )
        lines.append(
            "- Mean final belief (high/low): "
            f"{float(susceptibility.get('mean_final_belief_a', np.nan)):.4f} / "
            f"{float(susceptibility.get('mean_final_belief_b', np.nan)):.4f}"
        )
        lines.append(
            "- Rounds (high/low): "
            f"{int(float(susceptibility.get('n_rounds_a', 0.0)))} / "
            f"{int(float(susceptibility.get('n_rounds_b', 0.0)))}"
        )
        lines.append(
            "- Mean total delta gap: "
            f"{float(susceptibility.get('delta_gap_signed', np.nan)):+.4f}"
        )
    lines.append("")
    lines.append("## Naive Bad-Case")
    if str(naive.get("status") or "") == "not_found":
        lines.append("- No eligible naive bad-case found.")
    else:
        lines.append(f"- Corpus: {naive.get('corpus', '')}")
        lines.append(f"- Proposition: {naive.get('proposition', '')}")
        lines.append(f"- Stance: {naive.get('stance', '')}")
        lines.append(
            "- Matched rounds: " f"{int(float(naive.get('matched_count', 0.0)))}"
        )
        lines.append(
            "- Mean abs total delta (non-naive / naive): "
            f"{float(naive.get('baseline_mean_abs_total_delta', np.nan)):.4f} / "
            f"{float(naive.get('naive_mean_abs_total_delta', np.nan)):.4f}"
        )
        lines.append(
            "- Mean initial belief (non-naive / naive): "
            f"{float(naive.get('baseline_mean_initial_belief', np.nan)):.4f} / "
            f"{float(naive.get('naive_mean_initial_belief', np.nan)):.4f}"
        )
        lines.append(
            "- Mean final belief (non-naive / naive): "
            f"{float(naive.get('baseline_mean_final_belief', np.nan)):.4f} / "
            f"{float(naive.get('naive_mean_final_belief', np.nan)):.4f}"
        )
        lines.append(
            "- Naive excess (mean abs total delta): "
            f"{float(naive.get('naive_excess_abs_total_delta', np.nan)):+.4f}"
        )
        lines.append(
            "- Relative to non-naive baseline: "
            f"{float(naive.get('naive_excess_percent_of_baseline', np.nan)):+.1f}%"
        )
    lines.append("")
    return "\n".join(lines)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Write qualitative examples as markdown.

    Args:
        path: Output markdown path.
        rows: Qualitative example rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _render_markdown(rows)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Run qualitative-example selection and export."""
    args = parse_args()
    tolerance = float(args.initial_belief_match_tolerance)
    if tolerance <= 0.0:
        raise ValueError("initial_belief_match_tolerance must be > 0.")

    reference_file = Path(__file__).resolve()
    round_dynamics_csv = resolve_repo_path(
        args.round_dynamics_csv,
        reference_file=reference_file,
    )
    round_dynamics_by_policy_csv = resolve_repo_path(
        args.round_dynamics_by_policy_csv,
        reference_file=reference_file,
    )
    output_csv = resolve_repo_path(args.output_csv, reference_file=reference_file)
    output_md = resolve_repo_path(args.output_md, reference_file=reference_file)

    if not round_dynamics_csv.exists():
        raise FileNotFoundError(f"Round-dynamics CSV not found: {round_dynamics_csv}")
    if not round_dynamics_by_policy_csv.exists():
        raise FileNotFoundError(
            f"Round-dynamics by-policy CSV not found: {round_dynamics_by_policy_csv}"
        )

    susceptibility_rows = _load_round_rows(round_dynamics_csv)
    policy_rows = _load_policy_round_rows(round_dynamics_by_policy_csv)

    susceptibility_example = _best_susceptibility_example(
        rows=susceptibility_rows,
        corpus=str(args.susceptibility_corpus),
        tolerance=tolerance,
        min_rounds_per_prop=max(1, int(args.min_rounds_per_proposition)),
    )
    naive_example = _best_naive_bad_case(
        rows=policy_rows,
        base_corpus=str(args.naive_corpus),
        baseline_policy_model=str(args.non_naive_policy_model),
        naive_policy_model=str(args.naive_policy_model),
        tolerance=tolerance,
        min_rounds_per_cell=max(1, int(args.min_rounds_per_naive_cell)),
    )

    output_rows: list[dict[str, object]] = []
    if susceptibility_example is None:
        output_rows.append(_blank_row("susceptibility_prop_pair"))
    else:
        output_rows.append(susceptibility_example)
    if naive_example is None:
        output_rows.append(_blank_row("naive_bad_case"))
    else:
        output_rows.append(naive_example)

    _write_csv(output_csv, output_rows)
    _write_markdown(output_md, output_rows)
    print(f"Wrote qualitative examples CSV: {output_csv}")
    print(f"Wrote qualitative examples markdown: {output_md}")


if __name__ == "__main__":
    main()
