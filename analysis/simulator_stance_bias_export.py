"""Export stance-bias summaries from mirrored for-vs-against rollout episodes.

This module computes a paired asymmetry metric for each simulator corpus:
the mean absolute gap between stance-relative movement when arguing ``for``
versus ``against`` the same proposition from mirrored initial beliefs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .utils import resolve_repo_path, safe_float_or_nan

DEFAULT_EPISODES_JSONL = Path(
    "results/stance_mirror_dryrun_3sim_allprops_gpt5_fixed/episodes.jsonl"
)
DEFAULT_OUTPUT_PREFIX = Path("analysis/data/rl_human_match_sim_compare")
DEFAULT_BOOTSTRAP_SAMPLES = 4000
DEFAULT_SEED = 17

PAIR_COLUMNS = [
    "corpus",
    "source",
    "split",
    "proposition_id",
    "pair_family",
    "mirror_magnitude",
    "for_delta",
    "against_delta",
    "for_minus_against",
    "abs_for_minus_against",
]
SUMMARY_COLUMNS = [
    "corpus",
    "n_pairs",
    "mean_for_delta",
    "mean_against_delta",
    "mean_for_minus_against",
    "mean_abs_for_minus_against",
    "stance_bias_ci_low",
    "stance_bias_ci_high",
]


StanceEpisode = dict[str, object]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stance-bias export.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export mirrored for-vs-against stance-bias summaries from "
            "baseline-run episodes.jsonl."
        )
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=DEFAULT_EPISODES_JSONL,
        help="Input episodes.jsonl from mirrored-stance rollout run.",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Exact policy model id required for included rows.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix path for generated CSV files.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples used for 95% CI of mean stance bias.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used for bootstrap resampling.",
    )
    return parser.parse_args()


def _corpus_from_row(row: dict[str, object]) -> str | None:
    """Map one episode row into a canonical corpus key.

    Args:
        row: Parsed episode row dictionary.

    Returns:
        Canonical corpus key or ``None`` when the row is not in scope.
    """
    target_backend = str(row.get("target_backend") or "")
    use_structure = bool(row.get("llm_target_use_bayes_structure"))
    no_rhetoric = bool(row.get("simulated_target_no_rhetoric"))

    if target_backend == "llm_target":
        return "structure_target" if use_structure else "vanilla_llm_target"
    if target_backend == "simulated_target" and not no_rhetoric:
        return "full_simulated_target"
    return None


def _pair_family_from_bin(init_belief_bin: str) -> str | None:
    """Map initialization bins to mirrored pairing family keys.

    Args:
        init_belief_bin: Initialization bin label from episodes rows.

    Returns:
        ``mid``, ``very``, or ``None`` when unsupported.
    """
    if init_belief_bin in {"low", "high"}:
        return "mid"
    if init_belief_bin in {"very_low", "very_high"}:
        return "very"
    return None


def _stance_from_bin(init_belief_bin: str) -> str | None:
    """Infer for/against stance from initialization bin.

    Args:
        init_belief_bin: Initialization bin label from episodes rows.

    Returns:
        ``for``, ``against``, or ``None`` when unsupported.
    """
    if init_belief_bin in {"very_low", "low"}:
        return "for"
    if init_belief_bin in {"high", "very_high"}:
        return "against"
    return None


def _load_stance_episodes(
    *,
    episodes_jsonl: Path,
    policy_model: str,
) -> list[StanceEpisode]:
    """Load and normalize stance episodes from one episodes JSONL file.

    Args:
        episodes_jsonl: Input episodes path.
        policy_model: Required policy model id.

    Returns:
        Normalized stance-episode rows.
    """
    rows: list[StanceEpisode] = []
    with episodes_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if str(payload.get("policy_model") or "") != policy_model:
                continue

            corpus = _corpus_from_row(payload)
            if corpus is None:
                continue

            init_belief_bin = str(payload.get("init_belief_bin") or "")
            pair_family = _pair_family_from_bin(init_belief_bin)
            stance = _stance_from_bin(init_belief_bin)
            if pair_family is None or stance is None:
                continue

            initial_belief = safe_float_or_nan(payload.get("target_initial_belief"))
            terminal_delta = safe_float_or_nan(payload.get("terminal_delta"))
            if not np.isfinite(initial_belief) or not np.isfinite(terminal_delta):
                continue
            magnitude = float(min(initial_belief, 1.0 - initial_belief))
            stance_relative_delta = (
                float(terminal_delta)
                if stance == "for"
                else float(-float(terminal_delta))
            )
            rows.append(
                {
                    "corpus": corpus,
                    "source": str(payload.get("source") or ""),
                    "split": str(payload.get("split") or ""),
                    "proposition_id": str(payload.get("proposition_id") or ""),
                    "pair_family": pair_family,
                    "mirror_magnitude": magnitude,
                    "stance": stance,
                    "stance_relative_delta": stance_relative_delta,
                }
            )
    return rows


def _pair_rows(
    stance_rows: list[StanceEpisode],
) -> list[dict[str, object]]:
    """Match for/against rows and compute pair-level asymmetry.

    Args:
        stance_rows: Normalized stance-episode rows.

    Returns:
        Pair-level rows containing stance-delta gaps.
    """
    grouped: dict[tuple[str, str, str, str, str, float], dict[str, float]] = {}
    for row in stance_rows:
        key = (
            str(row["corpus"]),
            str(row["source"]),
            str(row["split"]),
            str(row["proposition_id"]),
            str(row["pair_family"]),
            round(float(row["mirror_magnitude"]), 10),
        )
        grouped.setdefault(key, {})[str(row["stance"])] = float(
            row["stance_relative_delta"]
        )

    pairs: list[dict[str, object]] = []
    for (
        corpus,
        source,
        split,
        proposition_id,
        pair_family,
        mirror_magnitude,
    ), payload in grouped.items():
        if "for" not in payload or "against" not in payload:
            continue
        for_delta = float(payload["for"])
        against_delta = float(payload["against"])
        diff = float(for_delta - against_delta)
        pairs.append(
            {
                "corpus": corpus,
                "source": source,
                "split": split,
                "proposition_id": proposition_id,
                "pair_family": pair_family,
                "mirror_magnitude": float(mirror_magnitude),
                "for_delta": for_delta,
                "against_delta": against_delta,
                "for_minus_against": diff,
                "abs_for_minus_against": float(abs(diff)),
            }
        )
    return pairs


def _mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute percentile-bootstrap CI around a sample mean.

    Args:
        values: One-dimensional numeric sample.
        n_bootstrap: Number of bootstrap resamples.
        rng: Random generator for reproducibility.

    Returns:
        Lower and upper confidence bounds.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    point = float(np.mean(finite))
    if int(n_bootstrap) <= 0 or finite.size == 1:
        return point, point
    draws = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sample = rng.choice(finite, size=finite.size, replace=True)
        draws[index] = float(np.mean(sample))
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    return float(ci_low), float(ci_high)


def _summary_rows(
    *,
    pairs: list[dict[str, object]],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    """Aggregate pair-level rows into corpus stance-bias summaries.

    Args:
        pairs: Pair-level asymmetry rows.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Summary rows, one per corpus.
    """
    by_corpus: dict[str, list[dict[str, object]]] = {}
    for row in pairs:
        corpus = str(row.get("corpus") or "")
        by_corpus.setdefault(corpus, []).append(row)

    rng = np.random.default_rng(int(seed))
    summaries: list[dict[str, object]] = []
    for corpus in sorted(by_corpus):
        corpus_rows = by_corpus[corpus]
        for_values = np.asarray(
            [safe_float_or_nan(row.get("for_delta")) for row in corpus_rows],
            dtype=float,
        )
        against_values = np.asarray(
            [safe_float_or_nan(row.get("against_delta")) for row in corpus_rows],
            dtype=float,
        )
        diff_values = np.asarray(
            [safe_float_or_nan(row.get("for_minus_against")) for row in corpus_rows],
            dtype=float,
        )
        abs_diff_values = np.asarray(
            [
                safe_float_or_nan(row.get("abs_for_minus_against"))
                for row in corpus_rows
            ],
            dtype=float,
        )
        ci_low, ci_high = _mean_ci(
            abs_diff_values,
            n_bootstrap=int(n_bootstrap),
            rng=rng,
        )
        summaries.append(
            {
                "corpus": corpus,
                "n_pairs": int(len(corpus_rows)),
                "mean_for_delta": float(np.mean(for_values)),
                "mean_against_delta": float(np.mean(against_values)),
                "mean_for_minus_against": float(np.mean(diff_values)),
                "mean_abs_for_minus_against": float(np.mean(abs_diff_values)),
                "stance_bias_ci_low": ci_low,
                "stance_bias_ci_high": ci_high,
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Write dictionaries to CSV with one fixed column order.

    Args:
        path: Destination CSV path.
        rows: Row dictionaries.
        columns: Ordered field names for CSV output.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """Execute stance-bias export from episodes JSONL.

    Returns:
        None.
    """
    args = parse_args()
    reference_file = Path(__file__).resolve()
    episodes_jsonl = resolve_repo_path(
        args.episodes_jsonl,
        reference_file=reference_file,
    )
    output_prefix = resolve_repo_path(args.output_prefix, reference_file=reference_file)
    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"Episodes JSONL not found: {episodes_jsonl}")

    stance_rows = _load_stance_episodes(
        episodes_jsonl=episodes_jsonl,
        policy_model=str(args.policy_model),
    )
    if not stance_rows:
        raise ValueError("No stance rows matched the requested filters.")

    pair_rows = _pair_rows(stance_rows)
    if not pair_rows:
        raise ValueError("No mirrored for-vs-against episode pairs were formed.")

    summary_rows = _summary_rows(
        pairs=pair_rows,
        n_bootstrap=max(0, int(args.bootstrap_samples)),
        seed=int(args.seed),
    )
    if not summary_rows:
        raise ValueError("No stance-bias summary rows were generated.")

    pair_csv = output_prefix.with_name(output_prefix.name + "_stance_bias_pairs.csv")
    summary_csv = output_prefix.with_name(
        output_prefix.name + "_stance_bias_summary.csv"
    )
    _write_csv(pair_csv, pair_rows, PAIR_COLUMNS)
    _write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)

    print(f"Exported stance-bias pair rows: {pair_csv}")
    print(f"Exported stance-bias summary rows: {summary_csv}")


if __name__ == "__main__":
    main()
