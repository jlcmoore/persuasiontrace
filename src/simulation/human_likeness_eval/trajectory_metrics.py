"""Trajectory-shape and movement-summary helpers for human-likeness eval."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from simulation.human_likeness import RoundTrajectory


def _flatten_updates(rows: list[RoundTrajectory]) -> np.ndarray:
    """
    Flatten per-round update sequences into one update array.

    Args:
        rows: Trajectory rows.

    Returns:
        1D numpy array of updates.
    """
    updates: list[float] = []
    for row in rows:
        updates.extend(row.updates)
    return np.asarray(updates, dtype=float)


def _updates_by_turn(rows: list[RoundTrajectory]) -> dict[int, np.ndarray]:
    """
    Group updates by turn index.

    Args:
        rows: Trajectory rows.

    Returns:
        Mapping turn index (1-based) -> updates array.
    """
    grouped: dict[int, list[float]] = {}
    for row in rows:
        for idx, value in enumerate(row.updates, start=1):
            grouped.setdefault(idx, []).append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in grouped.items()}


def _flatten_updates_for_turns(
    rows: list[RoundTrajectory],
    *,
    allowed_turns: set[int],
) -> np.ndarray:
    """
    Flatten updates while keeping only selected turn indices.

    Args:
        rows: Trajectory rows.
        allowed_turns: 1-based turn indices to retain.

    Returns:
        1D array of retained updates.
    """
    updates: list[float] = []
    for row in rows:
        for idx, value in enumerate(row.updates, start=1):
            if idx in allowed_turns:
                updates.append(float(value))
    return np.asarray(updates, dtype=float)


def _common_turns_with_min_n(
    *,
    human_rows: list[RoundTrajectory],
    structure_rows: list[RoundTrajectory],
    full_rows: list[RoundTrajectory],
    vanilla_rows: list[RoundTrajectory],
    include_vanilla: bool,
    min_n: int,
) -> set[int]:
    """
    Find turn indices with sufficient samples in all compared corpora.

    Args:
        human_rows: Human trajectories.
        structure_rows: Structure trajectories.
        full_rows: Full trajectories.
        vanilla_rows: Vanilla trajectories.
        include_vanilla: Whether vanilla is part of the comparison set.
        min_n: Minimum required sample count per corpus at a turn.

    Returns:
        Set of 1-based turn indices meeting the minimum count in all corpora.
    """
    if min_n <= 0:
        return set()

    by_turn_human = _updates_by_turn(human_rows)
    by_turn_structure = _updates_by_turn(structure_rows)
    by_turn_full = _updates_by_turn(full_rows)
    by_turn_vanilla = _updates_by_turn(vanilla_rows) if include_vanilla else {}
    candidate_turns = set(by_turn_human) & set(by_turn_structure) & set(by_turn_full)
    if include_vanilla:
        candidate_turns = candidate_turns & set(by_turn_vanilla)

    valid: set[int] = set()
    for turn in sorted(candidate_turns):
        if by_turn_human[turn].size < min_n:
            continue
        if by_turn_structure[turn].size < min_n:
            continue
        if by_turn_full[turn].size < min_n:
            continue
        if include_vanilla and by_turn_vanilla[turn].size < min_n:
            continue
        valid.add(turn)
    return valid


def _histogram_edges_for_arrays(
    *,
    arrays: list[np.ndarray],
    n_bins: int,
) -> np.ndarray:
    """
    Build fixed histogram edges over a set of numeric arrays.

    Args:
        arrays: Arrays to span.
        n_bins: Number of bins.

    Returns:
        Histogram edges with length ``n_bins + 1``.
    """
    valid_arrays = [array for array in arrays if array.size > 0]
    if not valid_arrays:
        return np.linspace(-1.0, 1.0, max(2, int(n_bins)) + 1, dtype=float)
    flat = np.concatenate(valid_arrays)
    lo = float(np.min(flat))
    hi = float(np.max(flat))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.linspace(-1.0, 1.0, max(2, int(n_bins)) + 1, dtype=float)
    if np.isclose(lo, hi):
        pad = max(1e-6, abs(lo) * 1e-3 + 1e-6)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, max(2, int(n_bins)) + 1, dtype=float)


def _pmf_from_histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Convert values to a normalized histogram PMF.

    Args:
        values: Input values.
        edges: Histogram edges.

    Returns:
        Probability mass vector summing to one.
    """
    counts, _ = np.histogram(values, bins=edges)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(max(1, edges.size - 1), dtype=float)
    return counts.astype(float) / total


def _turn_index_jsd(
    *,
    human_rows: list[RoundTrajectory],
    sim_rows: list[RoundTrajectory],
    turns: set[int],
    edges: np.ndarray,
) -> dict[str, object]:
    """
    Compute mean and weighted mean turn-index Jensen-Shannon divergence.

    Args:
        human_rows: Human trajectories.
        sim_rows: Simulator trajectories.
        turns: Turn indices to compare.
        edges: Shared histogram edges.

    Returns:
        Summary dictionary with aggregate and per-turn JSD values.
    """
    by_turn_human = _updates_by_turn(human_rows)
    by_turn_sim = _updates_by_turn(sim_rows)
    per_turn: list[dict[str, object]] = []
    for turn in sorted(turns):
        human_vals = by_turn_human.get(turn)
        sim_vals = by_turn_sim.get(turn)
        if human_vals is None or sim_vals is None:
            continue
        if human_vals.size == 0 or sim_vals.size == 0:
            continue
        p = _pmf_from_histogram(human_vals, edges)
        q = _pmf_from_histogram(sim_vals, edges)
        jsd = float(jensenshannon(p, q, base=2.0) ** 2)
        per_turn.append(
            {
                "turn": int(turn),
                "jsd": jsd,
                "human_n": int(human_vals.size),
                "sim_n": int(sim_vals.size),
            }
        )
    if not per_turn:
        return {}
    values = np.asarray([float(row["jsd"]) for row in per_turn], dtype=float)
    weights = np.asarray([int(row["human_n"]) for row in per_turn], dtype=float)
    weighted = float(np.average(values, weights=weights))
    return {
        "mean": float(np.mean(values)),
        "weighted_mean": weighted,
        "turn_count": int(len(per_turn)),
        "per_turn": per_turn,
    }


def _pooled_jsd_from_arrays(
    *,
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    edges: np.ndarray,
) -> float:
    """
    Compute Jensen-Shannon divergence between two pooled update arrays.

    Args:
        reference_values: Reference update samples.
        candidate_values: Candidate update samples.
        edges: Shared histogram edges.

    Returns:
        Squared Jensen-Shannon divergence in base-2 units.
    """
    if reference_values.size == 0 or candidate_values.size == 0:
        return float("nan")
    p = _pmf_from_histogram(reference_values, edges)
    q = _pmf_from_histogram(candidate_values, edges)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def _pooled_w1(a_rows: list[RoundTrajectory], b_rows: list[RoundTrajectory]) -> float:
    """
    Compute Wasserstein distance on pooled updates.

    Args:
        a_rows: First corpus.
        b_rows: Second corpus.

    Returns:
        Wasserstein-1 distance.
    """
    a = _flatten_updates(a_rows)
    b = _flatten_updates(b_rows)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def _sample_length_matched_updates(
    human_rows: list[RoundTrajectory],
    sim_rows: list[RoundTrajectory],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample simulator updates with round lengths matched to human trajectories.

    Matching is proposition-aware when possible: for each human round, sample
    a simulator round from the same proposition if available, otherwise from
    all simulator rounds. Use only the prefix up to the human round length.

    Args:
        human_rows: Human reference trajectories.
        sim_rows: Simulator trajectories.
        rng: Random generator.

    Returns:
        1D array of matched simulator updates.
    """
    if not human_rows or not sim_rows:
        return np.asarray([], dtype=float)

    sim_by_prop: dict[str, list[RoundTrajectory]] = {}
    for sim_row in sim_rows:
        sim_by_prop.setdefault(sim_row.proposition, []).append(sim_row)

    sampled_updates: list[float] = []
    for human_row in human_rows:
        candidates = sim_by_prop.get(human_row.proposition) or sim_rows
        sampled_row = candidates[int(rng.integers(0, len(candidates)))]
        keep = min(len(human_row.updates), len(sampled_row.updates))
        if keep <= 0:
            continue
        sampled_updates.extend(float(value) for value in sampled_row.updates[:keep])

    return np.asarray(sampled_updates, dtype=float)


def _length_matched_pooled_w1(
    human_rows: list[RoundTrajectory],
    sim_rows: list[RoundTrajectory],
    *,
    n_draws: int,
    seed: int,
) -> dict[str, object]:
    """
    Estimate pooled Wasserstein after length-matching simulator trajectories.

    Args:
        human_rows: Human corpus.
        sim_rows: Simulator corpus.
        n_draws: Number of Monte Carlo draws.
        seed: Random seed.

    Returns:
        Summary dictionary with mean/CI and raw distance draws.
    """
    if n_draws <= 0:
        return {}

    human_updates = _flatten_updates(human_rows)
    if human_updates.size == 0 or not sim_rows:
        return {}

    rng = np.random.default_rng(seed)
    distances: list[float] = []
    for _ in range(n_draws):
        matched_updates = _sample_length_matched_updates(
            human_rows,
            sim_rows,
            rng=rng,
        )
        if matched_updates.size == 0:
            continue
        distances.append(float(wasserstein_distance(human_updates, matched_updates)))

    if not distances:
        return {}

    distribution = np.asarray(distances, dtype=float)
    return {
        "mean": float(np.mean(distribution)),
        "ci_lo": float(np.quantile(distribution, 0.025)),
        "ci_hi": float(np.quantile(distribution, 0.975)),
        "distribution": distribution,
    }


def _prop_weighted_w1(
    human_rows: list[RoundTrajectory],
    sim_rows: list[RoundTrajectory],
    *,
    min_updates_per_prop: int = 2,
) -> tuple[float, float]:
    """
    Compute human-weighted per-proposition Wasserstein distance.

    Args:
        human_rows: Human reference corpus.
        sim_rows: Simulator corpus.
        min_updates_per_prop: Minimum updates required in both corpora.

    Returns:
        Tuple of (distance, human_update_coverage_share).
    """
    human_by_prop: dict[str, list[float]] = {}
    sim_by_prop: dict[str, list[float]] = {}

    for row in human_rows:
        human_by_prop.setdefault(row.proposition, []).extend(row.updates)
    for row in sim_rows:
        sim_by_prop.setdefault(row.proposition, []).extend(row.updates)

    eligible_props: list[str] = []
    for proposition, updates in human_by_prop.items():
        sim_updates = sim_by_prop.get(proposition)
        if sim_updates is None:
            continue
        if len(updates) < min_updates_per_prop:
            continue
        if len(sim_updates) < min_updates_per_prop:
            continue
        eligible_props.append(proposition)

    if not eligible_props:
        return float("nan"), 0.0

    human_total_updates = sum(len(values) for values in human_by_prop.values())
    covered_updates = sum(len(human_by_prop[prop]) for prop in eligible_props)
    coverage_share = (
        float(covered_updates) / float(human_total_updates)
        if human_total_updates > 0
        else 0.0
    )

    weight_total = float(sum(len(human_by_prop[prop]) for prop in eligible_props))
    distance = 0.0
    for proposition in eligible_props:
        weight = float(len(human_by_prop[proposition])) / weight_total
        distance += weight * wasserstein_distance(
            np.asarray(human_by_prop[proposition], dtype=float),
            np.asarray(sim_by_prop[proposition], dtype=float),
        )
    return float(distance), coverage_share


def _corpus_secondary_stats(rows: list[RoundTrajectory]) -> dict[str, float]:
    """
    Compute secondary shape descriptors for one corpus.

    Args:
        rows: Corpus rows.

    Returns:
        Dictionary of scalar summary statistics.
    """
    pooled = _flatten_updates(rows)
    if pooled.size == 0:
        nan = float("nan")
        return {
            "mean_delta": nan,
            "std_delta": nan,
            "toward_persuader_rate": nan,
            "first_mean": nan,
            "rest_mean": nan,
            "first_minus_rest": nan,
        }

    first_updates: list[float] = []
    rest_updates: list[float] = []
    for row in rows:
        if not row.updates:
            continue
        first_updates.append(float(row.updates[0]))
        if len(row.updates) > 1:
            rest_updates.extend(float(value) for value in row.updates[1:])

    first_mean = float(np.mean(first_updates)) if first_updates else float("nan")
    rest_mean = float(np.mean(rest_updates)) if rest_updates else float("nan")
    first_minus_rest = (
        float(first_mean - rest_mean)
        if np.isfinite(first_mean) and np.isfinite(rest_mean)
        else float("nan")
    )
    return {
        "mean_delta": float(np.mean(pooled)),
        "std_delta": float(np.std(pooled, ddof=0)),
        "toward_persuader_rate": float(np.mean(pooled > 0.0)),
        "first_mean": first_mean,
        "rest_mean": rest_mean,
        "first_minus_rest": first_minus_rest,
    }


def _bootstrap_primary(
    human_rows: list[RoundTrajectory],
    structure_rows: list[RoundTrajectory],
    full_rows: list[RoundTrajectory],
    *,
    n_boot: int,
    seed: int,
    bootstrap_statistic_ci_fn: Callable[..., tuple[tuple[float, float], np.ndarray]],
) -> dict[str, float]:
    """
    Bootstrap confidence intervals for primary distances.

    Args:
        human_rows: Human corpus.
        structure_rows: Structure-conditioned simulator corpus.
        full_rows: Full simulator corpus.
        n_boot: Number of bootstrap replicates.
        seed: Random seed.
        bootstrap_statistic_ci_fn: Bootstrap callback used for CI estimation.

    Returns:
        Dictionary with CI bounds and probability that structure is closer.
    """
    if n_boot <= 1:
        return {}
    if not human_rows or not structure_rows or not full_rows:
        return {}

    h_idx = np.arange(len(human_rows), dtype=int)
    s_idx = np.arange(len(structure_rows), dtype=int)
    f_idx = np.arange(len(full_rows), dtype=int)
    rng = np.random.default_rng(seed)

    def _rows_from_indices(
        rows: list[RoundTrajectory],
        sample_idx: np.ndarray,
    ) -> list[RoundTrajectory]:
        return [rows[int(i)] for i in np.asarray(sample_idx, dtype=int).ravel()]

    def _w1_structure(h_sample_idx: np.ndarray, s_sample_idx: np.ndarray) -> float:
        return _pooled_w1(
            _rows_from_indices(human_rows, h_sample_idx),
            _rows_from_indices(structure_rows, s_sample_idx),
        )

    def _w1_full(h_sample_idx: np.ndarray, f_sample_idx: np.ndarray) -> float:
        return _pooled_w1(
            _rows_from_indices(human_rows, h_sample_idx),
            _rows_from_indices(full_rows, f_sample_idx),
        )

    def _w1_diff(
        h_sample_idx: np.ndarray,
        s_sample_idx: np.ndarray,
        f_sample_idx: np.ndarray,
    ) -> float:
        d_structure = _w1_structure(h_sample_idx, s_sample_idx)
        d_full = _w1_full(h_sample_idx, f_sample_idx)
        return float(d_structure - d_full)

    def _bootstrap_ci(
        *,
        data: tuple[np.ndarray, ...],
        statistic: Callable[..., float],
    ) -> tuple[tuple[float, float], np.ndarray]:
        return bootstrap_statistic_ci_fn(
            data=data,
            statistic=statistic,
            n_boot=n_boot,
            confidence_level=0.95,
            rng=rng,
        )

    try:
        structure_ci, _ = _bootstrap_ci(
            data=(h_idx, s_idx),
            statistic=_w1_structure,
        )
        full_ci, _ = _bootstrap_ci(
            data=(h_idx, f_idx),
            statistic=_w1_full,
        )
        diff_ci, diff_distribution = _bootstrap_ci(
            data=(h_idx, s_idx, f_idx),
            statistic=_w1_diff,
        )
    except (RuntimeError, ValueError):
        return {}
    if diff_distribution.size == 0:
        return {}

    return {
        "structure_ci_lo": structure_ci[0],
        "structure_ci_hi": structure_ci[1],
        "full_ci_lo": full_ci[0],
        "full_ci_hi": full_ci[1],
        "diff_ci_lo": diff_ci[0],
        "diff_ci_hi": diff_ci[1],
        "p_structure_closer": float(np.mean(diff_distribution < 0.0)),
    }


def _corpus_summary_row(name: str, rows: list[RoundTrajectory]) -> dict[str, object]:
    """
    Build one summary row for a corpus.

    Args:
        name: Corpus name.
        rows: Corpus rows.

    Returns:
        Summary row dictionary.
    """
    updates = int(sum(len(row.updates) for row in rows))
    mean_updates = float(updates / len(rows)) if rows else float("nan")
    return {
        "corpus": name,
        "rounds": len(rows),
        "updates": updates,
        "unique_props": len({row.proposition for row in rows}),
        "mean_updates_per_round": mean_updates,
    }


def _belief_trajectory_values(row: RoundTrajectory) -> np.ndarray:
    """
    Build persuader-relative belief trajectory values for one round.

    Args:
        row: Round trajectory row.

    Returns:
        Array containing initial belief and serial-question values transformed
        into persuader-relative coordinates.
        Returns an empty array when values are unavailable.
    """
    initial = row.round_obj.target_initial_belief
    if not isinstance(initial, (int, float)):
        return np.asarray([], dtype=float)

    initial_relative = row.round_obj.persuader_relative_belief(float(initial))
    serial_questions = row.round_obj.get_serial_questions(persuader_relative=True)
    if isinstance(serial_questions, list):
        values: list[float] = [float(initial_relative)]
        for value in serial_questions:
            if not isinstance(value, (int, float)):
                return np.asarray([], dtype=float)
            values.append(float(value))
        if len(values) >= 2:
            return np.asarray(values, dtype=float)

    belief = float(initial_relative)
    values = [float(initial_relative)]
    for update in row.updates:
        belief += float(update)
        values.append(float(belief))
    return np.asarray(values, dtype=float)


def _sign_change_count(updates: np.ndarray, *, epsilon: float) -> int:
    """
    Count significant sign changes in an update sequence.

    Args:
        updates: Persuader-relative per-turn updates.
        epsilon: Absolute threshold for treating updates as zero.

    Returns:
        Number of sign changes after removing near-zero updates.
    """
    filtered: list[int] = []
    for value in np.asarray(updates, dtype=float):
        if value > epsilon:
            filtered.append(1)
        elif value < -epsilon:
            filtered.append(-1)
    if len(filtered) < 2:
        return 0
    changes = 0
    previous = filtered[0]
    for current in filtered[1:]:
        if current != previous:
            changes += 1
        previous = current
    return int(changes)


def _initial_belief_bin_from_value(initial_belief: float) -> str:
    """
    Map an initial belief value to a coarse bin label.

    Args:
        initial_belief: Initial belief value in [0,1].

    Returns:
        One of ``very_low``, ``low``, ``mid``, ``high``, ``very_high``,
        or ``unknown``.
    """
    value = float(initial_belief)
    if not np.isfinite(value):
        return "unknown"
    if value < 0.1:
        return "very_low"
    if value < 0.35:
        return "low"
    if value < 0.65:
        return "mid"
    if value < 0.9:
        return "high"
    return "very_high"


def _belief_bin_sort_key(label: str) -> tuple[int, str]:
    """
    Build a stable ordering key for initial-belief bin labels.

    Args:
        label: Bin label string.

    Returns:
        Tuple used for sorting where known bins are ordered first and unknown
        bins follow alphabetically.
    """
    normalized = str(label or "").strip()
    preferred = {
        "very_low": 0,
        "low": 1,
        "mid": 2,
        "high": 3,
        "very_high": 4,
        "unknown": 5,
    }
    if normalized in preferred:
        return (preferred[normalized], normalized)
    return (99, normalized)


def _round_dynamics_row(
    *,
    corpus: str,
    trajectory_index: int,
    row: RoundTrajectory,
    epsilon: float,
) -> dict[str, object]:
    """
    Build movement diagnostics for one round.

    Args:
        corpus: Corpus label.
        trajectory_index: 0-based index inside the corpus.
        row: Round trajectory.
        epsilon: Near-zero threshold.

    Returns:
        Dictionary of round-level movement diagnostics.
    """
    updates = np.asarray(row.updates, dtype=float)
    total_delta = float(np.sum(updates)) if updates.size > 0 else 0.0
    abs_total_delta = float(abs(total_delta))
    has_up = bool(np.any(updates > epsilon))
    has_down = bool(np.any(updates < -epsilon))
    sign_changes = _sign_change_count(updates, epsilon=epsilon)
    beliefs = _belief_trajectory_values(row)
    initial_belief = float(beliefs[0]) if beliefs.size > 0 else float("nan")
    final_belief = float(beliefs[-1]) if beliefs.size > 0 else float("nan")
    raw_belief_delta = (
        float(final_belief - initial_belief)
        if np.isfinite(initial_belief) and np.isfinite(final_belief)
        else float("nan")
    )
    max_up = float(np.max(updates)) if updates.size > 0 else float("nan")
    max_down = float(np.min(updates)) if updates.size > 0 else float("nan")
    abs_max_update = (
        float(np.max(np.abs(updates))) if updates.size > 0 else float("nan")
    )
    supports_raw = row.round_obj.persuader_supports_proposition
    supports_value: bool | None = (
        bool(supports_raw) if isinstance(supports_raw, bool) else None
    )
    stance = (
        "supports"
        if supports_value is True
        else ("opposes" if supports_value is False else "unknown")
    )
    proposition_text = " ".join(str(row.proposition).split())
    return {
        "corpus": corpus,
        "trajectory_index": int(trajectory_index),
        "n_turns": int(len(row.updates)),
        "total_delta": total_delta,
        "abs_total_delta": abs_total_delta,
        "raw_belief_delta": raw_belief_delta,
        "initial_belief": initial_belief,
        "final_belief": final_belief,
        "supports_proposition": supports_value,
        "stance": stance,
        "has_up_step": int(has_up),
        "has_down_step": int(has_down),
        "both_directions": int(has_up and has_down),
        "sign_changes": int(sign_changes),
        "max_up_step": max_up,
        "max_down_step": max_down,
        "abs_max_step": abs_max_update,
        "source_path": str(row.source_path),
        "source_line_index": int(row.source_line_index),
        "source_round_index": row.source_round_index,
        "proposition": proposition_text,
    }


def _movement_summary_row(
    *,
    corpus: str,
    rows: list[RoundTrajectory],
    epsilon: float,
) -> dict[str, object]:
    """
    Summarize movement diagnostics for one corpus.

    Args:
        corpus: Corpus label.
        rows: Trajectory rows.
        epsilon: Near-zero threshold.

    Returns:
        Corpus-level movement summary row.
    """
    if not rows:
        nan = float("nan")
        return {
            "corpus": corpus,
            "rounds": 0,
            "updates": 0,
            "mean_total_delta": nan,
            "median_total_delta": nan,
            "mean_abs_total_delta": nan,
            "toward_round_rate": nan,
            "away_round_rate": nan,
            "near_zero_round_rate": nan,
            "any_up_step_rate": nan,
            "any_down_step_rate": nan,
            "both_directions_rate": nan,
            "up_update_rate": nan,
            "down_update_rate": nan,
            "near_zero_update_rate": nan,
        }

    round_rows = [
        _round_dynamics_row(
            corpus=corpus,
            trajectory_index=index,
            row=row,
            epsilon=epsilon,
        )
        for index, row in enumerate(rows)
    ]
    totals = np.asarray(
        [float(item["total_delta"]) for item in round_rows], dtype=float
    )
    pooled = _flatten_updates(rows)
    pooled_count = int(pooled.size)
    up_updates = float(np.mean(pooled > epsilon)) if pooled_count > 0 else float("nan")
    down_updates = (
        float(np.mean(pooled < -epsilon)) if pooled_count > 0 else float("nan")
    )
    near_zero_updates = (
        float(np.mean(np.abs(pooled) <= epsilon)) if pooled_count > 0 else float("nan")
    )
    return {
        "corpus": corpus,
        "rounds": int(len(rows)),
        "updates": pooled_count,
        "mean_total_delta": float(np.mean(totals)),
        "median_total_delta": float(np.median(totals)),
        "mean_abs_total_delta": float(np.mean(np.abs(totals))),
        "toward_round_rate": float(np.mean(totals > epsilon)),
        "away_round_rate": float(np.mean(totals < -epsilon)),
        "near_zero_round_rate": float(np.mean(np.abs(totals) <= epsilon)),
        "any_up_step_rate": float(
            np.mean([float(item["has_up_step"]) for item in round_rows])
        ),
        "any_down_step_rate": float(
            np.mean([float(item["has_down_step"]) for item in round_rows])
        ),
        "both_directions_rate": float(
            np.mean([float(item["both_directions"]) for item in round_rows])
        ),
        "up_update_rate": up_updates,
        "down_update_rate": down_updates,
        "near_zero_update_rate": near_zero_updates,
    }


def _proposition_stance_delta_rows(
    *,
    round_dynamics_rows: list[dict[str, object]],
    epsilon: float,
) -> list[dict[str, object]]:
    """
    Aggregate round movement by corpus, proposition, and persuader stance.

    Args:
        round_dynamics_rows: Round-level movement rows.
        epsilon: Near-zero threshold for total-delta rates.

    Returns:
        Proposition/stance aggregate rows.
    """
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for item in round_dynamics_rows:
        corpus = str(item.get("corpus", ""))
        proposition = str(item.get("proposition", ""))
        stance = str(item.get("stance", "unknown"))
        total_raw = item.get("total_delta")
        if not corpus or not proposition:
            continue
        if not isinstance(total_raw, (int, float)):
            continue
        grouped[(corpus, proposition, stance)].append(float(total_raw))

    rows: list[dict[str, object]] = []
    for corpus, proposition, stance in sorted(grouped):
        totals = np.asarray(grouped[(corpus, proposition, stance)], dtype=float)
        count = int(totals.size)
        if count <= 0:
            continue
        rows.append(
            {
                "corpus": corpus,
                "proposition": proposition,
                "stance": stance,
                "n_rounds": count,
                "mean_total_delta": float(np.mean(totals)),
                "median_total_delta": float(np.median(totals)),
                "mean_abs_total_delta": float(np.mean(np.abs(totals))),
                "toward_round_rate": float(np.mean(totals > epsilon)),
                "away_round_rate": float(np.mean(totals < -epsilon)),
                "near_zero_round_rate": float(np.mean(np.abs(totals) <= epsilon)),
            }
        )
    return rows


def _proposition_stance_gap_vs_baseline_rows(
    *,
    proposition_rows: list[dict[str, object]],
    baseline_corpus: str,
) -> list[dict[str, object]]:
    """
    Compare proposition/stance means against a baseline corpus.

    Args:
        proposition_rows: Rows from `_proposition_stance_delta_rows`.
        baseline_corpus: Baseline corpus key (for example, `vanilla_llm_target`).

    Returns:
        Gap rows for shared proposition/stance cells.
    """
    keyed: dict[tuple[str, str, str], dict[str, object]] = {}
    corpora: set[str] = set()
    for row in proposition_rows:
        corpus = str(row.get("corpus", ""))
        proposition = str(row.get("proposition", ""))
        stance = str(row.get("stance", "unknown"))
        if not corpus or not proposition:
            continue
        corpora.add(corpus)
        keyed[(corpus, proposition, stance)] = row

    if baseline_corpus not in corpora:
        return []

    baseline_keys = [
        key for key in keyed if key[0] == baseline_corpus and key[1] and key[2]
    ]
    comparators = sorted(
        corpus
        for corpus in corpora
        if corpus not in {baseline_corpus, "human_reference"}
    )
    gap_rows: list[dict[str, object]] = []
    for comparator in comparators:
        for _, proposition, stance in baseline_keys:
            base_row = keyed.get((baseline_corpus, proposition, stance))
            comp_row = keyed.get((comparator, proposition, stance))
            if base_row is None or comp_row is None:
                continue
            base_mean = float(base_row["mean_total_delta"])
            comp_mean = float(comp_row["mean_total_delta"])
            gap_rows.append(
                {
                    "baseline_corpus": baseline_corpus,
                    "comparator_corpus": comparator,
                    "proposition": proposition,
                    "stance": stance,
                    "baseline_n_rounds": int(base_row["n_rounds"]),
                    "comparator_n_rounds": int(comp_row["n_rounds"]),
                    "baseline_mean_total_delta": base_mean,
                    "comparator_mean_total_delta": comp_mean,
                    "mean_total_delta_gap_baseline_minus_comparator": float(
                        base_mean - comp_mean
                    ),
                }
            )
    return sorted(
        gap_rows,
        key=lambda item: (
            str(item["comparator_corpus"]),
            str(item["proposition"]),
            str(item["stance"]),
        ),
    )
