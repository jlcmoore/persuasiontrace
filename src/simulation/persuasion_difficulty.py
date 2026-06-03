"""Reusable math and evaluation helpers for persuasion-difficulty analysis."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from simulation.belief_utils import reweight_joint_for_target_marginal
from simulation.target import BayesianNetwork, JointDistributionEntry
from simulation.target_bins import TARGET_BELIEF_BIN_RANGES

LOGIT_EPSILON = 1e-6
SLOPE_EPSILON = 1e-6
SLOPE_DELTA = 1e-3
MAX_DIFFICULTY_CAP = 1e5


@dataclass(frozen=True)
class DifficultyEvalConfig:
    """Settings bundle for proposition difficulty evaluation."""

    init_mode: str
    bins: list[str]
    samples_per_bin: int
    goal_delta: float


@dataclass(frozen=True)
class PropositionEvalContext:
    """Per-proposition context used by initialization evaluation."""

    source: str
    proposition_id: str
    bn: BayesianNetwork
    goal_delta: float


def clip_probability(value: float, epsilon: float = LOGIT_EPSILON) -> float:
    """Clip a value to an open probability interval.

    Args:
        value: Raw probability.
        epsilon: Endpoint buffer for numeric stability.

    Returns:
        Probability clipped to ``[epsilon, 1-epsilon]``.
    """
    return min(1.0 - epsilon, max(epsilon, float(value)))


def logit(probability: float) -> float:
    """Compute the logit transform.

    Args:
        probability: Input probability.

    Returns:
        Log-odds value.
    """
    p_value = clip_probability(probability)
    return math.log(p_value / (1.0 - p_value))


def opposite_direction_goal(p_target: float, goal_delta: float) -> float:
    """Return the directional goal belief opposite the current side.

    Args:
        p_target: Initial target belief.
        goal_delta: Desired absolute move size.

    Returns:
        Goal belief in ``[0,1]``.
    """
    delta = abs(float(goal_delta))
    if p_target >= 0.5:
        return max(0.0, p_target - delta)
    return min(1.0, p_target + delta)


def build_initialization_targets(
    *,
    bn: BayesianNetwork,
    init_mode: str,
    bins: list[str],
    samples_per_bin: int,
    rng: random.Random,
) -> list[tuple[str, float]]:
    """Build target-belief initializations for one proposition.

    Args:
        bn: Bayesian network for the proposition.
        init_mode: Selected initialization mode.
        bins: Active target bins.
        samples_per_bin: Number of random samples per bin.
        rng: Random generator for reproducibility.

    Returns:
        List of (init_label, target_marginal) pairs.
    """
    out: list[tuple[str, float]] = []
    prior = float(bn.marginal_target_probability(bn.joint_distribution))
    if init_mode in {"all", "prior"}:
        out.append(("prior", prior))

    if init_mode in {"all", "bin_centers"}:
        for bin_name in bins:
            low, high = TARGET_BELIEF_BIN_RANGES[bin_name]
            out.append((f"bin_center:{bin_name}", (low + high) / 2.0))

    if init_mode == "bin_samples":
        for bin_name in bins:
            low, high = TARGET_BELIEF_BIN_RANGES[bin_name]
            for sample_index in range(samples_per_bin):
                sampled = float(rng.uniform(low, high))
                out.append((f"bin_sample:{bin_name}:{sample_index}", sampled))

    return out


def tilt_distribution_on_node(
    distribution: list[JointDistributionEntry],
    node_id: str,
    log_lr: float,
) -> list[JointDistributionEntry]:
    """Apply a node-specific log-likelihood tilt to a joint distribution.

    Args:
        distribution: Input joint distribution.
        node_id: Node receiving the tilt.
        log_lr: Log-likelihood-ratio weight for node=True states.

    Returns:
        New normalized distribution after tilt.
    """
    tilted = [entry.model_copy(deep=True) for entry in distribution]
    multiplier = math.exp(float(log_lr))
    for entry in tilted:
        if bool(entry.state.get(node_id)):
            entry.probability = float(entry.probability) * multiplier

    total = sum(float(entry.probability) for entry in tilted)
    if total <= 0.0:
        raise ValueError("Tilt produced a non-positive normalization constant.")
    for entry in tilted:
        entry.probability = float(entry.probability) / total
    return tilted


def directional_slope_for_node(
    *,
    bn: BayesianNetwork,
    distribution: list[JointDistributionEntry],
    node_id: str,
    goal_direction: int,
    delta: float = SLOPE_DELTA,
) -> float:
    """Estimate directional target slope under tiny node tilts.

    Args:
        bn: Bayesian network for marginal computations.
        distribution: Current joint distribution.
        node_id: Node to perturb.
        goal_direction: +1 for increasing target, -1 for decreasing target.
        delta: Central-difference step in log-likelihood-ratio units.

    Returns:
        Directional slope of target belief per log-likelihood unit.
    """
    tilted_plus = tilt_distribution_on_node(distribution, node_id=node_id, log_lr=delta)
    tilted_minus = tilt_distribution_on_node(
        distribution, node_id=node_id, log_lr=-delta
    )
    p_plus = float(bn.marginal_target_probability(tilted_plus))
    p_minus = float(bn.marginal_target_probability(tilted_minus))
    local_slope = (p_plus - p_minus) / (2.0 * delta)
    return float(local_slope * goal_direction)


def structure_aware_difficulty(
    *,
    bn: BayesianNetwork,
    distribution: list[JointDistributionEntry],
    required_abs_delta: float,
    goal_direction: int,
) -> tuple[float, float, str, bool, bool]:
    """Compute structure-aware difficulty from best available belief lever.

    Args:
        bn: Bayesian network for the proposition.
        distribution: Current initialized joint distribution.
        required_abs_delta: Required absolute target move.
        goal_direction: +1 for increase, -1 for decrease.

    Returns:
        Tuple ``(difficulty, best_directional_slope, best_node, capped, infeasible)``.
    """
    best_node = ""
    best_slope = -float("inf")
    for node_id in bn.all_nodes:
        slope = directional_slope_for_node(
            bn=bn,
            distribution=distribution,
            node_id=node_id,
            goal_direction=goal_direction,
        )
        if slope > best_slope:
            best_slope = slope
            best_node = node_id

    if best_slope <= 0.0:
        return MAX_DIFFICULTY_CAP, float(best_slope), best_node, True, True

    usable_slope = max(best_slope, SLOPE_EPSILON)
    raw_difficulty = required_abs_delta / usable_slope
    if raw_difficulty >= MAX_DIFFICULTY_CAP:
        return MAX_DIFFICULTY_CAP, float(best_slope), best_node, True, False
    return float(raw_difficulty), float(best_slope), best_node, False, False


def evaluate_record(
    *,
    source: str,
    record: dict[str, Any],
    eval_config: DifficultyEvalConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Evaluate one proposition under multiple initializations.

    Args:
        source: Proposition source label.
        record: Proposition record containing id and bayesian_network.
        eval_config: Evaluation settings shared across propositions.
        rng: Random generator for reproducibility.

    Returns:
        Per-initialization metric rows.
    """
    proposition_id = str(record.get("id", ""))
    bn_payload = record.get("bayesian_network")
    if not isinstance(bn_payload, dict):
        return []

    bn = BayesianNetwork(**bn_payload)
    init_targets = build_initialization_targets(
        bn=bn,
        init_mode=eval_config.init_mode,
        bins=eval_config.bins,
        samples_per_bin=eval_config.samples_per_bin,
        rng=rng,
    )
    rows: list[dict[str, Any]] = []
    context = PropositionEvalContext(
        source=source,
        proposition_id=proposition_id,
        bn=bn,
        goal_delta=eval_config.goal_delta,
    )
    for init_label, init_target in init_targets:
        rows.append(
            evaluate_initialization(
                context=context,
                init_label=init_label,
                init_target=init_target,
            )
        )
    return rows


def evaluate_initialization(
    *,
    context: PropositionEvalContext,
    init_label: str,
    init_target: float,
) -> dict[str, Any]:
    """Evaluate one proposition at one initialized target belief.

    Args:
        context: Shared proposition context.
        init_label: Initialization label.
        init_target: Desired initialized target marginal.

    Returns:
        Metric row for one initialization.
    """
    initialized_dist = reweight_joint_for_target_marginal(
        context.bn.joint_distribution, target_marginal=init_target
    )
    init_belief = float(context.bn.marginal_target_probability(initialized_dist))
    goal_belief = float(opposite_direction_goal(init_belief, context.goal_delta))
    required_abs_delta = abs(goal_belief - init_belief)

    target_only_logit = abs(logit(goal_belief) - logit(init_belief))
    target_only_local = 1.0 / (
        clip_probability(init_belief) * (1.0 - clip_probability(init_belief))
    )

    goal_direction = 0
    if goal_belief > init_belief:
        goal_direction = 1
    if goal_belief < init_belief:
        goal_direction = -1

    if goal_direction == 0 or required_abs_delta == 0.0:
        structure_diff = 0.0
        best_slope = 0.0
        best_node = ""
        structure_capped = False
        structure_infeasible = False
    else:
        (
            structure_diff,
            best_slope,
            best_node,
            structure_capped,
            structure_infeasible,
        ) = structure_aware_difficulty(
            bn=context.bn,
            distribution=initialized_dist,
            required_abs_delta=required_abs_delta,
            goal_direction=goal_direction,
        )

    return {
        "source": context.source,
        "proposition_id": context.proposition_id,
        "init_label": init_label,
        "init_target_belief": init_belief,
        "goal_target_belief": goal_belief,
        "required_abs_delta": required_abs_delta,
        "target_only_difficulty_logit_delta": target_only_logit,
        "target_only_difficulty_local_per_unit": target_only_local,
        "structure_aware_difficulty": structure_diff,
        "structure_aware_best_directional_slope": best_slope,
        "structure_aware_best_node": best_node,
        "structure_aware_capped": structure_capped,
        "structure_aware_infeasible_direction": structure_infeasible,
    }


def summarize_init_label(init_label: str) -> str:
    """Map initialization labels to summary-friendly groups.

    Args:
        init_label: Detailed initialization label.

    Returns:
        Grouped initialization label for summary aggregation.
    """
    if not init_label.startswith("bin_sample:"):
        return init_label
    parts = init_label.split(":")
    if len(parts) >= 2:
        return f"bin_sample:{parts[1]}"
    return init_label


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build grouped summary rows for quick comparison.

    Args:
        rows: Per-proposition metric rows.

    Returns:
        Summary rows grouped by source and init label.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        init_group_label = summarize_init_label(str(row["init_label"]))
        key = (str(row["source"]), init_group_label)
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (source, init_label), values in sorted(grouped.items()):
        count = len(values)
        mean_target_only = (
            sum(float(row["target_only_difficulty_logit_delta"]) for row in values)
            / count
        )
        mean_structure = (
            sum(float(row["structure_aware_difficulty"]) for row in values) / count
        )
        mean_required_delta = (
            sum(float(row["required_abs_delta"]) for row in values) / count
        )
        summary.append(
            {
                "source": source,
                "init_label": init_label,
                "n_rows": count,
                "mean_required_abs_delta": mean_required_delta,
                "mean_target_only_difficulty_logit_delta": mean_target_only,
                "mean_structure_aware_difficulty": mean_structure,
                "mean_structure_minus_target_only": mean_structure - mean_target_only,
            }
        )
    return summary
