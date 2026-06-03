"""Utilities for manipulating Bayesian-network belief distributions."""

from __future__ import annotations

from typing import Any


def reweight_joint_for_target_marginal(
    distribution: list[Any], target_marginal: float
) -> list[Any]:
    """
    Reweight a joint distribution so P(Target=True)=target_marginal.

    Args:
        distribution: List of JointDistributionEntry-like objects.
        target_marginal: Desired marginal probability in [0,1].

    Returns:
        New list of entries with updated probabilities.
    """
    target_marginal = max(0.0, min(1.0, float(target_marginal)))
    new_distribution = [entry.model_copy(deep=True) for entry in distribution]

    true_idxs: list[int] = []
    false_idxs: list[int] = []
    for idx, entry in enumerate(new_distribution):
        if bool(entry.state.get("Target")):
            true_idxs.append(idx)
        else:
            false_idxs.append(idx)

    if not true_idxs or not false_idxs:
        raise ValueError(
            "Joint distribution must include both Target=True and Target=False states."
        )

    def _normalize_subset(indices: list[int], target_total: float) -> None:
        subset_total = sum(float(new_distribution[i].probability) for i in indices)
        if subset_total <= 0.0:
            uniform = target_total / float(len(indices))
            for index in indices:
                new_distribution[index].probability = uniform
            return
        scale = target_total / subset_total
        for index in indices:
            new_distribution[index].probability = (
                float(new_distribution[index].probability) * scale
            )

    _normalize_subset(true_idxs, target_marginal)
    _normalize_subset(false_idxs, 1.0 - target_marginal)

    total = sum(float(entry.probability) for entry in new_distribution)
    if total <= 0.0:
        raise ValueError("Reweighting produced a zero-probability distribution.")
    for entry in new_distribution:
        entry.probability = float(entry.probability) / total
    return new_distribution
