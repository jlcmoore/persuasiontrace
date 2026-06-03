"""Numerical helpers for simulator counterfactual replay metrics."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any


def normalize_distribution_payload(
    distribution: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a serialized joint distribution so probabilities sum to 1."""
    copied: list[dict[str, Any]] = []
    for entry in distribution:
        state = entry.get("state")
        probability = entry.get("probability")
        if not isinstance(state, dict):
            raise ValueError("Distribution entry is missing a valid state mapping.")
        if not isinstance(probability, (int, float)):
            raise ValueError("Distribution entry has non-numeric probability.")
        copied.append(
            {
                "state": {str(key): bool(value) for key, value in state.items()},
                "probability": float(probability),
            }
        )
    renormalize_probabilities(copied)
    return copied


def marginal_true_probability(
    distribution: list[dict[str, Any]],
    variable: str,
) -> float:
    """Compute P(variable=True) from a serialized joint distribution payload."""
    total = 0.0
    for entry in distribution:
        state = entry["state"]
        if variable not in state:
            raise ValueError(f"State payload is missing variable '{variable}'.")
        if bool(state[variable]):
            total += float(entry["probability"])
    return float(total)


def validate_ipf_targets(
    distribution: list[dict[str, Any]],
    targets: list[tuple[str, float]],
) -> None:
    """Validate target marginals and ensure variables exist in each state."""
    for variable, marginal in targets:
        value = float(marginal)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Requested marginal for '{variable}' must be in [0,1].")
        for entry in distribution:
            state = entry["state"]
            if variable not in state:
                raise ValueError(
                    f"Cannot fit marginal for '{variable}'; variable missing in state."
                )


def renormalize_probabilities(distribution: list[dict[str, Any]]) -> None:
    """Normalize in-place probability mass for a distribution payload."""
    total = sum(float(entry["probability"]) for entry in distribution)
    if total <= 0.0:
        raise ValueError("IPF update produced non-positive total probability.")
    for entry in distribution:
        entry["probability"] = float(entry["probability"]) / float(total)


def ipf_adjust_one_variable(
    distribution: list[dict[str, Any]],
    *,
    variable: str,
    desired_true: float,
    tol: float,
) -> float:
    """Apply one IPF adjustment step for a single variable and return error."""
    current_true = marginal_true_probability(distribution, variable)
    current_false = 1.0 - current_true
    error = abs(current_true - desired_true)

    if error <= tol:
        return error
    if current_true <= 0.0 or current_false <= 0.0:
        raise ValueError(
            f"Cannot reweight '{variable}' because one side has zero mass."
        )

    desired_false = 1.0 - desired_true
    scale_true = desired_true / current_true
    scale_false = desired_false / current_false

    for entry in distribution:
        multiplier = scale_true if bool(entry["state"][variable]) else scale_false
        entry["probability"] = float(entry["probability"]) * float(multiplier)
    renormalize_probabilities(distribution)
    return error


def ipf_match_marginals(
    distribution: list[dict[str, Any]],
    target_marginals: dict[str, float],
    *,
    max_iter: int,
    tol: float,
) -> list[dict[str, Any]]:
    """Fit a distribution to requested Bernoulli marginals via IPF."""
    if max_iter <= 0:
        raise ValueError("ipf max_iter must be positive.")
    if tol <= 0:
        raise ValueError("ipf tol must be positive.")

    normalized = normalize_distribution_payload(distribution)
    ordered_targets = sorted(target_marginals.items(), key=lambda item: item[0])
    validate_ipf_targets(normalized, ordered_targets)

    for _ in range(max_iter):
        max_error = 0.0
        for variable, marginal in ordered_targets:
            error = ipf_adjust_one_variable(
                normalized,
                variable=variable,
                desired_true=float(marginal),
                tol=tol,
            )
            max_error = max(max_error, error)
        if max_error <= tol:
            break
    return normalized


def mean_abs_error(values_a: list[float], values_b: list[float]) -> float:
    """Compute MAE on aligned prefixes of two vectors."""
    size = min(len(values_a), len(values_b))
    if size <= 0:
        return math.nan
    diffs = [abs(float(values_a[i]) - float(values_b[i])) for i in range(size)]
    return float(mean(diffs))


def node_mae(
    truth: dict[str, float] | None,
    pred: dict[str, float] | None,
) -> float:
    """Compute node-level MAE on common node ids."""
    if truth is None or pred is None:
        return math.nan
    common = sorted(set(truth) & set(pred))
    if not common:
        return math.nan
    diffs = [abs(float(truth[node]) - float(pred[node])) for node in common]
    return float(mean(diffs))


def node_delta_mae(
    initial_truth: dict[str, float] | None,
    final_truth: dict[str, float] | None,
    initial_pred: dict[str, float] | None,
    final_pred: dict[str, float] | None,
) -> float:
    """Compute MAE between true and predicted node deltas."""
    if (
        initial_truth is None
        or final_truth is None
        or initial_pred is None
        or final_pred is None
    ):
        return math.nan
    common = sorted(
        set(initial_truth) & set(final_truth) & set(initial_pred) & set(final_pred)
    )
    if not common:
        return math.nan
    diffs: list[float] = []
    for node in common:
        true_delta = float(final_truth[node]) - float(initial_truth[node])
        pred_delta = float(final_pred[node]) - float(initial_pred[node])
        diffs.append(abs(pred_delta - true_delta))
    return float(mean(diffs))


def finite_mean(values: list[float]) -> float:
    """Mean of finite values; NaN when no finite values exist."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return float(mean(finite))


def replay_human_like_score(
    *,
    mean_final_target_error: float,
    mean_final_node_mae: float,
    mean_node_delta_mae: float,
) -> float:
    """Convert aggregate replay errors into a bounded similarity score."""
    values = (
        mean_final_target_error,
        mean_final_node_mae,
        mean_node_delta_mae,
    )
    if not all(math.isfinite(value) for value in values):
        return math.nan
    return float(
        math.exp(-(mean_final_target_error + mean_final_node_mae + mean_node_delta_mae))
    )
