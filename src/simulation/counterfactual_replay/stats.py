"""Statistical helpers for counterfactual replay analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class PowerConfig:
    """Parameters used for sample-size guidance.

    Attributes:
        alpha: Two-sided type-I error rate.
        power: Desired statistical power.
    """

    alpha: float
    power: float


@dataclass(frozen=True)
class PairedDifferenceSummary:
    """Summary statistics for paired differences.

    Attributes:
        n_pairs: Number of finite paired observations.
        mean_difference: Mean of paired differences.
        ci_low: Lower 95% bootstrap confidence bound for the mean difference.
        ci_high: Upper 95% bootstrap confidence bound for the mean difference.
        std_difference: Sample standard deviation of paired differences.
    """

    n_pairs: int
    mean_difference: float
    ci_low: float
    ci_high: float
    std_difference: float


def finite_values(values: np.ndarray) -> np.ndarray:
    """Return finite entries from a numeric vector.

    Args:
        values: Input numeric vector.

    Returns:
        Vector containing only finite values.
    """

    return values[np.isfinite(values)]


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Compute mean and percentile bootstrap confidence interval.

    Args:
        values: Input vector.
        n_bootstrap: Number of bootstrap resamples.
        rng: Numpy random generator.

    Returns:
        Tuple of ``(mean, ci_low, ci_high)``.
    """

    clean_values = finite_values(values)
    if clean_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean_value = float(np.mean(clean_values))
    if clean_values.size == 1:
        return mean_value, mean_value, mean_value

    bootstrap_indices = rng.integers(
        0,
        clean_values.size,
        size=(n_bootstrap, clean_values.size),
    )
    bootstrap_means = clean_values[bootstrap_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean_value, float(ci_low), float(ci_high)


def replay_score_from_means(
    *,
    target_mean: float,
    node_mean: float,
    node_delta_mean: float,
) -> float:
    """Compute replay human-likeness score from component means.

    Args:
        target_mean: Mean final target error.
        node_mean: Mean final node MAE.
        node_delta_mean: Mean node-delta MAE.

    Returns:
        Replay score or NaN when any input is non-finite.
    """

    if not (
        np.isfinite(target_mean)
        and np.isfinite(node_mean)
        and np.isfinite(node_delta_mean)
    ):
        return float("nan")
    return float(math.exp(-(target_mean + node_mean + node_delta_mean)))


def bootstrap_replay_score_ci(
    *,
    target_values: np.ndarray,
    node_values: np.ndarray,
    node_delta_values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a 95% confidence interval for replay score.

    Args:
        target_values: Per-round target-error values.
        node_values: Per-round final-node MAE values.
        node_delta_values: Per-round node-delta MAE values.
        n_bootstrap: Number of bootstrap resamples.
        rng: Numpy random generator.

    Returns:
        Tuple of ``(ci_low, ci_high)`` for replay score.
    """

    finite_mask = (
        np.isfinite(target_values)
        & np.isfinite(node_values)
        & np.isfinite(node_delta_values)
    )
    aligned_target = target_values[finite_mask]
    aligned_node = node_values[finite_mask]
    aligned_node_delta = node_delta_values[finite_mask]
    n_rows = int(aligned_target.size)
    if n_rows == 0:
        return float("nan"), float("nan")
    if n_rows == 1:
        one_score = replay_score_from_means(
            target_mean=float(aligned_target[0]),
            node_mean=float(aligned_node[0]),
            node_delta_mean=float(aligned_node_delta[0]),
        )
        return one_score, one_score

    score_indices = rng.integers(0, n_rows, size=(n_bootstrap, n_rows))
    target_boot = aligned_target[score_indices].mean(axis=1)
    node_boot = aligned_node[score_indices].mean(axis=1)
    node_delta_boot = aligned_node_delta[score_indices].mean(axis=1)
    score_boot = np.exp(-(target_boot + node_boot + node_delta_boot))
    ci_low, ci_high = np.quantile(score_boot, [0.025, 0.975])
    return float(ci_low), float(ci_high)


def summarize_paired_differences(
    *,
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> PairedDifferenceSummary | None:
    """Summarize paired differences ``values_a - values_b`` with bootstrap CI.

    Args:
        values_a: First aligned vector.
        values_b: Second aligned vector.
        n_bootstrap: Number of bootstrap resamples.
        rng: Numpy random generator.

    Returns:
        ``PairedDifferenceSummary`` when finite pairs exist, otherwise ``None``.
    """

    finite_mask = np.isfinite(values_a) & np.isfinite(values_b)
    differences = values_a[finite_mask] - values_b[finite_mask]
    n_pairs = int(differences.size)
    if n_pairs == 0:
        return None

    mean_difference = float(np.mean(differences))
    if n_pairs == 1:
        return PairedDifferenceSummary(
            n_pairs=n_pairs,
            mean_difference=mean_difference,
            ci_low=mean_difference,
            ci_high=mean_difference,
            std_difference=float("nan"),
        )

    bootstrap_indices = rng.integers(0, n_pairs, size=(n_bootstrap, n_pairs))
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    std_difference = float(np.std(differences, ddof=1))
    return PairedDifferenceSummary(
        n_pairs=n_pairs,
        mean_difference=mean_difference,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        std_difference=std_difference,
    )


def paired_sample_size_guidance(
    *,
    mean_difference: float,
    std_difference: float,
    n_pairs: int,
    power_config: PowerConfig,
) -> tuple[float, float, float]:
    """Estimate paired-test sample-size guidance for an observed effect.

    Args:
        mean_difference: Observed mean paired difference.
        std_difference: Sample standard deviation of paired differences.
        n_pairs: Current number of paired observations.
        power_config: Desired alpha/power settings.

    Returns:
        Tuple ``(mde_current_n, estimated_total_n, additional_needed)``.
    """

    mde_current_n = float("nan")
    estimated_total_n = float("inf")
    additional_needed = float("inf")
    if n_pairs <= 1 or not np.isfinite(std_difference):
        return mde_current_n, estimated_total_n, additional_needed

    z_alpha = float(norm.ppf(1.0 - (power_config.alpha / 2.0)))
    z_power = float(norm.ppf(power_config.power))
    z_total = z_alpha + z_power

    mde_current_n = float(z_total * std_difference / math.sqrt(n_pairs))
    if abs(mean_difference) > 0.0:
        estimated_total_n = float(
            math.ceil((z_total * std_difference / abs(mean_difference)) ** 2)
        )
        additional_needed = float(max(0, int(estimated_total_n) - n_pairs))
    return mde_current_n, estimated_total_n, additional_needed
