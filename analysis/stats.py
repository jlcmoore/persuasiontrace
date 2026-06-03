"""
Small statistical helpers for analysis scripts.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def significance_stars(p_value: float | None) -> str:
    """
    Convert a p-value to significance stars.
    """
    if p_value is None or np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 5000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Compute mean and bootstrap CI for a 1D array.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean_val = float(np.mean(clean))
    if clean.size == 1:
        return mean_val, mean_val, mean_val
    idx = rng.integers(0, clean.size, size=(n_boot, clean.size))
    boot_means = np.mean(clean[idx], axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return mean_val, float(lo), float(hi)


def bootstrap_statistic_ci(
    *,
    data: tuple[np.ndarray, ...],
    statistic: Callable[..., float],
    n_boot: int,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float], np.ndarray]:
    """
    Compute a percentile bootstrap confidence interval for a statistic.

    Args:
        data: Tuple of input arrays passed to the statistic function.
        statistic: Statistic callable used by scipy bootstrap.
        n_boot: Number of bootstrap resamples.
        confidence_level: Confidence level for the interval.
        rng: Optional numpy random generator.

    Returns:
        Tuple of ((ci_low, ci_high), bootstrap_distribution).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    result = stats.bootstrap(
        data=data,
        statistic=statistic,
        n_resamples=n_boot,
        confidence_level=confidence_level,
        vectorized=False,
        method="percentile",
        rng=rng,
    )
    ci = (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )
    distribution = np.asarray(result.bootstrap_distribution, dtype=float)
    return ci, distribution


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni correction for a list of p-values.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (idx, pval) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * pval
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def paired_t_test(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> tuple[int, float, float, float]:
    """
    Paired t-test on pre/post arrays; returns (n, mean_diff, t, p).
    """
    mask = np.isfinite(pre) & np.isfinite(post)
    pre_vals = pre[mask]
    post_vals = post[mask]
    n = int(pre_vals.size)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan")
    diffs = post_vals - pre_vals
    mean_diff = float(np.mean(diffs))
    t_stat, p_val = stats.ttest_rel(post_vals, pre_vals)
    if alternative == "greater":
        if np.isfinite(p_val):
            p_val = p_val / 2 if mean_diff > 0 else 1 - p_val / 2
    elif alternative == "less":
        if np.isfinite(p_val):
            p_val = p_val / 2 if mean_diff < 0 else 1 - p_val / 2
    elif alternative != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return n, mean_diff, float(t_stat), float(p_val)


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Welch t-test on two arrays; returns (mean_diff, t, p).
    """
    vals_a = a[np.isfinite(a)]
    vals_b = b[np.isfinite(b)]
    if vals_a.size < 2 or vals_b.size < 2:
        return float("nan"), float("nan"), float("nan")
    t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    mean_diff = float(np.mean(vals_a) - np.mean(vals_b))
    return mean_diff, float(t_stat), float(p_val)
