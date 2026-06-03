"""Shared target-belief bin ranges for rollout and simulator analysis."""

from __future__ import annotations

TARGET_BELIEF_BIN_RANGES: dict[str, tuple[float, float]] = {
    "very_low": (0.0, 0.1),
    "low": (0.1, 0.35),
    "mid": (0.35, 0.65),
    "high": (0.65, 0.9),
    "very_high": (0.9, 1.0),
}
