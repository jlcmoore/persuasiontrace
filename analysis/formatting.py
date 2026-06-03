"""
Helpers for formatting analysis labels.
"""

from __future__ import annotations

from typing import Sequence


def split_condition_label(label: str) -> str:
    """
    Split condition labels onto two lines at the first " [".
    """
    if " [" not in label:
        return label
    return label.replace(" [", "\n[", 1)


def condition_color_map(conditions: Sequence[str]) -> dict[str, str]:
    """
    Return a stable color mapping for condition labels.
    """
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#b07aa1",
        "#9c755f",
        "#edc949",
        "#76b7b2",
        "#af7aa1",
        "#ff9da7",
    ]
    unique = sorted({str(cond) for cond in conditions if cond})
    color_map: dict[str, str] = {}
    for idx, cond in enumerate(unique):
        color_map[cond] = palette[idx % len(palette)]
    return color_map
