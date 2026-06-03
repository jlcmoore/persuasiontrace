"""
Shared plotting utilities for analysis figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

CONDITION_LABEL_MAP: dict[bool, str] = {
    False: "Generic",
    True: "Ppt. Prop.",
}
CONDITION_ORDER: list[bool] = [False, True]


def normalize_bool(value: object) -> bool | None:
    """Normalize common boolean encodings.

    Args:
        value: Input value to normalize.

    Returns:
        True/False for recognized values, otherwise None.
    """

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    return None


def build_x_positions(order: Iterable[bool]) -> dict[bool, int]:
    """Map ordered boolean categories to integer x positions.

    Args:
        order: Ordered boolean categories.

    Returns:
        Mapping from category value to x position.
    """

    return {value: idx for idx, value in enumerate(order)}


def dodge_positions(
    values: Iterable[bool],
    x_positions: dict[bool, int],
    offset: float,
) -> list[float]:
    """Compute x positions with an optional dodge offset.

    Args:
        values: Category values in plot order.
        x_positions: Mapping from category to base x position.
        offset: Horizontal offset to apply.

    Returns:
        List of x positions for plotting.
    """

    return [x_positions[val] + offset for val in values]


def load_condition_df(csv_path: str, feature_col: str | None = None):
    """Load a CSV with participant proposition and persuader type columns.

    Args:
        csv_path: Path to the CSV file.
        feature_col: Optional feature column to cast to string.

    Returns:
        Normalized pandas DataFrame.
    """

    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df["participant_proposition"] = df["participant_proposition"].apply(normalize_bool)
    df = df.dropna(subset=["participant_proposition"])
    df["persuader_type"] = df["persuader_type"].astype(str).str.lower()
    if feature_col and feature_col in df.columns:
        df[feature_col] = df[feature_col].astype(str)
    return df


def load_boolean_participant_df(csv_path: str) -> pd.DataFrame:
    """Load a CSV and normalize only the participant proposition boolean column.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with normalized ``participant_proposition`` values.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df["participant_proposition"] = df["participant_proposition"].apply(normalize_bool)
    df = df.dropna(subset=["participant_proposition"])
    return df


def build_condition_axis_data() -> tuple[dict[bool, str], list[bool], dict[bool, int]]:
    """Return the shared condition label map, order, and x-position mapping.

    Returns:
        Tuple of ``(label_map, order, x_positions)``.
    """
    label_map = dict(CONDITION_LABEL_MAP)
    order = list(CONDITION_ORDER)
    return label_map, order, build_x_positions(order)


def add_dodge_argument(parser: argparse.ArgumentParser) -> None:
    """Attach a standard ``--dodge`` plotting argument to a CLI parser.

    Args:
        parser: Parser to update.

    Returns:
        None.
    """
    parser.add_argument(
        "--dodge",
        type=float,
        default=0.0,
        help="Horizontal offset for separating series.",
    )


def save_figure(fig, output_path: str) -> None:
    """Save a figure to disk, ensuring the parent directory exists.

    Args:
        fig: Matplotlib figure to save.
        output_path: Output file path.

    Returns:
        None.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_errorbar(
    axis,
    x_vals: list[float],
    y_vals: list[float],
    yerr_low: list[float],
    yerr_high: list[float],
    color: str,
    label: str,
) -> None:
    """Plot error bars for a series.

    Args:
        axis: Matplotlib axes to draw on.
        x_vals: X positions.
        y_vals: Y values.
        yerr_low: Lower error sizes.
        yerr_high: Upper error sizes.
        color: Series color.
        label: Series label.

    Returns:
        None.
    """

    axis.errorbar(
        x_vals,
        y_vals,
        yerr=[yerr_low, yerr_high],
        marker="o",
        linewidth=1.5,
        color=color,
        label=label,
        capsize=3,
    )


def plot_persuader_errorbar_series(
    axis,
    *,
    frame: pd.DataFrame,
    x_positions: dict[bool, int],
    offset: float,
    color: str,
    label: str,
) -> tuple[list[float], list[float]]:
    """Plot one persuader-type errorbar series from a condition dataframe.

    Args:
        axis: Matplotlib axis.
        frame: Filtered dataframe for one persuader series.
        x_positions: Mapping from condition key to x position.
        offset: Horizontal dodge offset.
        color: Series color.
        label: Series label.

    Returns:
        Tuple of plotted ``(x_vals, y_vals)``.
    """
    ordered = frame.sort_values("participant_proposition")
    x_vals = dodge_positions(
        ordered["participant_proposition"].tolist(),
        x_positions,
        offset,
    )
    y_vals = ordered["estimate"].astype(float).tolist()
    yerr_low = (ordered["estimate"] - ordered["conf_low"]).astype(float).tolist()
    yerr_high = (ordered["conf_high"] - ordered["estimate"]).astype(float).tolist()
    plot_errorbar(axis, x_vals, y_vals, yerr_low, yerr_high, color, label)
    return x_vals, y_vals
