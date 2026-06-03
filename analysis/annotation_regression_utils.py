"""
Shared helpers for annotation regression scripts.
"""

from __future__ import annotations

import argparse


def add_summary_csv_arg(parser: argparse.ArgumentParser) -> None:
    """Add the summary CSV argument to a parser.

    Args:
        parser: Argument parser to update.

    Returns:
        None.
    """

    parser.add_argument(
        "--summary-csv",
        default="analysis/data/annotation_regression_summary.csv",
        help="Regression summary CSV path.",
    )
