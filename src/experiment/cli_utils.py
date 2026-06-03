"""
Shared CLI helpers for experiment-level scripts.
"""

from __future__ import annotations

import argparse


def add_min_date_arg(parser: argparse.ArgumentParser) -> None:
    """Attach a shared ``--min-date`` filter argument to a parser."""
    parser.add_argument(
        "--min-date",
        "--min",
        dest="min_date",
        type=str,
        default=None,
        help="Only load results at or after YYYY-MM-DD.",
    )
