"""Shared CLI argument helpers for simulator plotting scripts."""

from __future__ import annotations

import argparse


def add_initial_belief_match_tolerance_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
) -> None:
    """Attach the standard initial-belief matching tolerance option.

    Args:
        parser: Argument parser to update.
        help_text: Help text shown for the argument.

    Returns:
        None.
    """
    parser.add_argument(
        "--initial-belief-match-tolerance",
        type=float,
        default=None,
        help=help_text,
    )
