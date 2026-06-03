"""Shared rhetorical-mode definitions used across simulator prompts."""

from __future__ import annotations

RHETORICAL_MODE_DEFINITIONS: tuple[tuple[str, str], ...] = (
    (
        "logos",
        "Use of facts, logic, or reasoning to persuade (causal explanations, "
        "comparisons, statistics). Exclude mere assertions of opinion without "
        "explanation.",
    ),
    (
        "pathos",
        "Emotional or affective appeals (fear, empathy, pride). Vivid storytelling "
        "to move the listener.",
    ),
    (
        "ethos",
        "Attempts to build the speaker's credibility, trustworthiness, or authority "
        "(stating lived or professional expertise).",
    ),
)


def rhetorical_mode_definition_lines(
    *,
    prefix: str = "- ",
    uppercase_names: bool = False,
) -> list[str]:
    """
    Return formatted rhetorical-mode definition lines for prompt construction.

    Args:
        prefix: Prefix prepended to each definition line.
        uppercase_names: Whether to uppercase mode names.

    Returns:
        List of formatted definition lines in canonical mode order.
    """
    lines: list[str] = []
    for mode_name, definition in RHETORICAL_MODE_DEFINITIONS:
        name = mode_name.upper() if uppercase_names else mode_name
        lines.append(f"{prefix}{name}: {definition}")
    return lines
