"""Shared utility helpers for analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def safe_slug(text: str, *, max_chars: int = 64, default: str = "item") -> str:
    """
    Convert free text to a short filesystem-safe slug.

    Args:
        text: Input text.
        max_chars: Maximum slug length.
        default: Fallback slug when normalized text is empty.

    Returns:
        Lowercase slug with ASCII letters, digits, and underscores.
    """
    characters: list[str] = []
    for char in str(text).strip().lower():
        if char.isascii() and char.isalnum():
            characters.append(char)
        else:
            characters.append("_")
    slug = "".join(characters)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    if not slug:
        slug = str(default).strip("_") or "item"
    return slug[: max(1, int(max_chars))]


def resolve_repo_path(path: Path, *, reference_file: Path) -> Path:
    """Resolve a path relative to the repository root.

    Args:
        path: Raw path that may be relative or absolute.
        reference_file: Absolute path to the caller module file.

    Returns:
        Absolute path anchored at the repository root.
    """
    if path.is_absolute():
        return path
    repo_root = reference_file.resolve().parents[1]
    return (repo_root / path).resolve()


def safe_float_or_nan(value: Any) -> float:
    """Parse a numeric value and return NaN on invalid input.

    Args:
        value: Raw value to parse.

    Returns:
        Parsed float value, or NaN when parsing fails.
    """
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def safe_int_or_none(value: Any) -> int | None:
    """Parse an integer value, returning None for invalid input.

    Args:
        value: Raw value to parse.

    Returns:
        Parsed integer, or ``None`` when parsing fails.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
