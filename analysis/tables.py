"""
Helpers for printing compact CLI tables.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def _stringify(value: Any) -> str:
    """
    Convert a value to a display string.
    """
    if value is None:
        return ""
    return str(value)


def _sanitize(text: str) -> str:
    """
    Keep table entries single-line and ASCII-friendly.
    """
    return text.replace("\n", " ").replace("\r", " ")


def _truncate(text: str, width: int) -> str:
    """
    Truncate a string to fit the width, adding ellipsis when needed.
    """
    if width <= 0 or len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_table(
    rows: Iterable[dict[str, Any]],
    *,
    columns: list[str],
    title: str | None = None,
    formatters: dict[str, Callable[[Any], str]] | None = None,
    widths: dict[str, int] | None = None,
    aligns: dict[str, str] | None = None,
) -> None:
    """
    Print a simple aligned table to stdout.
    """
    formatters = formatters or {}
    widths = widths or {}
    aligns = aligns or {}

    processed = []
    for row in rows:
        out_row = {}
        for col in columns:
            val = row.get(col)
            formatter = formatters.get(col, _stringify)
            text = formatter(val)
            text = _sanitize(text)
            if col in widths:
                text = _truncate(text, widths[col])
            out_row[col] = text
        processed.append(out_row)

    col_widths = {}
    for col in columns:
        col_widths[col] = max(
            len(col),
            *(len(row[col]) for row in processed),
        )

    if title:
        print(title)

    header = " ".join(col.ljust(col_widths[col]) for col in columns)
    print(header)
    print("-" * len(header))

    for row in processed:
        parts = []
        for col in columns:
            text = row[col]
            width = col_widths[col]
            align = aligns.get(col, "left")
            if align == "right":
                parts.append(text.rjust(width))
            else:
                parts.append(text.ljust(width))
        print(" ".join(parts))
