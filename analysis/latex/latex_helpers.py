"""Shared helpers for rendering LaTeX content in exporter modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def escape_latex_inline(text: str) -> str:
    """Escape text for inline LaTeX usage.

    Args:
        text: Raw text value.

    Returns:
        LaTeX-safe inline text.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for raw, replacement in replacements.items():
        escaped = escaped.replace(raw, replacement)
    return escaped


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk.

    Args:
        path: Absolute path to a JSONL file.

    Returns:
        Parsed JSON object rows from the file.

    Raises:
        FileNotFoundError: If the requested file path does not exist.
        ValueError: If any non-empty line is not valid JSON object syntax.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL source file not found: {path}")

    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text("utf-8").splitlines(), start=1
    ):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        record = json.loads(stripped_line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(record)
    return records
