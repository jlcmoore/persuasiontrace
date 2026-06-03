"""I/O utilities for simulation data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def read_jsonl_records(
    file_path: str | Path,
    required_keys: list[str] | None = None,
    *,
    skip_error_records: bool = True,
    flatten_list_rows: bool = False,
) -> list[dict[str, Any]]:
    """
    Read JSONL records from a file, ignoring empty lines and parse errors.

    Args:
        file_path: Path to the JSONL file.
        required_keys: Optional list of keys that must be present in a record.
        skip_error_records: Whether to drop rows containing an ``error`` key.
        flatten_list_rows: Whether to also accept ``list[dict]`` line payloads.

    Returns:
        A list of parsed dictionaries.
    """
    path = Path(file_path)
    if not path.exists():
        logging.error("Input file %s not found.", path)
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(item, dict):
                candidates = [item]
            elif flatten_list_rows and isinstance(item, list):
                candidates = [row for row in item if isinstance(row, dict)]
            else:
                continue

            for candidate in candidates:
                if skip_error_records and "error" in candidate:
                    continue
                if required_keys and not all(key in candidate for key in required_keys):
                    continue
                records.append(candidate)
    return records


def write_jsonl_records(file_path: str | Path, rows: list[dict[str, Any]]) -> None:
    """
    Write dictionaries to a JSONL file.

    Args:
        file_path: Destination path.
        rows: Rows to write.
    """
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl_graphs(
    file_path: str, required_keys: list[str] | None = None
) -> list[dict[str, Any]]:
    """
    Read JSONL graphs from a file, ignoring empty lines and errors.

    Args:
        file_path: Path to the JSONL file.
        required_keys: Optional list of keys that must be present in a record.

    Returns:
        A list of parsed graph dictionaries.
    """
    return read_jsonl_records(
        file_path=file_path,
        required_keys=required_keys,
        skip_error_records=True,
    )
