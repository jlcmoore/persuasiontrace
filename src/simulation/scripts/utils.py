"""
Utility helpers for simulation scripts.
"""

import json
import os
from typing import Any


def resolve_proposition_source(
    graphs: list[dict[str, Any]], proposition_source: str | None
) -> str:
    """
    Resolve the proposition source from input graphs when not explicitly provided.

    Args:
        graphs: List of graph dictionaries that may include "proposition_source".
        proposition_source: Optional proposition source passed by the caller.

    Returns:
        The resolved proposition source string.
    """
    if proposition_source is None:
        sources = {g.get("proposition_source") for g in graphs}
        sources.discard(None)
        if not sources:
            raise ValueError(
                "proposition_source is required when input graphs do not define it."
            )
        if len(sources) > 1:
            raise ValueError(
                f"Input graphs contain multiple proposition_source values: {sources}"
            )
        proposition_source = sources.pop()
    return proposition_source


def load_existing_ids(output_file: str) -> set[str]:
    """
    Load existing proposition ids from a DB-format JSONL output file.

    Args:
        output_file: Path to the output JSONL file.

    Returns:
        A set of proposition ids found in the output file.
    """
    existing_targets: set[str] = set()
    if not os.path.exists(output_file):
        return existing_targets

    with open(output_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = record.get("id")
            if isinstance(record_id, str):
                existing_targets.add(record_id)
            elif "target" in record:
                raise ValueError(
                    "Output file uses legacy format (missing id). "
                    "Please regenerate or remove it before resuming."
                )

    return existing_targets
