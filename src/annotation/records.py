"""
Utilities for loading annotation records and aggregating scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from annotation.schema import ANNOTATION_CODES
from experiment.condition import Condition
from experiment.round_lookup import RoundKey, normalize_source_path


@dataclass(frozen=True)
class AnnotationRow:
    """Parsed annotation record with scores."""

    round_key: RoundKey
    scores: dict[str, float]


@dataclass(frozen=True)
class MessageAnnotation:
    """Parsed annotation record with message scores and rationales."""

    round_key: RoundKey
    message_index: int
    role: str | None
    scores: dict[str, float]
    rationales: dict[str, str]


def expand_paths(paths: Iterable[str]) -> list[Path]:
    """Expand comma-separated path arguments into file paths."""
    expanded: list[Path] = []
    for entry in paths:
        for part in entry.split(","):
            cleaned = part.strip()
            if cleaned:
                expanded.append(Path(cleaned))
    return expanded


def resolve_annotation_paths(raw_paths: Sequence[str]) -> list[Path]:
    """Resolve annotation file paths, expanding globs."""
    expanded = expand_paths(raw_paths)
    paths: list[Path] = []
    for entry in expanded:
        if any(char in str(entry) for char in ["*", "?", "["]):
            paths.extend(sorted(Path().glob(str(entry))))
        else:
            paths.append(entry)
    return paths


def condition_from_source_path(source_path: str) -> Condition | None:
    """Infer a Condition from a results source path."""
    if not source_path:
        return None
    path = Path(source_path)
    condition_dir = path.parent.name
    if not condition_dir:
        return None
    try:
        return Condition.from_dir(condition_dir)
    except ValueError:
        return None


def extract_scores(parsed: dict[str, object]) -> dict[str, float]:
    """Extract numeric scores from a parsed annotation response."""
    scores: dict[str, float] = {}
    for code in ANNOTATION_CODES:
        entry = parsed.get(code)
        if not isinstance(entry, dict):
            continue
        value = entry.get("score")
        if isinstance(value, (int, float)):
            scores[code] = float(value)
    return scores


def extract_rationales(parsed: dict[str, object]) -> dict[str, str]:
    """Extract rationales from a parsed annotation response."""
    rationales: dict[str, str] = {}
    for code in ANNOTATION_CODES:
        entry = parsed.get(code)
        if not isinstance(entry, dict):
            continue
        value = entry.get("rationale")
        if isinstance(value, str):
            rationales[code] = value
    return rationales


def extract_message_text(message: dict[str, object]) -> str:
    """Extract text content from a message dict."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return " ".join(parts)
    return ""


def parse_annotation_record(
    record: dict[str, object],
) -> tuple[dict[str, float], dict[str, object]] | None:
    """Return scores and target dict for an annotation record."""
    parsed_tuple = _extract_parsed_scores_target(record)
    if parsed_tuple is None:
        return None
    _, scores, target = parsed_tuple
    return scores, target


def parse_message_annotation_record(
    record: dict[str, object],
) -> MessageAnnotation | None:
    """Return a message-level annotation record, if available."""
    parsed_tuple = _extract_parsed_scores_target(record)
    if parsed_tuple is None:
        return None
    parsed, scores, target = parsed_tuple
    unit_type = target.get("unit_type")
    if unit_type not in (None, "message"):
        return None
    round_key = round_key_from_target(target)
    if round_key is None:
        return None
    message_index = target.get("message_index")
    if not isinstance(message_index, int) or message_index < 0:
        return None
    role_value = target.get("role")
    role = str(role_value) if isinstance(role_value, str) else None
    rationales = extract_rationales(parsed)
    return MessageAnnotation(
        round_key=round_key,
        message_index=message_index,
        role=role,
        scores=scores,
        rationales=rationales,
    )


def _extract_parsed_scores_target(
    record: dict[str, object],
) -> tuple[dict[str, object], dict[str, float], dict[str, object]] | None:
    """
    Extract parsed payload, score map, and target payload from a record.

    Args:
        record: Raw annotation record.

    Returns:
        Tuple of (parsed payload, extracted scores, target payload), or None
        when required fields are missing or invalid.
    """
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return None
    scores = extract_scores(parsed)
    if not scores:
        return None
    target = record.get("target")
    if not isinstance(target, dict):
        return None
    return parsed, scores, target


def round_key_from_target(target: dict[str, object]) -> RoundKey | None:
    """Build a RoundKey from an annotation target dict."""
    source_path = normalize_source_path(str(target.get("source_path", "")))
    line_index = int(target.get("line_index", -1))
    round_index = int(target.get("round_index", -1))
    if not source_path or line_index < 0 or round_index < 0:
        return None
    return RoundKey(
        source_path=source_path,
        line_index=line_index,
        round_index=round_index,
    )


def load_annotation_rows(paths: Sequence[Path]) -> list[AnnotationRow]:
    """Load annotation rows from JSONL files."""
    rows: list[AnnotationRow] = []
    records = iter_annotation_records(paths)
    for record in records:
        parsed = parse_annotation_record(record)
        if parsed is None:
            continue
        scores, target = parsed
        round_key = round_key_from_target(target)
        if round_key is None:
            continue
        rows.append(AnnotationRow(round_key=round_key, scores=scores))
    return rows


def load_message_annotations(paths: Sequence[Path]) -> list[MessageAnnotation]:
    """Load message-level annotations from JSONL files."""
    rows: list[MessageAnnotation] = []
    records = iter_annotation_records(paths)
    for record in records:
        parsed = parse_message_annotation_record(record)
        if parsed is None:
            continue
        rows.append(parsed)
    return rows


def compute_dialogue_scores(
    annotations: Sequence[AnnotationRow],
) -> dict[RoundKey, dict[str, float]]:
    """Aggregate annotation scores to the dialogue level."""
    accum: dict[RoundKey, dict[str, list[float]]] = {}
    for row in annotations:
        per_round = accum.setdefault(
            row.round_key, {code: [] for code in ANNOTATION_CODES}
        )
        for code, value in row.scores.items():
            per_round[code].append(value)

    averages: dict[RoundKey, dict[str, float]] = {}
    for key, values in accum.items():
        means: dict[str, float] = {}
        for code, scores in values.items():
            if scores:
                means[code] = float(sum(scores)) / len(scores)
        averages[key] = means
    return averages


def iter_annotation_records(
    paths: Sequence[Path],
    *,
    max_rows: int | None = None,
) -> list[dict[str, object]]:
    """Load annotation records from JSONL files."""
    rows: list[dict[str, object]] = []
    for _, record in _iter_jsonl_objects(paths):
        if record.get("type") != "annotation":
            continue
        rows.append(record)
        if max_rows is not None and len(rows) >= max_rows:
            return rows
    return rows


def iter_annotation_jsonl_records(
    paths: Sequence[Path],
    *,
    max_rows: int | None = None,
) -> list[tuple[Path, dict[str, object]]]:
    """Load raw JSONL records, counting only annotations toward max_rows."""
    rows: list[tuple[Path, dict[str, object]]] = []
    annotation_count = 0
    for path, record in _iter_jsonl_objects(paths):
        rows.append((path, record))
        if record.get("type") == "annotation":
            annotation_count += 1
            if max_rows is not None and annotation_count >= max_rows:
                return rows
    return rows


def _iter_jsonl_objects(
    paths: Sequence[Path],
) -> Iterable[tuple[Path, dict[str, object]]]:
    """
    Yield decoded JSON object records from existing JSONL files.

    Args:
        paths: JSONL candidate paths.

    Yields:
        Tuples of source path and decoded object record.
    """
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parsed = json.loads(stripped)
                if not isinstance(parsed, dict):
                    continue
                yield path, parsed


def add_annotation_args(parser: argparse.ArgumentParser) -> None:
    """Add annotation path arguments to a parser."""
    parser.add_argument(
        "--annotations",
        action="append",
        default=[],
        help="Annotation JSONL path(s). Can be repeated or comma-separated.",
    )


def add_max_rows_arg(parser: argparse.ArgumentParser) -> None:
    """Add max rows argument to a parser."""
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of annotation rows read.",
    )


def resolve_annotation_paths_with_default(raw_paths: Sequence[str]) -> list[Path]:
    """Resolve annotation paths, defaulting to annotations/*.jsonl."""
    paths = resolve_annotation_paths(raw_paths)
    if paths:
        return paths
    return list(Path("annotations").glob("*.jsonl"))


def resolve_annotation_paths_from_args(args) -> list[Path]:
    """Resolve annotation paths from parsed CLI args."""
    raw_paths = getattr(args, "annotations", None) or []
    return resolve_annotation_paths_with_default(raw_paths)
