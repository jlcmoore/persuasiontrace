"""
Search rounds by condition, outcomes, and annotation signals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from annotation.records import (
    ANNOTATION_CODES,
    MessageAnnotation,
    add_annotation_args,
    compute_dialogue_scores,
    extract_message_text,
    load_annotation_rows,
    load_message_annotations,
    resolve_annotation_paths_from_args,
)
from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import (
    add_condition_filter_args,
    filters_from_args,
)
from experiment.round import IndexedRound, Round
from experiment.round_lookup import RoundKey, build_round_lookup


@dataclass(frozen=True)
class SearchFilters:
    """Container for search filters."""

    min_delta: float | None
    max_delta: float | None
    persuaded: bool | None
    min_scores: dict[str, float]
    max_scores: dict[str, float]
    required_codes: dict[str, float]
    limit: int | None
    print_message_annotations: bool
    quotes: list[str]
    quote_mode_any: bool


def parse_bool(value: str) -> bool:
    """Parse a string into a boolean value."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_score_filters(raw_entries: Sequence[str]) -> dict[str, float]:
    """Parse score filters like 'pathos=2.5' into a dict."""
    filters: dict[str, float] = {}
    for entry in raw_entries:
        for part in entry.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            if "=" not in cleaned:
                raise ValueError(f"Score filter must be code=value: {cleaned}")
            code, raw_value = cleaned.split("=", 1)
            code = code.strip()
            raw_value = raw_value.strip()
            if code not in ANNOTATION_CODES:
                raise ValueError(f"Unknown annotation code: {code}")
            filters[code] = float(raw_value)
    return filters


def parse_code_thresholds(raw_entries: Sequence[str]) -> dict[str, float]:
    """Parse code thresholds like 'pathos=2' or 'pathos' into a dict."""
    thresholds: dict[str, float] = {}
    for entry in raw_entries:
        for part in entry.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            if "=" in cleaned:
                code, raw_value = cleaned.split("=", 1)
                code = code.strip()
                raw_value = raw_value.strip()
                if code not in ANNOTATION_CODES:
                    raise ValueError(f"Unknown annotation code: {code}")
                thresholds[code] = float(raw_value)
            else:
                if cleaned not in ANNOTATION_CODES:
                    raise ValueError(f"Unknown annotation code: {cleaned}")
                thresholds[cleaned] = 0.0
    return thresholds


def parse_args() -> argparse.Namespace:
    """Parse CLI args for annotation search."""
    parser = argparse.ArgumentParser(
        description="Search rounds by condition, outcomes, and annotations."
    )
    add_min_date_arg(parser)
    add_condition_filter_args(parser)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Minimum persuader-relative delta (target_belief_change).",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=None,
        help="Maximum persuader-relative delta (target_belief_change).",
    )
    parser.add_argument(
        "--persuaded",
        type=parse_bool,
        default=None,
        help="Filter by whether the target was persuaded (true/false).",
    )
    parser.add_argument(
        "--min-score",
        action="append",
        default=[],
        help="Minimum annotation score per code, e.g. pathos=1.",
    )
    parser.add_argument(
        "--max-score",
        action="append",
        default=[],
        help="Maximum annotation score per code, e.g. pathos=2.",
    )
    parser.add_argument(
        "--has-code",
        action="append",
        default=[],
        help="Require score >= threshold for a code (e.g. pathos or pathos=2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of matches to print.",
    )
    parser.add_argument(
        "--no-message-annotations",
        action="store_true",
        help="Disable per-message annotation scores and rationales.",
    )
    parser.add_argument(
        "--quote",
        action="append",
        default=[],
        help="Require substring match in any message (repeatable).",
    )
    parser.add_argument(
        "--quote-mode",
        choices=["all", "any"],
        default="all",
        help="Whether all or any --quote filters must match.",
    )
    add_annotation_args(parser)
    return parser.parse_args()


def get_persuader_relative_delta(round_obj: Round) -> float | None:
    """Return persuader-relative belief change if available."""
    if (
        round_obj.target_initial_belief is None
        or round_obj.target_final_belief is None
        or round_obj.persuader_supports_proposition is None
    ):
        return None
    return round_obj.target_belief_change(is_target=False)


def get_persuaded_value(round_obj: Round) -> bool | None:
    """Return whether the target was persuaded, if available."""
    if (
        not round_obj.finished()
        or round_obj.persuader_supports_proposition is None
        or round_obj.target_initial_belief is None
        or round_obj.target_final_belief is None
    ):
        return None
    return round_obj.target_persuaded()


def matches_numeric_filter(
    value: float | None, min_value: float | None, max_value: float | None
) -> bool:
    """Return True if value passes a min/max filter or is undefined."""
    if value is None:
        return True
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def matches_score_filters(
    scores: dict[str, float] | None,
    min_scores: dict[str, float],
    max_scores: dict[str, float],
    required_codes: dict[str, float],
) -> bool:
    """Return True if scores pass all code filters or are undefined."""
    if scores is None:
        return True
    for code, min_value in min_scores.items():
        value = scores.get(code)
        if value is None:
            continue
        if value < min_value:
            return False
    for code, max_value in max_scores.items():
        value = scores.get(code)
        if value is None:
            continue
        if value > max_value:
            return False
    for code, threshold in required_codes.items():
        value = scores.get(code)
        if value is None:
            continue
        if value < threshold:
            return False
    return True


def format_annotation_scores(scores: dict[str, float] | None) -> str:
    """Format annotation scores for printing."""
    if not scores:
        return "Annotations: N/A"
    parts = []
    for code in ANNOTATION_CODES:
        value = scores.get(code)
        if value is None:
            continue
        parts.append(f"{code}={value:.2f}")
    if not parts:
        return "Annotations: N/A"
    return "Annotations: " + ", ".join(parts)


def format_score(value: float | None) -> str:
    """Format a numeric score for display."""
    if value is None:
        return "N/A"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def print_message_annotations(
    round_obj: Round, annotations: Sequence[MessageAnnotation]
) -> None:
    """Print message-level annotations for a round."""
    if not annotations:
        return
    print("\n## Message Annotations\n")
    for annotation in sorted(annotations, key=lambda row: row.message_index):
        message_role = annotation.role or "unknown"
        message_text = ""
        if 0 <= annotation.message_index < len(round_obj.messages):
            message = round_obj.messages[annotation.message_index]
            if isinstance(message, dict):
                role_value = message.get("role")
                if isinstance(role_value, str) and role_value:
                    message_role = role_value
                message_text = extract_message_text(message)
        print(f"Message {annotation.message_index} ({message_role})")
        if message_text:
            print(f"  Text: {message_text}")
        else:
            print("  Text: N/A")
        for code in ANNOTATION_CODES:
            score = annotation.scores.get(code)
            rationale = annotation.rationales.get(code)
            if score is None and not rationale:
                continue
            score_str = format_score(score)
            if rationale:
                print(f"  - {code}: {score_str} | {rationale}")
            else:
                print(f"  - {code}: {score_str}")
        print("")


def print_round_summary(
    indexed_round: IndexedRound,
    delta: float | None,
    persuaded: bool | None,
    scores: dict[str, float] | None,
    message_annotations: Sequence[MessageAnnotation] | None,
) -> None:
    """Print a matching round with metadata."""
    print("---")
    print(
        f"Source: {indexed_round.index.source_path} "
        f"line={indexed_round.index.line_index} "
        f"round={indexed_round.index.round_index}"
    )
    print(f"Condition: {indexed_round.condition}")
    if delta is None:
        print("Persuader-relative delta: N/A")
    else:
        print(f"Persuader-relative delta: {delta:.3f}")
    if persuaded is None:
        print("Persuaded: N/A")
    else:
        print(f"Persuaded: {persuaded}")
    print(format_annotation_scores(scores))
    print(str(indexed_round.round))
    if message_annotations:
        print_message_annotations(indexed_round.round, message_annotations)


def build_score_filters(
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return parsed score filters from CLI args."""
    min_scores = parse_score_filters(args.min_score)
    max_scores = parse_score_filters(args.max_score)
    required_codes = parse_code_thresholds(args.has_code)
    return min_scores, max_scores, required_codes


def build_search_filters(args: argparse.Namespace) -> SearchFilters:
    """Build search filters from CLI args."""
    min_scores, max_scores, required_codes = build_score_filters(args)
    return SearchFilters(
        min_delta=args.min_delta,
        max_delta=args.max_delta,
        persuaded=args.persuaded,
        min_scores=min_scores,
        max_scores=max_scores,
        required_codes=required_codes,
        limit=args.limit,
        print_message_annotations=not args.no_message_annotations,
        quotes=[entry.strip() for entry in args.quote if entry.strip()],
        quote_mode_any=args.quote_mode == "any",
    )


def load_score_map(args: argparse.Namespace) -> dict[RoundKey, dict[str, float]]:
    """Load annotation scores keyed by round."""
    annotation_paths = resolve_annotation_paths_from_args(args)
    annotation_rows = load_annotation_rows(annotation_paths)
    return compute_dialogue_scores(annotation_rows)


def load_message_annotation_map(
    args: argparse.Namespace,
) -> dict[RoundKey, list[MessageAnnotation]]:
    """Load message-level annotations keyed by round."""
    if args.no_message_annotations:
        return {}
    annotation_paths = resolve_annotation_paths_from_args(args)
    rows = load_message_annotations(annotation_paths)
    per_round: dict[RoundKey, list[MessageAnnotation]] = {}
    for row in rows:
        per_round.setdefault(row.round_key, []).append(row)
    return per_round


def normalize_text(value: str) -> str:
    """Normalize text for substring matching."""
    return value.lower()


def build_round_message_blob(round_obj: Round) -> str:
    """Concatenate all message text for substring matching."""
    parts: list[str] = []
    for message in round_obj.messages:
        if not isinstance(message, dict):
            continue
        parts.append(extract_message_text(message))
    return normalize_text(" ".join(part for part in parts if part))


def matches_quote_filters(round_obj: Round, filters: SearchFilters) -> bool:
    """Return True if round text matches quote filters."""
    if not filters.quotes:
        return True
    blob = build_round_message_blob(round_obj)
    if not blob:
        return False
    matches = [normalize_text(q) in blob for q in filters.quotes]
    return any(matches) if filters.quote_mode_any else all(matches)


def run_search(
    *,
    round_lookup: dict[RoundKey, IndexedRound],
    score_map: dict[RoundKey, dict[str, float]],
    message_annotation_map: dict[RoundKey, list[MessageAnnotation]],
    filters: SearchFilters,
) -> int:
    """Execute the search and print matching rounds."""
    matched = 0
    for key, indexed_round in round_lookup.items():
        round_obj = indexed_round.round
        delta = get_persuader_relative_delta(round_obj)
        if not matches_numeric_filter(delta, filters.min_delta, filters.max_delta):
            continue
        if not matches_quote_filters(round_obj, filters):
            continue
        persuaded = get_persuaded_value(round_obj)
        if filters.persuaded is not None:
            if persuaded is not None and persuaded != filters.persuaded:
                continue
        scores = score_map.get(key)
        if not matches_score_filters(
            scores, filters.min_scores, filters.max_scores, filters.required_codes
        ):
            continue

        message_annotations = message_annotation_map.get(key, [])
        print_round_summary(
            indexed_round, delta, persuaded, scores, message_annotations
        )
        matched += 1
        if filters.limit is not None and matched >= filters.limit:
            break
    return matched


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    condition_filters = filters_from_args(args)
    filters = build_search_filters(args)
    score_map = load_score_map(args)
    message_annotation_map = load_message_annotation_map(args)
    round_lookup = build_round_lookup(
        min_date=args.min_date, condition_filters=condition_filters
    )
    matched = run_search(
        round_lookup=round_lookup,
        score_map=score_map,
        message_annotation_map=message_annotation_map,
        filters=filters,
    )
    print(f"Matches: {matched}")


if __name__ == "__main__":
    main()
