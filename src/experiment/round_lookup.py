"""
Helpers for indexing rounds and looking them up by source indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from experiment.condition import Condition
from experiment.condition_filters import condition_matches_filters
from experiment.round import IndexedRound, load_round_results


@dataclass(frozen=True)
class RoundKey:
    """Key that identifies a round in results storage."""

    source_path: str
    line_index: int
    round_index: int


def normalize_source_path(source_path: str) -> str:
    """Normalize source paths for stable joins."""
    return str(Path(source_path).resolve())


def round_key_from_index(index) -> RoundKey:
    """Build a RoundKey from an IndexedRound index."""
    return RoundKey(
        source_path=normalize_source_path(str(index.source_path)),
        line_index=int(index.line_index),
        round_index=int(index.round_index),
    )


def flatten_indexed_rounds(
    rounds: Sequence[IndexedRound] | Sequence[Sequence[IndexedRound]],
) -> Iterable[IndexedRound]:
    """Flatten nested IndexedRound containers."""
    if not rounds:
        return []
    first = rounds[0]
    if isinstance(first, IndexedRound):
        return list(rounds)
    flattened: list[IndexedRound] = []
    for sub in rounds:
        flattened.extend(list(sub))
    return flattened


def iter_indexed_rounds(
    *,
    min_date: str | None = None,
    condition_filters: dict[str, object] | None = None,
    include_all_files: bool = False,
) -> Iterable[tuple[Condition, IndexedRound]]:
    """Yield (Condition, IndexedRound) pairs from the results directory."""
    condition_to_rounds = load_round_results(
        min_date=min_date,
        include_indices=True,
        include_all_files=include_all_files,
    )
    for condition, round_groups in condition_to_rounds.items():
        if condition_filters and not condition_matches_filters(
            condition, condition_filters
        ):
            continue
        for indexed_round in flatten_indexed_rounds(round_groups):
            yield condition, indexed_round


def build_round_lookup(
    *,
    min_date: str | None,
    condition_filters: dict[str, object] | None,
    include_all_files: bool = False,
) -> dict[RoundKey, IndexedRound]:
    """Load rounds and map them by RoundKey."""
    lookup: dict[RoundKey, IndexedRound] = {}
    for _, indexed_round in iter_indexed_rounds(
        min_date=min_date,
        condition_filters=condition_filters,
        include_all_files=include_all_files,
    ):
        key = round_key_from_index(indexed_round.index)
        lookup[key] = indexed_round
    return lookup
