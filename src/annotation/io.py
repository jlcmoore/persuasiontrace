"""
Iterators for round-based annotation inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from annotation.locations import AnnotationTarget, MessageSpan, RoundIndex
from experiment.condition import Condition
from experiment.round import Round
from experiment.round_lookup import iter_indexed_rounds


@dataclass(frozen=True)
class AnnotationMessage:
    """Payload for a single message annotation target."""

    target: AnnotationTarget
    condition: Condition
    round_obj: Round
    message: dict[str, str]


def iter_persuader_messages(
    *,
    min_date: str | None = None,
    condition_filters: dict[str, object] | None = None,
    include_all_files: bool = False,
) -> Iterator[AnnotationMessage]:
    """
    Yield persuader messages with stable annotation targets.
    """
    for condition, indexed_round in iter_indexed_rounds(
        min_date=min_date,
        condition_filters=condition_filters,
        include_all_files=include_all_files,
    ):
        for message_index, message in enumerate(indexed_round.messages):
            role = str(message.get("role", "")).lower()
            if role != "persuader":
                continue
            target = AnnotationTarget(
                round_index=_coerce_round_index(indexed_round.index),
                span=MessageSpan(
                    role=role,
                    message_index=message_index,
                    unit_type="message",
                ),
            )
            yield AnnotationMessage(
                target=target,
                condition=condition,
                round_obj=indexed_round.round,
                message=message,
            )


def _coerce_round_index(index: RoundIndex) -> RoundIndex:
    """Return a RoundIndex with a stable string path."""
    return RoundIndex(
        source_path=str(index.source_path),
        line_index=int(index.line_index),
        round_index=int(index.round_index),
    )
