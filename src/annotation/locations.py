"""
Location models for anchoring annotation records to round data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RoundIndex:
    """Identify a round within the results directory."""

    source_path: str
    line_index: int
    round_index: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for this round location."""
        return {
            "source_path": self.source_path,
            "line_index": self.line_index,
            "round_index": self.round_index,
        }


@dataclass(frozen=True)
class MessageSpan:
    """Describe the unit within a round that is being annotated."""

    role: str
    message_index: int
    unit_type: Literal["message", "sentence"] = "message"
    sentence_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for this span location."""
        return {
            "role": self.role,
            "message_index": self.message_index,
            "unit_type": self.unit_type,
            "sentence_index": self.sentence_index,
        }


@dataclass(frozen=True)
class AnnotationTarget:
    """Bundle round and span locations for a single annotation target."""

    round_index: RoundIndex
    span: MessageSpan

    def as_dict(self) -> dict[str, object]:
        """Return a merged JSON-friendly mapping for this target."""
        payload = dict(self.round_index.as_dict())
        payload.update(self.span.as_dict())
        return payload
