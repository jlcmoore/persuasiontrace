"""
Annotation utilities for storing and locating LLM-coded results.
"""

from __future__ import annotations

from .locations import AnnotationTarget, MessageSpan, RoundIndex

__all__ = [
    "AnnotationTarget",
    "MessageSpan",
    "RoundIndex",
]
