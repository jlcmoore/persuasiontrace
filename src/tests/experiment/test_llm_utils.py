"""Unit tests for experiment.llm_utils."""

from __future__ import annotations

import logging

import litellm
import pytest

from experiment.llm_utils import (
    atomizer_temperature_for_model,
    disable_litellm_logging,
    model_supports_reasoning,
)


def test_disable_litellm_logging_suppresses_debug_info():
    """Helper should silence LiteLLM logger and debug-print side effects."""
    original_level = logging.getLogger("LiteLLM").level
    original_verbose = litellm.set_verbose
    original_suppress = litellm.suppress_debug_info
    try:
        disable_litellm_logging()
        assert logging.getLogger("LiteLLM").level == logging.CRITICAL
        assert litellm.set_verbose is False
        assert litellm.suppress_debug_info is True
    finally:
        logging.getLogger("LiteLLM").setLevel(original_level)
        litellm.set_verbose = original_verbose
        litellm.suppress_debug_info = original_suppress


def test_model_supports_reasoning_reads_model_cost(monkeypatch: pytest.MonkeyPatch):
    """Reasoning support should mirror LiteLLM model-cost metadata."""
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/nonreasoning-test-model",
        {"supports_reasoning": False},
    )
    assert model_supports_reasoning("openai/nonreasoning-test-model") is False
    assert atomizer_temperature_for_model("openai/nonreasoning-test-model") == 0.0


def test_model_supports_reasoning_uses_gpt5_fallback():
    """Date-pinned GPT-5 variants should be treated as reasoning-capable."""
    assert model_supports_reasoning("openai/gpt-5.4-nano-2026-03-17") is True
    assert atomizer_temperature_for_model("openai/gpt-5.4-nano-2026-03-17") is None
