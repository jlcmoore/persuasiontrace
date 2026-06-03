"""Tests for deterministic persuader policy helpers."""

from __future__ import annotations

from experiment.persuader_policies import (
    is_naive_persuader_model,
    naive_persuader_action_for_round,
    naive_persuader_message,
)


def test_is_naive_persuader_model_exact_key() -> None:
    """Naive persuader model key should resolve case-insensitively."""
    assert is_naive_persuader_model("naive") is True
    assert is_naive_persuader_model("NaIvE") is True
    assert is_naive_persuader_model("naive-persuader") is False
    assert is_naive_persuader_model("naive_persuader") is False
    assert is_naive_persuader_model("openai/gpt-4.1-nano") is False


def test_naive_persuader_message_supports_and_opposes() -> None:
    """Naive policy should emit stance-specific fixed templates."""
    supports_msg = naive_persuader_message(
        proposition="People should recycle",
        supports_proposition=True,
    )
    opposes_msg = naive_persuader_message(
        proposition="People should recycle",
        supports_proposition=False,
    )
    assert supports_msg == "This proposition is true: People should recycle."
    assert opposes_msg == "This proposition is false: People should recycle."


def test_naive_persuader_action_for_round() -> None:
    """Round helper should map round stance and proposition into a fixed action."""

    class DummyRound:
        """Minimal round stub for naive policy tests."""

        proposition = "The sky is blue"
        persuader_supports_proposition = True

    content, thought, reasoning = naive_persuader_action_for_round(DummyRound())
    assert content == "This proposition is true: The sky is blue."
    assert thought is None
    assert reasoning is None


def test_naive_persuader_action_uses_control_dialogue_proposition() -> None:
    """Round helper should prefer control proposition during control dialogues."""

    class DummyCondition:
        """Minimal condition stub exposing control-dialogue mode."""

        control_dialogue = True

    class DummyRound:
        """Round stub with both base and control propositions."""

        proposition = (
            "There should be mandatory quotas for women in leadership positions."
        )
        proposition_during_round = "Dogs are better than cats."
        persuader_supports_proposition = False

        @staticmethod
        def get_condition() -> DummyCondition:
            """Return control-dialogue condition."""
            return DummyCondition()

    content, thought, reasoning = naive_persuader_action_for_round(DummyRound())
    assert content == "This proposition is false: Dogs are better than cats."
    assert thought is None
    assert reasoning is None
