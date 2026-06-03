"""Unit tests for experiment.llm_batch."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiment.llm_batch import batch_chat


class TestBatchChat(unittest.TestCase):
    """Tests for batch chat completion helpers."""

    def test_together_qwen35_disables_thinking(self):
        """Together Qwen3.5 calls should include enable_thinking=False."""

        captured: dict[str, object] = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            return [{"choices": [{"message": {"content": "ok"}}]}]

        with patch(
            "experiment.llm_batch.litellm.supports_reasoning", return_value=False
        ):
            with patch(
                "experiment.llm_batch.litellm.batch_completion",
                side_effect=fake_batch_completion,
            ):
                responses = batch_chat(
                    model="together_ai/Qwen/Qwen3.5-9B",
                    messages_list=[[{"role": "user", "content": "hi"}]],
                    max_tokens=16,
                )

        self.assertEqual(len(responses), 1)
        self.assertEqual(
            captured.get("extra_body"),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_non_qwen_model_has_no_extra_body_override(self):
        """Non-Together-Qwen models should not receive Qwen extra_body."""

        captured: dict[str, object] = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            return [{"choices": [{"message": {"content": "ok"}}]}]

        with patch(
            "experiment.llm_batch.litellm.supports_reasoning", return_value=False
        ):
            with patch(
                "experiment.llm_batch.litellm.batch_completion",
                side_effect=fake_batch_completion,
            ):
                responses = batch_chat(
                    model="openai/gpt-4.1-nano",
                    messages_list=[[{"role": "user", "content": "hi"}]],
                    max_tokens=16,
                )

        self.assertEqual(len(responses), 1)
        self.assertNotIn("extra_body", captured)

    def test_reasoning_models_set_reasoning_effort_none_when_supported(self):
        """Reasoning models should default to reasoning_effort='none'."""
        captured: dict[str, object] = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            return [{"choices": [{"message": {"content": "ok"}}]}]

        with patch(
            "experiment.llm_batch.litellm.supports_reasoning", return_value=True
        ):
            with patch(
                "experiment.llm_batch.litellm.get_llm_provider",
                return_value=("gpt-5.1", "openai", None, None),
            ):
                with patch(
                    "experiment.llm_batch.litellm.get_supported_openai_params",
                    return_value=["reasoning_effort", "temperature"],
                ):
                    with patch(
                        "experiment.llm_batch.litellm.batch_completion",
                        side_effect=fake_batch_completion,
                    ):
                        responses = batch_chat(
                            model="openai/o3",
                            messages_list=[[{"role": "user", "content": "hi"}]],
                            max_tokens=16,
                        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(captured.get("reasoning_effort"), "none")

    def test_disable_reasoning_false_allows_reasoning_model_default_behavior(self):
        """Explicit override should allow reasoning without forced effort=none."""
        captured: dict[str, object] = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            return [{"choices": [{"message": {"content": "ok"}}]}]

        with patch(
            "experiment.llm_batch.litellm.supports_reasoning", return_value=True
        ):
            with patch(
                "experiment.llm_batch.litellm.get_llm_provider",
                return_value=("o3", "openai", None, None),
            ):
                with patch(
                    "experiment.llm_batch.litellm.get_supported_openai_params",
                    return_value=["reasoning_effort", "temperature"],
                ):
                    with patch(
                        "experiment.llm_batch.litellm.batch_completion",
                        side_effect=fake_batch_completion,
                    ):
                        responses = batch_chat(
                            model="openai/o3",
                            messages_list=[[{"role": "user", "content": "hi"}]],
                            max_tokens=16,
                            disable_reasoning=False,
                        )

        self.assertEqual(len(responses), 1)
        self.assertNotIn("reasoning_effort", captured)

    def test_reasoning_effort_override_applies_when_supported(self):
        """Explicit reasoning_effort should override the default no-reasoning mode."""
        captured: dict[str, object] = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            return [{"choices": [{"message": {"content": "ok"}}]}]

        with patch(
            "experiment.llm_batch.litellm.supports_reasoning", return_value=True
        ):
            with patch(
                "experiment.llm_batch.litellm.get_llm_provider",
                return_value=("gpt-5.1", "openai", None, None),
            ):
                with patch(
                    "experiment.llm_batch.litellm.get_supported_openai_params",
                    return_value=["reasoning_effort", "temperature"],
                ):
                    with patch(
                        "experiment.llm_batch.litellm.batch_completion",
                        side_effect=fake_batch_completion,
                    ):
                        responses = batch_chat(
                            model="openai/gpt-5.1",
                            messages_list=[[{"role": "user", "content": "hi"}]],
                            max_tokens=16,
                            reasoning_effort="low",
                        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(captured.get("reasoning_effort"), "low")

    def test_gpt5_nano_forces_temperature_one(self):
        """GPT-5 family models should always use temperature=1."""
        captured: dict[str, object] = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            return [{"choices": [{"message": {"content": "ok"}}]}]

        with patch(
            "experiment.llm_batch.litellm.get_llm_provider",
            return_value=("gpt-5.4-nano", "openai", None, None),
        ):
            with patch(
                "experiment.llm_batch.litellm.supports_reasoning", return_value=False
            ):
                with patch(
                    "experiment.llm_batch.litellm.batch_completion",
                    side_effect=fake_batch_completion,
                ):
                    responses = batch_chat(
                        model="openai/gpt-5.4-nano",
                        messages_list=[[{"role": "user", "content": "hi"}]],
                        temperature=0,
                        max_tokens=16,
                    )

        self.assertEqual(len(responses), 1)
        self.assertEqual(captured.get("temperature"), 1.0)

    def test_multi_model_gpt5_temperature_override(self):
        """Multi-model calls should force GPT-5 temps to 1 while preserving others."""
        captured_calls: list[dict[str, object]] = []

        def fake_batch_completion(**kwargs):
            captured_calls.append(dict(kwargs))
            messages = kwargs.get("messages", [])
            return [{"choices": [{"message": {"content": "ok"}}]} for _ in messages]

        def fake_get_provider(model):
            if model == "openai/gpt-5.4-nano":
                return ("gpt-5.4-nano", "openai", None, None)
            return ("gpt-4.1-nano", "openai", None, None)

        with patch(
            "experiment.llm_batch.litellm.get_llm_provider",
            side_effect=fake_get_provider,
        ):
            with patch(
                "experiment.llm_batch.litellm.batch_completion",
                side_effect=fake_batch_completion,
            ):
                with patch(
                    "experiment.llm_batch.litellm.supports_reasoning",
                    return_value=False,
                ):
                    responses = batch_chat(
                        model=["openai/gpt-5.4-nano", "openai/gpt-4.1-nano"],
                        messages_list=[
                            [{"role": "user", "content": "a"}],
                            [{"role": "user", "content": "b"}],
                        ],
                        temperature=0,
                        max_tokens=16,
                    )

        self.assertEqual(len(responses), 2)
        temperatures_by_model = {
            str(call["model"]): float(call["temperature"]) for call in captured_calls
        }
        self.assertEqual(temperatures_by_model["openai/gpt-5.4-nano"], 1.0)
        self.assertEqual(temperatures_by_model["openai/gpt-4.1-nano"], 0.0)

    def test_gpt5_mini_uses_minimal_reasoning_effort(self):
        """GPT-5 mini/nano variants should use reasoning_effort='minimal'."""
        captured: dict[str, object] = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            return [{"choices": [{"message": {"content": "ok"}}]}]

        with patch(
            "experiment.llm_batch.litellm.supports_reasoning", return_value=True
        ):
            with patch(
                "experiment.llm_batch.litellm.get_llm_provider",
                return_value=("gpt-5-mini", "openai", None, None),
            ):
                with patch(
                    "experiment.llm_batch.litellm.get_supported_openai_params",
                    return_value=["reasoning_effort", "temperature"],
                ):
                    with patch(
                        "experiment.llm_batch.litellm.batch_completion",
                        side_effect=fake_batch_completion,
                    ):
                        responses = batch_chat(
                            model="openai/gpt-5-mini",
                            messages_list=[[{"role": "user", "content": "hi"}]],
                            max_tokens=16,
                        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(captured.get("reasoning_effort"), "minimal")


if __name__ == "__main__":
    unittest.main()
