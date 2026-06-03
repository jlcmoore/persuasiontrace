"""LiteLLM batch helpers for chat completions (main package).

Provides a thin wrapper to submit batches of chat.messages and a helper to
extract response text in common formats.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional, Sequence, Tuple

import litellm


def _disable_thinking_extra_body_for_model(model: str) -> dict[str, Any] | None:
    """
    Return provider-specific request body to disable reasoning/thinking mode.

    Args:
        model: LiteLLM model string.

    Returns:
        Extra request body dictionary or None.
    """
    if model.startswith("together_ai/Qwen/Qwen3.5-"):
        # Together Qwen3.5 defaults to thinking mode and can emit only
        # reasoning_content with empty content; disable thinking so content
        # carries the assistant message for downstream logging/training.
        # We intentionally do not rely on litellm.supports_reasoning() here;
        # as of now it returns False for this model/provider pair.
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def _supports_reasoning_effort_param(model: str) -> bool:
    """
    Check whether the model/provider pair accepts `reasoning_effort`.

    Args:
        model: LiteLLM model string.

    Returns:
        True if reasoning_effort is supported.
    """
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model)
        supported_params = litellm.get_supported_openai_params(
            model=resolved_model,
            custom_llm_provider=provider,
        )
    except (litellm.BadRequestError, TypeError, ValueError, KeyError):
        return False
    if not supported_params:
        return False
    return "reasoning_effort" in supported_params


def _is_gpt5_family_model(model: str) -> bool:
    """
    Return True when the resolved model belongs to the GPT-5 family.

    Args:
        model: LiteLLM model string.

    Returns:
        True for model names beginning with `gpt-5`.
    """
    resolved_model = model
    try:
        provider_model, _, _, _ = litellm.get_llm_provider(model=model)
        if isinstance(provider_model, str) and provider_model.strip():
            resolved_model = provider_model
    except (litellm.BadRequestError, TypeError, ValueError, KeyError):
        if "/" in model:
            resolved_model = model.rsplit("/", maxsplit=1)[-1]
    return str(resolved_model).startswith("gpt-5")


def _temperature_for_model(model: str, requested_temperature: float | None) -> float:
    """
    Resolve model-specific temperature defaults and constraints.

    Args:
        model: LiteLLM model string.
        requested_temperature: Optional user-provided temperature.

    Returns:
        Temperature to send to LiteLLM.
    """
    if _is_gpt5_family_model(model):
        return 1.0
    if requested_temperature is not None:
        return float(requested_temperature)
    if litellm.supports_reasoning(model):
        return 1.0
    return 0.0


def _default_reasoning_effort_for_model(model: str) -> str:
    """
    Resolve a safe default reasoning effort for models that accept the field.

    Args:
        model: LiteLLM model string.

    Returns:
        Reasoning effort value to use when disabling reasoning by default.
    """
    resolved_model = model
    provider_name = ""
    try:
        provider_model, provider, _, _ = litellm.get_llm_provider(model=model)
        if isinstance(provider_model, str) and provider_model.strip():
            resolved_model = provider_model
        if isinstance(provider, str):
            provider_name = provider.strip().lower()
    except (litellm.BadRequestError, TypeError, ValueError, KeyError):
        if "/" in model:
            provider_name = model.split("/", maxsplit=1)[0].strip().lower()
            resolved_model = model.rsplit("/", maxsplit=1)[-1]

    # Anthropic mapping in current LiteLLM versions can fail for
    # reasoning_effort='none'; use the lightest reliable setting.
    if provider_name == "anthropic":
        return "minimal"

    # Some GPT-5 variants reject reasoning_effort='none';
    # use the lightest supported value for those specific variants.
    if (
        resolved_model == "gpt-5"
        or resolved_model.startswith("gpt-5-")
        or resolved_model.startswith("gpt-5-mini")
        or resolved_model.startswith("gpt-5-nano")
    ):
        return "minimal"
    return "none"


def _reasoning_effort_override_for_model(
    model: str,
    *,
    disable_reasoning: bool,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """
    Return reasoning-effort overrides for supported reasoning models.

    Args:
        model: LiteLLM model string.
        disable_reasoning: Whether to default reasoning-capable models to no reasoning.
        reasoning_effort: Optional explicit reasoning-effort override.

    Returns:
        A dictionary that may include `reasoning_effort`.
    """
    if not litellm.supports_reasoning(model):
        return {}
    if not _supports_reasoning_effort_param(model):
        return {}
    if reasoning_effort is not None:
        return {"reasoning_effort": str(reasoning_effort)}
    if disable_reasoning:
        return {"reasoning_effort": _default_reasoning_effort_for_model(model)}
    return {}


def _sanitize_messages_for_model(
    model: str,
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize batched chat messages for provider-specific validation quirks.

    Args:
        model: LiteLLM model string.
        messages: One chat-message sequence.

    Returns:
        Sanitized message list preserving semantic content.
    """
    sanitized = list(messages)
    requires_non_empty_content = model.startswith("together_ai/") or model.startswith(
        "anthropic/"
    )
    if not requires_non_empty_content:
        return sanitized

    filtered: list[dict[str, Any]] = []
    for message in sanitized:
        content = message.get("content")
        if isinstance(content, str) and not content.strip():
            continue
        filtered.append(message)
    if filtered:
        return filtered
    return sanitized


def batch_chat(  # pylint: disable=too-many-arguments
    *,
    model: str | Sequence[str],
    messages_list: Sequence[list[dict]],
    timeout: int | None = None,
    max_tokens: int | None = None,
    max_workers: int | None = None,
    temperature: float | Sequence[float] | None = None,
    num_retries: int | None = None,
    disable_reasoning: bool = True,
    reasoning_effort: str | None = None,
) -> List[dict]:
    """Submit a batch of chat messages and return raw responses.

    Supports either a single model string or a sequence of model strings
    of the same length as messages_list. If multiple models are provided,
    requests are grouped by model and executed in parallel.

    Returns one response per input item, preserving order.
    """
    if isinstance(model, str):
        # Single model case: use native litellm.batch_completion
        kwargs: dict = {
            "model": model,
            "messages": [
                _sanitize_messages_for_model(model, messages)
                for messages in messages_list
            ],
        }
        kwargs["temperature"] = _temperature_for_model(model, temperature)

        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(
            _reasoning_effort_override_for_model(
                model,
                disable_reasoning=disable_reasoning,
                reasoning_effort=reasoning_effort,
            )
        )
        extra_body = _disable_thinking_extra_body_for_model(model)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        if max_workers is not None:
            kwargs["max_workers"] = max_workers
        if num_retries is not None:
            kwargs["num_retries"] = num_retries

        responses = litellm.batch_completion(**kwargs)
        if hasattr(responses, "__iter__") and not isinstance(responses, list):
            responses = list(responses)
        return responses

    # Multiple models case: group by model and run in parallel
    models = list(model)
    if len(models) != len(messages_list):
        raise ValueError("Length of models must match length of messages_list.")

    # Compute temperatures
    if temperature is None:
        temps = [_temperature_for_model(m, None) for m in models]
    elif isinstance(temperature, (int, float)):
        temps = [_temperature_for_model(m, float(temperature)) for m in models]
    else:
        temps = list(temperature)
        if len(temps) != len(models):
            raise ValueError("Length of temperatures must match length of models.")
        temps = [
            _temperature_for_model(model_name, float(temp))
            for model_name, temp in zip(models, temps)
        ]

    # Group by model
    # model_groups: model_name -> list of (original_index, messages, temperature)
    model_groups: Dict[str, List[Tuple[int, list[dict], float]]] = {}
    for i, (m, msgs, t) in enumerate(zip(models, messages_list, temps)):
        sanitized_messages = _sanitize_messages_for_model(m, msgs)
        model_groups.setdefault(m, []).append((i, sanitized_messages, t))

    for model_name, group in model_groups.items():
        group_temps = {temp for _, _, temp in group}
        if len(group_temps) > 1:
            raise ValueError(
                "Found multiple temperatures for model "
                f"{model_name}: {sorted(group_temps)}"
            )

    results: List[Optional[dict]] = [None] * len(messages_list)

    executor_workers = len(model_groups)
    if max_workers is not None:
        executor_workers = min(executor_workers, max_workers)
    per_model_workers = None
    if max_workers is not None:
        per_model_workers = max(1, max_workers // max(1, executor_workers))

    def run_model_batch(model_name: str, group: List[Tuple[int, list[dict], float]]):
        indices, group_messages, group_temps = zip(*group)
        # Note: we use the first temperature in the group for the whole batch_completion call
        # because litellm.batch_completion doesn't take a list of temperatures either.
        # This is usually fine as same models should have same temperature.
        kwargs: dict = {
            "model": model_name,
            "messages": list(group_messages),
            "temperature": group_temps[0],
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(
            _reasoning_effort_override_for_model(
                model_name,
                disable_reasoning=disable_reasoning,
                reasoning_effort=reasoning_effort,
            )
        )
        extra_body = _disable_thinking_extra_body_for_model(model_name)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        if per_model_workers is not None:
            kwargs["max_workers"] = per_model_workers
        if num_retries is not None:
            kwargs["num_retries"] = num_retries

        batch_responses = litellm.batch_completion(**kwargs)
        if len(batch_responses) != len(indices):
            raise RuntimeError(
                "LiteLLM returned a different number of responses than requested "
                f"for model {model_name}."
            )
        for idx, resp in zip(indices, batch_responses):
            results[idx] = resp

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=executor_workers
    ) as executor:
        futures = [
            executor.submit(run_model_batch, m_name, group)
            for m_name, group in model_groups.items()
        ]
        concurrent.futures.wait(futures)
        for future in futures:
            future.result()

    # Check for failures (None in results)
    if any(r is None for r in results):
        raise RuntimeError("Some batch requests failed to return a response.")

    return list(results)  # type: ignore


def extract_text_from_response(resp: dict) -> str:
    """Extract text content from a LiteLLM response dict."""
    # Try OpenAI-like: choices[0].message.content
    choices = resp.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
    # Fallbacks
    txt = resp.get("text") or resp.get("content")
    return txt or ""
