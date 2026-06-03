"""Utilities for interacting with LLMs, replacing modelendpoints."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import litellm

from experiment.llm_batch import extract_text_from_response

logger = logging.getLogger(__name__)


def disable_litellm_logging() -> None:
    """Silence the default LiteLLM logger for cleaner CLI output.

    Returns:
        None.
    """
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.CRITICAL)
    litellm.set_verbose = False
    litellm.suppress_debug_info = True


def model_supports_reasoning(model: str) -> bool | None:
    """Return whether the model supports native reasoning controls.

    Args:
        model: LiteLLM model identifier.

    Returns:
        True when reasoning support is detected, False when explicit non-reasoning
        support is detected, or None when metadata is unavailable.
    """
    if not isinstance(model, str):
        return None
    normalized_model = model.strip()
    if not normalized_model:
        return None

    tail = normalized_model.rsplit("/", maxsplit=1)[-1]
    tail_lower = tail.lower()
    provider: str | None = None
    if "/" in normalized_model:
        provider = normalized_model.split("/", maxsplit=1)[0].strip().lower()

    candidates = [normalized_model, tail]
    if provider:
        candidates.append(f"{provider}/{tail}")

    for candidate in candidates:
        info = litellm.model_cost.get(candidate)
        if not isinstance(info, dict):
            continue
        supports_reasoning = info.get("supports_reasoning")
        if isinstance(supports_reasoning, bool):
            return supports_reasoning

    # Fallback for date-pinned GPT-5 aliases that may not be mapped yet.
    if tail_lower.startswith("gpt-5"):
        return True
    if "-20" in tail_lower and tail_lower.startswith("gpt-"):
        base_name = tail_lower.split("-20", maxsplit=1)[0]
        if base_name.startswith("gpt-5"):
            return True
        base_info = litellm.model_cost.get(base_name)
        if isinstance(base_info, dict):
            base_supports_reasoning = base_info.get("supports_reasoning")
            if isinstance(base_supports_reasoning, bool):
                return base_supports_reasoning
        if provider:
            provider_base_info = litellm.model_cost.get(f"{provider}/{base_name}")
            if isinstance(provider_base_info, dict):
                provider_base_supports_reasoning = provider_base_info.get(
                    "supports_reasoning"
                )
                if isinstance(provider_base_supports_reasoning, bool):
                    return provider_base_supports_reasoning
    return None


def atomizer_temperature_for_model(model: str) -> float | None:
    """Choose the atomizer temperature policy for one model.

    Args:
        model: LiteLLM model identifier used by the atomizer.

    Returns:
        ``None`` for reasoning-capable models, otherwise ``0.0``.
    """
    supports_reasoning = model_supports_reasoning(model)
    if supports_reasoning is True:
        return None
    return 0.0


COT_DELIMITER = "---"


def call_llm(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Call the model with the given messages and return extracted text and raw response.

    Args:
        model: Model identifier passed to LiteLLM.
        messages: Chat messages for the completion call.
        **kwargs: Additional keyword arguments forwarded to LiteLLM.
            Use `num_retries` to enable LiteLLM retries.

    Returns:
        Dictionary with keys `text` and `raw_response`.
    """
    if "num_retries" not in kwargs:
        kwargs["num_retries"] = 2

    try:
        response = litellm.completion(model=model, messages=messages, **kwargs)
    except (litellm.OpenAIError, OSError, ValueError) as exc:
        logger.warning("LiteLLM call failed for model=%s: %s", model, exc)
        raise ValueError(f"LLM call failed for model {model}: {exc}") from exc

    text = extract_text_from_response(response)
    return {"text": text, "raw_response": response}


def split_thought_from_response(text: str) -> Tuple[Optional[str], str]:
    """Split a model's thought from its final response.

    Args:
        text: Response text that may contain a `<thought>` block.

    Returns:
        Tuple of `(thought, response)` where thought may be None.
    """
    if not text:
        return None, ""

    thought_start_tag = "<thought>"
    thought_end_tag = "</thought>"

    if thought_start_tag in text and thought_end_tag in text:
        start_idx = text.find(thought_start_tag) + len(thought_start_tag)
        end_idx = text.find(thought_end_tag)
        thought = text[start_idx:end_idx].strip()
        response = text[end_idx + len(thought_end_tag) :].strip()
        return thought, response

    return None, text


def split_reasoning_from_response(text: str) -> Tuple[Optional[str], str]:
    """Split a model's reasoning trace from its final response.

    Args:
        text: Response text that may contain a `<reasoning>` block.

    Returns:
        Tuple of `(reasoning, response)` where reasoning may be None.
    """
    if not text:
        return None, ""

    reasoning_start_tag = "<reasoning>"
    reasoning_end_tag = "</reasoning>"

    if reasoning_start_tag in text and reasoning_end_tag in text:
        start_idx = text.find(reasoning_start_tag) + len(reasoning_start_tag)
        end_idx = text.find(reasoning_end_tag)
        reasoning = text[start_idx:end_idx].strip()
        response = text[end_idx + len(reasoning_end_tag) :].strip()
        return reasoning, response

    return None, text


def convert_roles(
    messages: List[Dict[str, str]], conversion: Dict[str, str]
) -> List[Dict[str, str]]:
    """Convert message roles according to a conversion dictionary.

    Args:
        messages: Message dictionaries containing a `role` key.
        conversion: Mapping from original role to replacement role.

    Returns:
        New list of messages with updated roles.
    """
    new_messages: List[Dict[str, str]] = []
    for message in messages:
        new_message = message.copy()
        if "role" in message:
            new_message["role"] = conversion.get(message["role"], message["role"])
        new_messages.append(new_message)
    return new_messages
