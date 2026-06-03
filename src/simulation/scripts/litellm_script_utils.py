"""
Shared LiteLLM script helpers for simulation CLI tools.
"""

from __future__ import annotations

import math
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

LITELLM_API_ERRORS = (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
)


def clean_json_response(content: str) -> str:
    """Strip optional markdown code fences from model output.

    Args:
        content: Raw model output text.

    Returns:
        Cleaned text suitable for JSON parsing.
    """
    clean = content.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def extract_content(response: Any) -> str:
    """Extract assistant message content from a LiteLLM response.

    Args:
        response: LiteLLM response object.

    Returns:
        Assistant text content, or empty string when unavailable.
    """
    choices = getattr(response, "choices", []) or []
    if not choices:
        return ""
    message_obj = getattr(choices[0], "message", None)
    if isinstance(message_obj, dict):
        return str(message_obj.get("content", "") or "")
    return str(getattr(message_obj, "content", "") or "")


def cost_for_tokens(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate USD cost from LiteLLM model pricing metadata.

    Args:
        model: LiteLLM model id.
        prompt_tokens: Prompt token total.
        completion_tokens: Completion token total.

    Returns:
        Estimated USD cost or ``None`` if pricing metadata is unavailable.
    """
    candidate_keys = [model]
    if "/" in model:
        candidate_keys.append(model.rsplit("/", maxsplit=1)[-1])
    model_info: dict[str, Any] | None = None
    for candidate in candidate_keys:
        info = litellm.model_cost.get(candidate)
        if isinstance(info, dict):
            model_info = info
            break
    if model_info is None:
        return None
    input_cost = model_info.get("input_cost_per_token")
    output_cost = model_info.get("output_cost_per_token")
    if not isinstance(input_cost, (float, int)) or not isinstance(
        output_cost, (float, int)
    ):
        return None
    return float(prompt_tokens) * float(input_cost) + float(completion_tokens) * float(
        output_cost
    )


def fallback_message_token_estimate(messages: list[dict[str, Any]]) -> int:
    """Estimate message token count with a simple character heuristic.

    Args:
        messages: Message payload.

    Returns:
        Approximate token count.
    """
    payload = str(messages)
    return max(1, math.ceil(len(payload) / 4))


def print_dry_run_header(model: str) -> None:
    """Print a consistent dry-run header.

    Args:
        model: Model identifier.
    """
    print("Dry run only. No API calls were made.")
    print("-" * 50)
    print(f"model={model}")


def print_estimated_cost_range(cost_min: float | None, cost_max: float | None) -> None:
    """Print a consistent estimated-cost line.

    Args:
        cost_min: Minimum estimated cost.
        cost_max: Maximum estimated cost.
    """
    if cost_min is not None and cost_max is not None:
        print(f"estimated_cost_usd_range=[{cost_min:.4f}, {cost_max:.4f}]")
    else:
        print("estimated_cost_usd_range=unavailable_for_model_pricing")


def estimate_cost_range_from_token_bounds(
    *,
    model: str,
    prompt_tokens_total: int,
    completion_tokens_min_total: int,
    completion_tokens_max_total: int,
) -> tuple[float | None, float | None]:
    """Estimate min/max USD cost from token bounds.

    Args:
        model: Model identifier.
        prompt_tokens_total: Prompt-token total.
        completion_tokens_min_total: Minimum completion-token total.
        completion_tokens_max_total: Maximum completion-token total.

    Returns:
        Tuple ``(cost_min, cost_max)`` with ``None`` for unavailable pricing.
    """
    cost_min = cost_for_tokens(
        model=model,
        prompt_tokens=prompt_tokens_total,
        completion_tokens=completion_tokens_min_total,
    )
    cost_max = cost_for_tokens(
        model=model,
        prompt_tokens=prompt_tokens_total,
        completion_tokens=completion_tokens_max_total,
    )
    return cost_min, cost_max


def begin_dry_run_report_with_cost_range(
    *,
    model: str,
    prompt_tokens_total: int,
    completion_tokens_min_total: int,
    completion_tokens_max_total: int,
) -> tuple[float | None, float | None]:
    """Print dry-run header and return estimated min/max cost range.

    Args:
        model: Model identifier.
        prompt_tokens_total: Prompt-token total.
        completion_tokens_min_total: Minimum completion-token total.
        completion_tokens_max_total: Maximum completion-token total.

    Returns:
        Tuple ``(cost_min, cost_max)``.
    """
    cost_min, cost_max = estimate_cost_range_from_token_bounds(
        model=model,
        prompt_tokens_total=prompt_tokens_total,
        completion_tokens_min_total=completion_tokens_min_total,
        completion_tokens_max_total=completion_tokens_max_total,
    )
    print_dry_run_header(model)
    return cost_min, cost_max
