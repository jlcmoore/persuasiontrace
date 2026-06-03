"""
Utilities for non-LLM persuader policy variants.
"""

from __future__ import annotations

NAIVE_PERSUADER_MODEL_ALIASES: frozenset[str] = frozenset({"naive"})


def is_naive_persuader_model(model: str | None) -> bool:
    """
    Determine whether a model identifier selects the naive persuader policy.

    Args:
        model: Persuader model identifier.

    Returns:
        True when the model id maps to the naive policy.
    """
    if not isinstance(model, str):
        return False
    normalized = model.strip().lower()
    return normalized in NAIVE_PERSUADER_MODEL_ALIASES


def naive_persuader_message(
    *,
    proposition: str,
    supports_proposition: bool | None,
) -> str:
    """
    Build the deterministic naive persuader message for one round turn.

    Args:
        proposition: Proposition text.
        supports_proposition: Persuader stance for the proposition.

    Returns:
        One fixed-format persuader message.
    """
    proposition_text = str(proposition).strip() or "the proposition"
    if proposition_text.endswith((".", "!", "?")):
        proposition_text = proposition_text[:-1].rstrip()
    if supports_proposition is True:
        return f"This proposition is true: {proposition_text}."
    if supports_proposition is False:
        return f"This proposition is false: {proposition_text}."
    return f"This proposition is: {proposition_text}."


def naive_persuader_action_for_round(round_obj) -> tuple[str, None, None]:
    """
    Build a naive persuader action tuple for one round object.

    Args:
        round_obj: Round-like object with `proposition` and
            `persuader_supports_proposition` attributes.

    Returns:
        Tuple of `(content, thought, reasoning_trace)`.
    """
    proposition = str(getattr(round_obj, "proposition", "") or "")
    condition = None
    get_condition = getattr(round_obj, "get_condition", None)
    if callable(get_condition):
        condition = get_condition()
    else:
        condition = getattr(round_obj, "condition", None)

    if bool(getattr(condition, "control_dialogue", False)):
        proposition_during_round = getattr(round_obj, "proposition_during_round", None)
        if (
            isinstance(proposition_during_round, str)
            and proposition_during_round.strip()
        ):
            proposition = proposition_during_round

    supports_raw = getattr(round_obj, "persuader_supports_proposition", None)
    supports_proposition: bool | None = (
        bool(supports_raw) if isinstance(supports_raw, bool) else None
    )
    content = naive_persuader_message(
        proposition=proposition,
        supports_proposition=supports_proposition,
    )
    return content, None, None
