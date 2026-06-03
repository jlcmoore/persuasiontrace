"""Canonical styling for simulator-vs-human comparison corpora."""

from __future__ import annotations

import textwrap

CORE_COMPARISON_CORPUS_ORDER = [
    "human_reference",
    "vanilla_llm_target",
    "structure_target",
    "full_simulated_target",
]
COMPARISON_CORPUS_ORDER_WITH_NO_RHETORIC = [
    *CORE_COMPARISON_CORPUS_ORDER,
    "full_no_rhetoric_target",
]
COMPARISON_CORPUS_LABEL_MAP: dict[str, str] = {
    "human_reference": "Human",
    "vanilla_llm_target": "Unstructured LLM",
    "structure_target": "Structured LLM",
    "full_simulated_target": "BN Target (ours)",
    "full_no_rhetoric_target": "BN Target (No Rhetoric)",
}
COMPARISON_CORPUS_TICK_LABEL_MAP: dict[str, str] = {
    "human_reference": "Human",
    "vanilla_llm_target": "Unstruct.\nLLM",
    "structure_target": "Struct.\nLLM",
    "full_simulated_target": "BN\n(ours)",
}
COMPARISON_CORPUS_COLOR_MAP: dict[str, str] = {
    "human_reference": "#111111",
    "vanilla_llm_target": "#2ca02c",
    "structure_target": "#1f77b4",
    "full_simulated_target": "#d62728",
    "full_no_rhetoric_target": "#ff7f0e",
}
PAPER_SQUARE_FIGURE_SIZE_INCHES = (3.0, 3.0)
PAPER_RESULTS_FIGURE_SIZE_INCHES = (3.0, 2.0)
PERSONA_MARKER = "__persona="
PERSONA_AVERAGE_SUFFIX = "logos_pathos_ethos_average"
PERSONA_DISPLAY_MAP: dict[str, str] = {
    "logical": "logos",
    "emotional": "pathos",
    "authoritarian": "ethos",
}
COMPARISON_CORPUS_PREFIX_TO_BASE: tuple[tuple[str, str], ...] = (
    ("human_reference__", "human_reference"),
    ("vanilla_llm_target__", "vanilla_llm_target"),
    ("structure_target__", "structure_target"),
    ("full_simulated_target__", "full_simulated_target"),
    ("full_no_rhetoric_target__", "full_no_rhetoric_target"),
)


def comparison_corpus_base_key(corpus_name: str) -> str:
    """Resolve a possibly-parameterized corpus key to its canonical base key.

    Args:
        corpus_name: Raw or canonical corpus key.

    Returns:
        Canonical corpus key when recognizable, otherwise the original value.
    """
    base_key = corpus_name
    if corpus_name not in COMPARISON_CORPUS_ORDER_WITH_NO_RHETORIC:
        for prefix, canonical in COMPARISON_CORPUS_PREFIX_TO_BASE:
            if corpus_name.startswith(prefix):
                base_key = canonical
                break
    return base_key


def comparison_corpus_label(corpus_name: str) -> str:
    """Build a canonical human-readable label for one comparison corpus.

    Args:
        corpus_name: Raw or canonical corpus key.

    Returns:
        Display label used across tables and non-compact figure labels.
    """
    base_key = comparison_corpus_base_key(corpus_name)
    if base_key == "full_simulated_target" and PERSONA_MARKER in corpus_name:
        persona_name = corpus_name.split(PERSONA_MARKER, maxsplit=1)[1]
        if persona_name == PERSONA_AVERAGE_SUFFIX:
            return "BN Target (ours, avg)"
        persona_display = PERSONA_DISPLAY_MAP.get(persona_name, persona_name)
        return f"BN Target (ours, {persona_display})"
    return COMPARISON_CORPUS_LABEL_MAP.get(base_key, corpus_name)


def comparison_corpus_sort_key(corpus_name: str) -> tuple[int, str]:
    """Sort corpus keys by canonical comparison order.

    Args:
        corpus_name: Corpus key to rank.

    Returns:
        Tuple used for stable sorting.
    """
    base_key = comparison_corpus_base_key(corpus_name)
    if base_key in COMPARISON_CORPUS_ORDER_WITH_NO_RHETORIC:
        return (
            COMPARISON_CORPUS_ORDER_WITH_NO_RHETORIC.index(base_key),
            corpus_name,
        )
    return (len(COMPARISON_CORPUS_ORDER_WITH_NO_RHETORIC), corpus_name)


def comparison_corpus_tick_label(corpus_name: str) -> str:
    """Build a compact x-axis tick label for one comparison corpus.

    Args:
        corpus_name: Canonical corpus key.

    Returns:
        Tick label text with explicit line breaks for compact square plots.
    """
    base_key = comparison_corpus_base_key(corpus_name)
    if base_key == "full_simulated_target" and PERSONA_MARKER in corpus_name:
        persona_name = corpus_name.split(PERSONA_MARKER, maxsplit=1)[1]
        if persona_name == PERSONA_AVERAGE_SUFFIX:
            return "BN\n(avg)"
        persona_display = PERSONA_DISPLAY_MAP.get(persona_name, persona_name)
        return textwrap.fill(
            f"BN ({persona_display})",
            width=8,
            break_long_words=False,
            break_on_hyphens=False,
        )
    mapped = COMPARISON_CORPUS_TICK_LABEL_MAP.get(base_key)
    if mapped is not None:
        return mapped
    default_label = comparison_corpus_label(corpus_name)
    return textwrap.fill(
        default_label,
        width=12,
        break_long_words=False,
        break_on_hyphens=False,
    )


def comparison_corpus_is_visible_in_main_plots(corpus_name: str) -> bool:
    """Return whether a corpus should appear in the main comparison figures.

    Args:
        corpus_name: Canonical corpus key.

    Returns:
        ``True`` when the corpus should be plotted, otherwise ``False``.
    """
    return corpus_name != "full_no_rhetoric_target"
