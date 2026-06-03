"""Corpus-variant parsing and fan-corpus expansion helpers."""

from __future__ import annotations

from collections import defaultdict

from simulation.human_likeness import RoundTrajectory

POLICY_CORPUS_SEPARATOR = "__policy="
PERSONA_CORPUS_SEPARATOR = "__persona="
POLICY_INSTRUCTION_CORPUS_SEPARATOR = "__instruction="
SIMULATOR_MODEL_CORPUS_SEPARATOR = "__sim="


def _split_policy_variant_corpus(corpus: str) -> tuple[str, str | None]:
    """
    Split a corpus key into base key and optional policy-model suffix.

    Args:
        corpus: Corpus key, optionally suffixed with a policy model.

    Returns:
        Tuple of (base_corpus, policy_model_or_none).
    """
    if POLICY_CORPUS_SEPARATOR not in corpus:
        return corpus, None
    base_corpus, policy_model = corpus.rsplit(POLICY_CORPUS_SEPARATOR, maxsplit=1)
    if not base_corpus or not policy_model:
        return corpus, None
    return base_corpus, policy_model


def _join_policy_variant_corpus(*, base_corpus: str, policy_model: str) -> str:
    """
    Build a corpus key with an explicit policy-model suffix.

    Args:
        base_corpus: Base corpus key.
        policy_model: Persuader model string.

    Returns:
        Suffixed corpus key.
    """
    return f"{base_corpus}{POLICY_CORPUS_SEPARATOR}{policy_model}"


def _split_persona_variant_corpus(corpus: str) -> tuple[str, str | None]:
    """
    Split a corpus key into base key and optional persona suffix.

    Args:
        corpus: Corpus key, optionally suffixed with a persona.

    Returns:
        Tuple of (base_corpus, persona_or_none).
    """
    if PERSONA_CORPUS_SEPARATOR not in corpus:
        return corpus, None
    base_corpus, persona = corpus.rsplit(PERSONA_CORPUS_SEPARATOR, maxsplit=1)
    if not base_corpus or not persona:
        return corpus, None
    return base_corpus, persona


def _split_simulator_model_variant_corpus(corpus: str) -> tuple[str, str | None]:
    """
    Split a corpus key into base key and optional simulator-model suffix.

    Args:
        corpus: Corpus key, optionally suffixed with a simulator model.

    Returns:
        Tuple of (base_corpus, simulator_model_or_none).
    """
    if SIMULATOR_MODEL_CORPUS_SEPARATOR not in corpus:
        return corpus, None
    base_corpus, simulator_model = corpus.rsplit(
        SIMULATOR_MODEL_CORPUS_SEPARATOR, maxsplit=1
    )
    if not base_corpus or not simulator_model:
        return corpus, None
    return base_corpus, simulator_model


def _split_policy_instruction_variant_corpus(corpus: str) -> tuple[str, str | None]:
    """
    Split a corpus key into base key and optional policy-instruction suffix.

    Args:
        corpus: Corpus key, optionally suffixed with instruction variant.

    Returns:
        Tuple of (base_corpus, instruction_variant_or_none).
    """
    if POLICY_INSTRUCTION_CORPUS_SEPARATOR not in corpus:
        return corpus, None
    base_corpus, instruction_variant = corpus.rsplit(
        POLICY_INSTRUCTION_CORPUS_SEPARATOR, maxsplit=1
    )
    if not base_corpus or not instruction_variant:
        return corpus, None
    return base_corpus, instruction_variant


def _join_persona_variant_corpus(*, base_corpus: str, persona: str) -> str:
    """
    Build a corpus key with an explicit persona suffix.

    Args:
        base_corpus: Base corpus key.
        persona: Simulated target persona.

    Returns:
        Suffixed corpus key.
    """
    return f"{base_corpus}{PERSONA_CORPUS_SEPARATOR}{persona}"


def _join_simulator_model_variant_corpus(
    *,
    base_corpus: str,
    simulator_model: str,
) -> str:
    """
    Build a corpus key with an explicit simulator-model suffix.

    Args:
        base_corpus: Base corpus key.
        simulator_model: Simulator model string.

    Returns:
        Suffixed corpus key.
    """
    return f"{base_corpus}{SIMULATOR_MODEL_CORPUS_SEPARATOR}{simulator_model}"


def _join_policy_instruction_variant_corpus(
    *,
    base_corpus: str,
    instruction_variant: str,
) -> str:
    """
    Build a corpus key with an explicit policy-instruction suffix.

    Args:
        base_corpus: Base corpus key.
        instruction_variant: Instruction variant label (on/off/unknown).

    Returns:
        Suffixed corpus key.
    """
    return f"{base_corpus}{POLICY_INSTRUCTION_CORPUS_SEPARATOR}{instruction_variant}"


def _simulator_model_for_row(row: RoundTrajectory) -> str:
    """
    Resolve the simulator-side model identifier for one trajectory row.

    Args:
        row: Round trajectory row.

    Returns:
        Model identifier string, or ``unknown`` when unavailable.
    """
    simulated_model = row.condition.roles.simulated_target
    if isinstance(simulated_model, str) and simulated_model.strip():
        return simulated_model.strip()
    llm_target_model = row.condition.roles.llm_target
    if isinstance(llm_target_model, str) and llm_target_model.strip():
        return llm_target_model.strip()
    return "unknown"


def _corpora_with_simulator_model_variants_for_fan(
    corpora: list[tuple[str, list[RoundTrajectory]]],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """
    Split fan-plot corpora by simulator model when multiple models are present.

    Args:
        corpora: Base corpora as (corpus_name, rows) tuples.

    Returns:
        Expanded corpora list with simulator-model-specific corpus names when
        a corpus mixes multiple simulator models.
    """
    expanded: list[tuple[str, list[RoundTrajectory]]] = []
    for corpus, rows in corpora:
        if corpus == "human_reference" or not rows:
            expanded.append((corpus, rows))
            continue

        grouped: dict[str, list[RoundTrajectory]] = defaultdict(list)
        for row in rows:
            grouped[_simulator_model_for_row(row)].append(row)

        if len(grouped) <= 1:
            expanded.append((corpus, rows))
            continue

        for simulator_model in sorted(grouped):
            expanded.append(
                (
                    _join_simulator_model_variant_corpus(
                        base_corpus=corpus,
                        simulator_model=simulator_model,
                    ),
                    grouped[simulator_model],
                )
            )
    return expanded


def _corpora_with_policy_variants_for_fan(
    corpora: list[tuple[str, list[RoundTrajectory]]],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """
    Split fan-plot corpora by persuader model when multiple models are present.

    Args:
        corpora: Base corpora as (corpus_name, rows) tuples.

    Returns:
        Expanded corpora list with policy-model-specific corpus names when
        a corpus mixes multiple persuader models.
    """
    expanded: list[tuple[str, list[RoundTrajectory]]] = []
    for corpus, rows in corpora:
        if corpus == "human_reference" or not rows:
            expanded.append((corpus, rows))
            continue

        grouped: dict[str, list[RoundTrajectory]] = {}
        for row in rows:
            raw_model = row.condition.roles.llm_persuader
            if isinstance(raw_model, str) and raw_model.strip():
                model = raw_model.strip()
            else:
                model = "unknown"
            grouped.setdefault(model, []).append(row)

        if len(grouped) <= 1:
            expanded.append((corpus, rows))
            continue

        for model in sorted(grouped):
            expanded.append(
                (
                    _join_policy_variant_corpus(
                        base_corpus=corpus,
                        policy_model=model,
                    ),
                    grouped[model],
                )
            )
    return expanded


def _instruction_variant_for_row(row: RoundTrajectory) -> str:
    """
    Resolve policy-instruction variant label for one trajectory row.

    Args:
        row: Round trajectory row.

    Returns:
        One of ``on``, ``off``, or ``unknown``.
    """
    trace_payload = row.round_obj.simulated_target_trace
    if not isinstance(trace_payload, dict):
        return "unknown"
    enabled_raw = trace_payload.get("extra_policy_instruction_enabled")
    if isinstance(enabled_raw, bool):
        return "on" if enabled_raw else "off"
    return "unknown"


def _corpora_with_instruction_variants_for_fan(
    corpora: list[tuple[str, list[RoundTrajectory]]],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """
    Split fan-plot corpora by policy-instruction variant when mixed.

    Args:
        corpora: Base corpora as (corpus_name, rows) tuples.

    Returns:
        Expanded corpora list with instruction-variant corpus names when
        a corpus contains multiple instruction variants.
    """
    expanded: list[tuple[str, list[RoundTrajectory]]] = []
    for corpus, rows in corpora:
        if corpus == "human_reference" or not rows:
            expanded.append((corpus, rows))
            continue

        grouped: dict[str, list[RoundTrajectory]] = defaultdict(list)
        for row in rows:
            grouped[_instruction_variant_for_row(row)].append(row)

        known_variants = sorted(
            variant for variant in grouped if variant in {"on", "off"}
        )
        if len(known_variants) <= 1 and len(grouped) == 1:
            expanded.append((corpus, rows))
            continue

        for variant in sorted(grouped):
            expanded.append(
                (
                    _join_policy_instruction_variant_corpus(
                        base_corpus=corpus,
                        instruction_variant=variant,
                    ),
                    grouped[variant],
                )
            )
    return expanded


def _corpora_with_persona_variants_for_fan(
    corpora: list[tuple[str, list[RoundTrajectory]]],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """
    Split fan-plot corpora by simulated-target persona when multiple exist.

    Args:
        corpora: Base corpora as (corpus_name, rows) tuples.

    Returns:
        Expanded corpora list with persona-specific corpus names when a corpus
        mixes multiple simulated-target personas.
    """
    expanded: list[tuple[str, list[RoundTrajectory]]] = []
    for corpus, rows in corpora:
        if corpus == "human_reference" or not rows:
            expanded.append((corpus, rows))
            continue

        grouped: dict[str, list[RoundTrajectory]] = {}
        for row in rows:
            persona_raw = row.condition.roles.simulated_target_persona
            if isinstance(persona_raw, str) and persona_raw.strip():
                persona = persona_raw.strip()
            else:
                persona = "none"
            grouped.setdefault(persona, []).append(row)

        non_none_personas = sorted(persona for persona in grouped if persona != "none")
        if len(non_none_personas) <= 1:
            expanded.append((corpus, rows))
            continue

        for persona in sorted(grouped):
            expanded.append(
                (
                    _join_persona_variant_corpus(
                        base_corpus=corpus,
                        persona=persona,
                    ),
                    grouped[persona],
                )
            )
    return expanded


def _corpus_sort_key(
    corpus: str,
) -> tuple[int, str, int, str, int, str, int, str, int, str]:
    """
    Build a stable sort key for corpus plot ordering.

    Args:
        corpus: Corpus key.

    Returns:
        Sort key tuple preferring canonical corpus ordering, then policy suffix.
    """
    (
        base_with_simulator_model_and_instruction_and_persona,
        policy_model,
    ) = _split_policy_variant_corpus(corpus)
    (
        base_with_instruction_and_persona,
        simulator_model,
    ) = _split_simulator_model_variant_corpus(
        base_with_simulator_model_and_instruction_and_persona
    )
    (
        base_with_persona,
        instruction_variant,
    ) = _split_policy_instruction_variant_corpus(base_with_instruction_and_persona)
    base_corpus, persona = _split_persona_variant_corpus(base_with_persona)
    base_order = [
        "human_reference",
        "vanilla_llm_target",
        "structure_target",
        "full_simulated_target",
        "full_no_rhetoric_target",
    ]
    base_rank_lookup = {name: index for index, name in enumerate(base_order)}
    base_rank = base_rank_lookup.get(base_corpus, len(base_rank_lookup))
    persona_rank = 0 if persona is None else 1
    persona_text = persona or ""
    instruction_rank = 0 if instruction_variant is None else 1
    instruction_text = instruction_variant or ""
    simulator_model_rank = 0 if simulator_model is None else 1
    simulator_model_text = simulator_model or ""
    policy_rank = 0 if policy_model is None else 1
    policy_text = policy_model or ""
    return (
        base_rank,
        base_corpus,
        persona_rank,
        persona_text,
        instruction_rank,
        instruction_text,
        simulator_model_rank,
        simulator_model_text,
        policy_rank,
        policy_text,
    )


def _filter_corpora_for_fan(
    corpora: list[tuple[str, list[RoundTrajectory]]],
    selected_base_corpora: set[str],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """
    Filter fan corpora to requested base corpus keys.

    Args:
        corpora: Candidate corpora.
        selected_base_corpora: Base corpus keys to keep. Empty means all.

    Returns:
        Filtered corpora in the original order.
    """
    if not selected_base_corpora:
        return corpora
    return [
        (corpus, rows)
        for corpus, rows in corpora
        if _split_policy_variant_corpus(
            _split_simulator_model_variant_corpus(
                _split_policy_instruction_variant_corpus(
                    _split_persona_variant_corpus(corpus)[0]
                )[0]
            )[0]
        )[0]
        in selected_base_corpora
    ]


def _filter_corpora_by_fan_policy_model(
    corpora: list[tuple[str, list[RoundTrajectory]]],
    selected_policy_models: set[str],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """
    Keep fan corpora whose policy-model suffix matches selection.

    Args:
        corpora: Candidate fan corpora.
        selected_policy_models: Policy model IDs to keep. Empty means all.

    Returns:
        Filtered corpora.
    """
    if not selected_policy_models:
        return corpora
    filtered: list[tuple[str, list[RoundTrajectory]]] = []
    for corpus, rows in corpora:
        _, policy_model = _split_policy_variant_corpus(corpus)
        if policy_model is None:
            filtered.append((corpus, rows))
            continue
        if policy_model in selected_policy_models:
            filtered.append((corpus, rows))
    return filtered
