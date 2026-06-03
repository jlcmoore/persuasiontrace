"""
Compute full joint probability distributions with an API LLM.

This script estimates the joint distribution over Belief_i nodes plus Target by
enumerating all boolean states and asking an LLM to return one probability per
state in a strict JSON payload.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import litellm
from tqdm import tqdm

from experiment.condition import PropositionSource
from experiment.llm_utils import disable_litellm_logging
from simulation.io import read_jsonl_graphs
from simulation.scripts.litellm_script_utils import (
    LITELLM_API_ERRORS,
    begin_dry_run_report_with_cost_range,
    clean_json_response,
    extract_content,
    fallback_message_token_estimate,
    print_dry_run_header,
    print_estimated_cost_range,
)
from simulation.scripts.utils import load_existing_ids, resolve_proposition_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
disable_litellm_logging()

PROPOSITION_SOURCE_CHOICES = tuple(source.value for source in PropositionSource)


@dataclass(frozen=True)
class BatchRequestConfig:
    """Configuration for one LiteLLM batch request.

    Args:
        max_workers: Worker count for batch requests.
        timeout: Per-request timeout in seconds.
        max_tokens: Maximum output tokens per request.
        num_retries: LiteLLM internal retry count.
        temperature: Sampling temperature.
    """

    max_workers: int
    timeout: int
    max_tokens: int
    num_retries: int
    temperature: float


@dataclass(frozen=True)
class ApiScoringConfig:
    """Configuration for API-based joint-distribution scoring.

    Args:
        batch_size: Graphs per batch request.
        max_workers: Worker count for batch requests.
        timeout: Per-request timeout in seconds.
        max_tokens: Maximum output tokens per request.
        max_retries: Batch-level retry attempts.
        num_retries: LiteLLM internal retry count.
        temperature: Sampling temperature.
        epsilon: Additive smoothing before normalization.
        limit: Optional max graph count to process, where ``0`` means all.
    """

    batch_size: int
    max_retries: int
    epsilon: float
    request: BatchRequestConfig
    limit: int = 0


def node_keys(num_beliefs: int) -> list[str]:
    """Build ordered node keys for one graph.

    Args:
        num_beliefs: Number of belief nodes.

    Returns:
        Ordered keys ``Belief_1..Belief_N, Target``.
    """
    keys = [f"Belief_{index}" for index in range(1, num_beliefs + 1)]
    keys.append("Target")
    return keys


def enumerate_state_payload(keys: list[str]) -> list[dict[str, Any]]:
    """Enumerate all boolean states for the provided keys.

    Args:
        keys: Ordered node keys.

    Returns:
        List of dictionaries containing ``id`` and ``state``.
    """
    states: list[dict[str, Any]] = []
    total = 2 ** len(keys)
    for state_index in range(total):
        bits = format(state_index, f"0{len(keys)}b")
        assignment = {key: (bit == "1") for key, bit in zip(keys, bits)}
        states.append({"id": f"S{state_index + 1}", "state": assignment})
    return states


def build_messages(
    *,
    target: str,
    belief_nodes: list[str],
    states: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build prompt messages for one graph scoring request.

    Args:
        target: Target proposition text.
        belief_nodes: Belief-node proposition texts.
        states: Enumerated states with ``id`` and ``state``.

    Returns:
        Chat messages to send to LiteLLM.
    """
    statements: list[dict[str, str]] = []
    for index, text in enumerate(belief_nodes, start=1):
        statements.append({"key": f"Belief_{index}", "text": str(text)})
    statements.append({"key": "Target", "text": str(target)})

    state_lines = "\n".join(
        json.dumps(entry, separators=(",", ":")) for entry in states
    )

    system_prompt = (
        "You are estimating a joint probability distribution over boolean survey "
        "responses for one participant. Return strict JSON only."
    )
    user_prompt = (
        "Statements (keys and natural-language meaning):\n"
        f"{json.dumps(statements, ensure_ascii=True)}\n\n"
        "State space to score:\n"
        f"{state_lines}\n\n"
        "Return JSON in exactly this shape:\n"
        '{"probabilities":[{"id":"S1","probability":0.0}]}\n\n'
        "Rules:\n"
        "1) Include every provided id exactly once.\n"
        "2) probability must be a number in [0,1].\n"
        "3) Probabilities should sum to 1.\n"
        "4) Do not include explanations or markdown."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_probabilities_payload(
    *,
    content: str,
    state_ids: list[str],
    epsilon: float,
) -> dict[str, float]:
    """Parse and normalize a returned probability payload.

    Args:
        content: Raw response content.
        state_ids: Expected state ids for this graph.
        epsilon: Additive smoothing constant before normalization.

    Returns:
        Mapping ``state_id -> normalized probability``.
    """
    try:
        payload = json.loads(clean_json_response(content))
    except json.JSONDecodeError as error:
        raise ValueError(f"Response is not valid JSON: {error}") from error

    entries = payload.get("probabilities")
    if not isinstance(entries, list):
        raise ValueError("Expected top-level key 'probabilities' as a list.")

    parsed: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        state_id = entry.get("id")
        probability = entry.get("probability")
        if not isinstance(state_id, str):
            continue
        if not isinstance(probability, (int, float)):
            continue
        parsed[state_id] = float(probability)

    scores: dict[str, float] = {}
    for state_id in state_ids:
        raw_value = parsed.get(state_id, 0.0)
        clipped = max(0.0, raw_value)
        scores[state_id] = clipped + epsilon

    total = sum(scores.values())
    if total <= 0.0:
        raise ValueError("All returned probabilities were zero after clipping.")

    return {state_id: value / total for state_id, value in scores.items()}


def score_batch(
    *,
    model: str,
    messages_batch: list[list[dict[str, str]]],
    config: ApiScoringConfig,
) -> list[Any]:
    """Submit one batch of scoring requests.

    Args:
        model: LiteLLM model id.
        messages_batch: Batch of chat messages.
        config: API scoring configuration.

    Returns:
        List of LiteLLM responses in input order.
    """
    responses = litellm.batch_completion(
        model=model,
        messages=messages_batch,
        max_workers=config.request.max_workers,
        timeout=config.request.timeout,
        max_tokens=config.request.max_tokens,
        num_retries=config.request.num_retries,
        temperature=config.request.temperature,
    )
    return list(responses)


def build_messages_for_graph(
    graph: dict[str, Any], proposition_source: str
) -> list[dict[str, str]]:
    """Build scoring messages for one graph after validation.

    Args:
        graph: Graph record containing ``bayesian_network``.
        proposition_source: Expected proposition source.

    Returns:
        Prompt messages for the graph.
    """
    if "bayesian_network" not in graph:
        raise ValueError("Input graphs must include a 'bayesian_network' field.")
    graph_data = graph.get("bayesian_network") or {}
    if "belief_nodes" not in graph_data or "target" not in graph_data:
        raise ValueError(
            "Input graphs must include belief_nodes and target in bayesian_network."
        )

    existing_source = graph.get("proposition_source")
    if existing_source and existing_source != proposition_source:
        raise ValueError(
            f"Input graph has proposition_source={existing_source!r}, "
            f"expected {proposition_source!r}."
        )
    graph["proposition_source"] = proposition_source

    belief_nodes = list(graph_data.get("belief_nodes", []))
    return build_messages(
        target=str(graph_data.get("target", "")),
        belief_nodes=belief_nodes,
        states=enumerate_state_payload(node_keys(len(belief_nodes))),
    )


def score_pending_indices(
    *,
    model: str,
    batch_graphs: list[dict[str, Any]],
    pending_indices: list[int],
    proposition_source: str,
    config: ApiScoringConfig,
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Score one retry pass over pending batch indices.

    Args:
        model: LiteLLM model id.
        batch_graphs: Graphs in the current output batch.
        pending_indices: Indices still needing successful parsing.
        proposition_source: Expected proposition source.
        config: API scoring configuration.

    Returns:
        Tuple of successful updates by batch index and next pending list.
    """
    messages_batch = [
        build_messages_for_graph(batch_graphs[pending_index], proposition_source)
        for pending_index in pending_indices
    ]
    responses = score_batch(model=model, messages_batch=messages_batch, config=config)

    successes: dict[int, dict[str, Any]] = {}
    next_pending: list[int] = []
    for response_index, pending_index in enumerate(pending_indices):
        graph = batch_graphs[pending_index]
        try:
            updated_graph = graph_from_response(
                graph=graph,
                response=responses[response_index],
                epsilon=config.epsilon,
            )
            successes[pending_index] = updated_graph
        except (ValueError, TypeError) as error:
            graph_id = graph.get("id") or "unknown-id"
            LOGGER.warning("Parse failure for %s: %s", graph_id, error)
            next_pending.append(pending_index)
    return successes, next_pending


def graph_from_response(
    *,
    graph: dict[str, Any],
    response: Any,
    epsilon: float,
) -> dict[str, Any]:
    """Attach an estimated joint distribution to one input graph.

    Args:
        graph: Input graph record.
        response: LiteLLM response for this graph.
        epsilon: Additive smoothing used in normalization.

    Returns:
        Updated graph record with ``joint_distribution`` and ``coverage``.
    """
    graph_data = graph.get("bayesian_network") or {}
    belief_nodes = graph_data.get("belief_nodes", [])
    keys = node_keys(len(belief_nodes))
    states = enumerate_state_payload(keys)
    state_ids = [entry["id"] for entry in states]

    content = extract_content(response)
    if not content:
        raise ValueError("Empty response content.")
    probs = parse_probabilities_payload(
        content=content,
        state_ids=state_ids,
        epsilon=epsilon,
    )

    joint_distribution: list[dict[str, Any]] = []
    for entry in states:
        state_id = entry["id"]
        probability = probs[state_id]
        joint_distribution.append(
            {
                "state": entry["state"],
                "probability": probability,
                "logprob": math.log(probability),
                "raw_probability": probability,
            }
        )
    joint_distribution.sort(key=lambda row: row["probability"], reverse=True)

    graph_data["coverage"] = 1.0
    graph_data["joint_distribution"] = joint_distribution
    graph["bayesian_network"] = graph_data
    return graph


def run_compute(
    *,
    input_file: str,
    output_file: str,
    model: str,
    proposition_source: str | None,
    config: ApiScoringConfig,
) -> None:
    """Compute API-LLM joint distributions for all input graphs.

    Args:
        input_file: Input JSONL with belief structures.
        output_file: Destination JSONL path.
        model: LiteLLM model identifier.
        proposition_source: Source label when missing in input.
        config: Scoring configuration.
    """
    graphs, proposition_source = load_pending_graphs(
        input_file=input_file,
        output_file=output_file,
        proposition_source=proposition_source,
        limit=config.limit,
    )
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not graphs:
        LOGGER.info("No new graphs to score.")
        return

    failed_total = score_graphs_and_write_output(
        model=model,
        graphs=graphs,
        proposition_source=proposition_source,
        output_file=output_file,
        config=config,
    )

    LOGGER.info("Finished scoring. Output written to %s", output_file)
    if failed_total > 0:
        LOGGER.warning("Skipped %d graphs after retry exhaustion.", failed_total)


def load_pending_graphs(
    *,
    input_file: str,
    output_file: str,
    proposition_source: str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    """Load input graphs and apply limit + resume filtering.

    Args:
        input_file: Input JSONL path.
        output_file: Output JSONL path for resume filtering.
        proposition_source: Optional source label.
        limit: Optional graph limit (0 means all).

    Returns:
        Tuple of pending graphs and resolved proposition source.
    """
    graphs = read_jsonl_graphs(input_file)
    if limit > 0:
        graphs = graphs[:limit]
    if not graphs:
        return [], proposition_source or ""

    resolved_source = resolve_proposition_source(graphs, proposition_source)
    existing_targets = load_existing_ids(output_file)
    if existing_targets:
        graphs = [graph for graph in graphs if graph.get("id") not in existing_targets]
    return graphs, resolved_source


def token_count_for_messages(model: str, messages: list[dict[str, str]]) -> int:
    """Estimate token count for one message list.

    Args:
        model: LiteLLM model id.
        messages: Prompt messages.

    Returns:
        Estimated token count.
    """
    try:
        return int(litellm.token_counter(model=model, messages=messages))
    except (TypeError, ValueError, KeyError):
        return fallback_message_token_estimate(messages)


def estimated_completion_tokens(num_nodes: int) -> tuple[int, int]:
    """Estimate completion token bounds for probability payload output.

    Args:
        num_nodes: Count of boolean variables in one state assignment.

    Returns:
        Tuple of ``(min_tokens, max_tokens)`` per call.
    """
    states_count = 2**num_nodes
    min_tokens = 30 + 10 * states_count
    max_tokens = 60 + 16 * states_count
    return min_tokens, max_tokens


def run_dry_run(
    *,
    graphs: list[dict[str, Any]],
    model: str,
    proposition_source: str,
    config: ApiScoringConfig,
) -> None:
    """Print planned calls and token/cost estimates without API calls.

    Args:
        graphs: Pending graphs to score.
        model: LiteLLM model id.
        proposition_source: Resolved source label.
        config: Scoring configuration.
    """
    calls = len(graphs)
    if calls == 0:
        print_dry_run_header(model)
        print("graphs_pending=0")
        print("llm_calls_estimate=0")
        print_estimated_cost_range(0.0, 0.0)
        return

    prompt_tokens_total = 0
    comp_min_total = 0
    comp_max_total = 0

    for graph in graphs:
        graph_data = graph.get("bayesian_network") or {}
        belief_nodes = list(graph_data.get("belief_nodes", []))
        messages = build_messages(
            target=str(graph_data.get("target", "")),
            belief_nodes=belief_nodes,
            states=enumerate_state_payload(node_keys(len(belief_nodes))),
        )
        prompt_tokens_total += token_count_for_messages(model=model, messages=messages)
        per_call_min, per_call_max = estimated_completion_tokens(
            num_nodes=len(belief_nodes) + 1
        )
        comp_min_total += per_call_min
        comp_max_total += per_call_max

    cost_min, cost_max = begin_dry_run_report_with_cost_range(
        model=model,
        prompt_tokens_total=prompt_tokens_total,
        completion_tokens_min_total=comp_min_total,
        completion_tokens_max_total=comp_max_total,
    )
    print(f"proposition_source={proposition_source}")
    print(f"graphs_pending={calls}")
    print(f"batch_size={config.batch_size}")
    print(f"max_workers={config.request.max_workers}")
    print(f"max_retries={config.max_retries}")
    print(f"llm_calls_estimate={calls}")
    print(f"prompt_tokens_estimate_total={prompt_tokens_total}")
    print("completion_tokens_estimate_total=" f"[{comp_min_total}, {comp_max_total}]")
    print_estimated_cost_range(cost_min, cost_max)
    print("-" * 50)


def score_one_batch(
    *,
    model: str,
    batch_graphs: list[dict[str, Any]],
    proposition_source: str,
    config: ApiScoringConfig,
) -> dict[int, dict[str, Any]]:
    """Score one batch with retries and return successful updates.

    Args:
        model: LiteLLM model id.
        batch_graphs: Graph records in the current batch.
        proposition_source: Expected proposition source.
        config: Scoring configuration.

    Returns:
        Mapping from batch index to updated graph.
    """
    pending_indices = list(range(len(batch_graphs)))
    successes: dict[int, dict[str, Any]] = {}

    for attempt in range(config.max_retries):
        if not pending_indices:
            break
        try:
            round_successes, pending_indices = score_pending_indices(
                model=model,
                batch_graphs=batch_graphs,
                pending_indices=pending_indices,
                proposition_source=proposition_source,
                config=config,
            )
            successes.update(round_successes)
        except LITELLM_API_ERRORS as error:
            LOGGER.warning(
                "Batch API failure on attempt %d/%d: %s",
                attempt + 1,
                config.max_retries,
                error,
            )
    return successes


def score_graphs_and_write_output(
    *,
    model: str,
    graphs: list[dict[str, Any]],
    proposition_source: str,
    output_file: str,
    config: ApiScoringConfig,
) -> int:
    """Score all graphs and write successful rows to output JSONL.

    Args:
        model: LiteLLM model id.
        graphs: Input graph rows.
        proposition_source: Expected proposition source.
        output_file: Destination JSONL path.
        config: Scoring configuration.

    Returns:
        Number of graphs skipped after retry exhaustion.
    """
    file_mode = "a" if os.path.exists(output_file) else "w"
    failed_total = 0
    with open(output_file, file_mode, encoding="utf-8") as output_handle:
        for start in tqdm(
            range(0, len(graphs), config.batch_size), desc="Scoring Graphs"
        ):
            batch_graphs = graphs[start : start + config.batch_size]
            successes = score_one_batch(
                model=model,
                batch_graphs=batch_graphs,
                proposition_source=proposition_source,
                config=config,
            )
            for batch_index, _ in enumerate(batch_graphs):
                updated = successes.get(batch_index)
                if updated is None:
                    failed_total += 1
                    continue
                output_handle.write(json.dumps(updated, ensure_ascii=True) + "\n")
                output_handle.flush()
    return failed_total


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute full joint distributions for belief networks using an API LLM."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/simulation/data/belief_structures.jsonl",
        help="Input JSONL file containing belief structures.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/simulation/data/belief_distributions.jsonl",
        help="Output JSONL file for estimated joint distributions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5.4-mini",
        help="LiteLLM model id to use for scoring.",
    )
    parser.add_argument(
        "--proposition-source",
        type=str,
        choices=PROPOSITION_SOURCE_CHOICES,
        default=None,
        help="Source label to attach when input rows are missing it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of graphs to process (0 means all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Graphs per batch request.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Max workers for LiteLLM batch requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum output tokens per request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Batch-level retries for parse or transient API failures.",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=2,
        help="LiteLLM internal retry count per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for scoring calls.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9,
        help="Additive smoothing before probability normalization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and token/cost estimates without API calls.",
    )
    args = parser.parse_args()

    request_config = BatchRequestConfig(
        max_workers=int(args.max_workers),
        timeout=int(args.timeout),
        max_tokens=int(args.max_tokens),
        num_retries=int(args.num_retries),
        temperature=float(args.temperature),
    )
    config = ApiScoringConfig(
        batch_size=int(args.batch_size),
        max_retries=int(args.max_retries),
        epsilon=float(args.epsilon),
        request=request_config,
        limit=int(args.limit),
    )

    if bool(args.dry_run):
        graphs, resolved_source = load_pending_graphs(
            input_file=args.input,
            output_file=args.output,
            proposition_source=args.proposition_source,
            limit=config.limit,
        )
        run_dry_run(
            graphs=graphs,
            model=args.model,
            proposition_source=resolved_source,
            config=config,
        )
        return

    run_compute(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        proposition_source=args.proposition_source,
        config=config,
    )


if __name__ == "__main__":
    main()
