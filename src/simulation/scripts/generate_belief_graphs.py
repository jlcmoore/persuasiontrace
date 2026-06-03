"""
Generate belief dependency graphs for propositions using LiteLLM.
"""

import argparse
import json
import logging
import os
from typing import Any, Iterable, Mapping, Optional, Sequence

import litellm

from experiment.condition import PropositionSource
from experiment.llm_utils import disable_litellm_logging
from simulation.scripts.litellm_script_utils import (
    LITELLM_API_ERRORS,
    begin_dry_run_report_with_cost_range,
    clean_json_response,
    extract_content,
    fallback_message_token_estimate,
    print_estimated_cost_range,
)
from simulation.scripts.utils import load_existing_ids

# Shared LiteLLM configuration mirroring llm-delusions
litellm.drop_params = True
litellm.vertex_ai_safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
disable_litellm_logging()

PROPOSITION_SOURCE_CHOICES = tuple(source.value for source in PropositionSource)


def get_schema_instruction(min_beliefs: int, max_beliefs: int) -> str:
    """Generate the JSON schema instruction for a belief-count range."""
    belief_placeholders = ",\n    ".join(
        [f'"string (Belief {i+1})"' for i in range(max_beliefs)]
    )
    return f"""
You must output valid JSON matching this exact schema:
{{
  "belief_nodes": [
    {belief_placeholders}
  ],
  "edges": [
    {{"from": 1, "to": 0, "positive_influence": true}},
    {{"from": 2, "to": 0, "positive_influence": false}}
  ]
}}

- Node 0 is implicitly the target proposition.
- "belief_nodes" contains ONLY the newly generated supporting/opposing beliefs.
- The 1-based index in "from" refers to the position in the "belief_nodes" array.
- "positive_influence" is true if believing the source makes the target MORE likely.
- "positive_influence" is false if believing the source makes the target LESS likely.
- Every node must eventually connect to Node 0, but indirect paths (e.g., A -> B -> Node 0) are highly encouraged to show deep reasoning.
- Prefer direct Belief_i -> Target edges unless an intermediate node is truly
  necessary as a mediator.
- Do not add a hierarchy layer only for rhetorical detail or narrative flow.
- Every belief node must add distinct causal value for predicting the target;
  remove nodes that are merely consequences, restatements, or weak elaborations.
- If a chain A -> B -> Target can be represented as A -> Target without losing
  clear causal meaning, prefer the flattened edge.
- There must be BETWEEN {min_beliefs} and {max_beliefs} nodes in "belief_nodes".
Respond strictly with the JSON object and no markdown blocks.
"""


class LLMClientError(RuntimeError):
    """Raised when a LiteLLM request fails."""

    def __init__(self, message: str, *, inner: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.inner = inner


def batch_completion(
    messages: Iterable[Sequence[Mapping[str, object]]],
    *,
    model: str,
    timeout: int = 60,
    max_workers: int = 8,
) -> list[object]:
    """Call litellm.batch_completion with shared error handling."""
    try:
        responses = litellm.batch_completion(
            model=model,
            messages=list(messages),
            timeout=timeout,
            max_workers=max_workers,
            temperature=0.2,
        )
        return list(responses)
    except LITELLM_API_ERRORS as err:
        raise LLMClientError(f"LiteLLM batch request failed: {err}", inner=err) from err


def prepare_messages(
    prop_text: str, min_beliefs: int, max_beliefs: int
) -> list[dict[str, Any]]:
    """Prepare the message payload for a single proposition."""
    prompt = (
        f'Given the target proposition: "{prop_text}"\n\n'
        f"Produce BETWEEN {min_beliefs} and {max_beliefs} natural-language "
        "belief statements "
        "such that differences in these statements would explain why different "
        "people endorse or reject the target.\n\n"
        "Requirements for each belief:\n"
        "1. A standalone natural-language statement.\n"
        "2. Truth-apt: Something that can reasonably be assigned a probability.\n"
        "3. Distinct: No near-duplicates.\n"
        "4. Causally useful: Beliefs form a causal web reaching the target.\n\n"
        "Hierarchy quality constraints:\n"
        "- Use mediation edges only when the mediator is indispensable.\n"
        "- Avoid unnecessary depth; flatten weak chains into direct target causes.\n"
        "- Do not include nodes that are causally downstream consequences of the target.\n"
        "- Do not include near-synonyms or rhetorical variants of another node.\n\n"
        "Return the beliefs in 'belief_nodes' (do not include the target) and define "
        "the 'edges' where 'positive_influence' is a boolean."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert in cognitive science and causal reasoning. "
                f"{get_schema_instruction(min_beliefs, max_beliefs)}"
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_and_validate_graph(
    content: str, min_beliefs: int, max_beliefs: int
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Parse content into a graph and validate it structurally.
    Returns (graph_data, error_message).
    """
    try:
        clean_content = clean_json_response(content)
        graph_data = json.loads(clean_content)

        belief_nodes = graph_data.get("belief_nodes", [])
        edges = graph_data.get("edges", [])

        if len(belief_nodes) < min_beliefs or len(belief_nodes) > max_beliefs:
            return (
                None,
                "Expected belief_nodes count in "
                f"[{min_beliefs}, {max_beliefs}], got {len(belief_nodes)}",
            )

        for edge in edges:
            if not all(k in edge for k in ("from", "to", "positive_influence")):
                return (
                    None,
                    "Missing required edge keys (from, to, positive_influence).",
                )

        return {"belief_nodes": belief_nodes, "edges": edges}, None
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    except ValueError as exc:
        return None, str(exc)


def process_single_batch(
    batch_props: list[dict[str, Any]],
    batch_messages: list[list[dict[str, Any]]],
    model: str,
    max_workers: int,
    min_beliefs: int,
    max_beliefs: int,
    f_out: Any,
    proposition_source: str,
) -> list[dict[str, Any]]:
    """
    Process one batch, write successes, and return a list of (prop_text, error)
    for any failed items.
    """
    failed_items: list[dict[str, Any]] = []
    try:
        responses = batch_completion(
            batch_messages, model=model, max_workers=max_workers
        )

        for prop_data, resp in zip(batch_props, responses):
            prop_text = prop_data["id"]
            content = extract_content(resp)
            if not content:
                failed_items.append({**prop_data, "error": "Empty response"})
                continue

            graph_data, err_msg = parse_and_validate_graph(
                content, min_beliefs, max_beliefs
            )

            if graph_data:
                graph_data["target"] = prop_text
                result = {
                    "id": prop_text,
                    "factual_domain": prop_data.get("factual_domain", False),
                    "proposition_is_correct": prop_data.get("proposition_is_correct"),
                    "control_dialogue": prop_data.get("control_dialogue", False),
                    "original_text": prop_data.get("original_text"),
                    "proposition_source": proposition_source,
                    "bayesian_network": graph_data,
                }
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                print(f"[Success] {prop_text[:60]}...")
            else:
                failed_items.append(
                    {**prop_data, "error": err_msg or "Unknown validation error"}
                )

    except LLMClientError as exc:
        logging.error("Batch failure: %s", exc)
        for prop_data in batch_props:
            failed_items.append({**prop_data, "error": str(exc)})

    return failed_items


def process_with_retries(
    propositions: list[dict[str, Any]],
    model: str,
    batch_size: int,
    min_beliefs: int,
    max_beliefs: int,
    max_retries: int,
    f_out: Any,
    proposition_source: str,
) -> None:
    """Process propositions in batches with retries for structural failures."""
    remaining_props = propositions
    next_remaining_props: list[dict[str, Any]] = []

    for attempt in range(max_retries):
        if not remaining_props:
            break

        if attempt > 0:
            print(f"--- Retry attempt {attempt + 1}/{max_retries} ---")

        all_messages = [
            prepare_messages(p["id"], min_beliefs, max_beliefs) for p in remaining_props
        ]
        next_remaining_props = []

        for i in range(0, len(all_messages), batch_size):
            batch_msgs = all_messages[i : i + batch_size]
            batch_props = remaining_props[i : i + batch_size]
            failed = process_single_batch(
                batch_props,
                batch_msgs,
                model,
                batch_size,
                min_beliefs,
                max_beliefs,
                f_out,
                proposition_source,
            )
            next_remaining_props.extend(failed)

        remaining_props = next_remaining_props

    # After all retries, log the persistent failures and write them to output
    for prop_data in next_remaining_props:
        prop_text = prop_data["id"]
        err_msg = prop_data.get("error", "Unknown validation error")
        print(f"[Failed permanently] {prop_text[:60]}... ({err_msg})")
        f_out.write(
            json.dumps(
                {
                    "id": prop_text,
                    "error": err_msg,
                    "proposition_source": proposition_source,
                }
            )
            + "\n"
        )
        f_out.flush()


def load_propositions(input_file: str) -> list[dict[str, Any]]:
    """Load propositions from a JSONL input file.

    Args:
        input_file: Input JSONL path.

    Returns:
        Parsed proposition rows normalized for graph generation.
    """
    propositions: list[dict[str, Any]] = []
    if not os.path.exists(input_file):
        logging.error("Input file not found: %s", input_file)
        return propositions

    with open(input_file, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line in %s", input_file)
                continue
            if "id" not in data:
                continue
            prop_data = {"id": data["id"]}
            for key in (
                "factual_domain",
                "proposition_is_correct",
                "control_dialogue",
                "original_text",
            ):
                if key in data:
                    prop_data[key] = data[key]
            propositions.append(prop_data)
    return propositions


def select_pending_propositions(
    propositions: list[dict[str, Any]], output_file: str, limit: int
) -> list[dict[str, Any]]:
    """Filter propositions to pending rows and apply optional limit.

    Args:
        propositions: Loaded proposition rows.
        output_file: Existing output file used for resume filtering.
        limit: Optional max rows to keep.

    Returns:
        Filtered proposition rows.
    """
    existing_targets = load_existing_ids(output_file)
    if existing_targets:
        propositions = [
            prop for prop in propositions if prop["id"] not in existing_targets
        ]
    if limit > 0:
        propositions = propositions[:limit]
    return propositions


def token_count_for_messages(model: str, messages: list[dict[str, Any]]) -> int:
    """Estimate prompt token count for one message list.

    Args:
        model: LiteLLM model id.
        messages: Chat messages.

    Returns:
        Estimated token count.
    """
    try:
        return int(litellm.token_counter(model=model, messages=messages))
    except (TypeError, ValueError, KeyError):
        return fallback_message_token_estimate(messages)


def estimated_completion_tokens_range(
    min_beliefs: int, max_beliefs: int
) -> tuple[int, int]:
    """Return heuristic completion-token bounds per generation call.

    Args:
        min_beliefs: Minimum requested beliefs.
        max_beliefs: Maximum requested beliefs.

    Returns:
        Tuple of ``(min_tokens, max_tokens)``.
    """
    min_tokens = 90 + 28 * min_beliefs
    max_tokens = 140 + 42 * max_beliefs
    return int(min_tokens), int(max_tokens)


def run_dry_run(
    propositions: list[dict[str, Any]],
    *,
    model: str,
    min_beliefs: int,
    max_beliefs: int,
    batch_size: int,
    max_retries: int,
) -> None:
    """Print dry-run plan, token estimate, and cost estimate.

    Args:
        propositions: Pending proposition rows.
        model: LiteLLM model id.
        min_beliefs: Minimum requested beliefs.
        max_beliefs: Maximum requested beliefs.
        batch_size: Planned batch size.
        max_retries: Planned max retries.
    """
    calls = len(propositions)
    prompt_token_counts = [
        token_count_for_messages(
            model=model,
            messages=prepare_messages(
                prop_text=str(prop["id"]),
                min_beliefs=min_beliefs,
                max_beliefs=max_beliefs,
            ),
        )
        for prop in propositions
    ]
    prompt_tokens_total = sum(prompt_token_counts)
    prompt_tokens_avg = float(prompt_tokens_total) / float(calls) if calls > 0 else 0.0
    comp_min_per_call, comp_max_per_call = estimated_completion_tokens_range(
        min_beliefs=min_beliefs, max_beliefs=max_beliefs
    )
    comp_min_total = calls * comp_min_per_call
    comp_max_total = calls * comp_max_per_call

    cost_kwargs = {
        "model": model,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_min_total": comp_min_total,
        "completion_tokens_max_total": comp_max_total,
    }
    cost_min, cost_max = begin_dry_run_report_with_cost_range(**cost_kwargs)
    print(f"propositions_pending={calls}")
    print(f"batch_size={batch_size}")
    print(f"max_retries={max_retries}")
    print(f"belief_count_range=[{min_beliefs}, {max_beliefs}] " "(excluding Target)")
    print(f"llm_calls_estimate={calls}")
    print(
        "prompt_tokens_estimate_total="
        f"{prompt_tokens_total} (avg_per_call={prompt_tokens_avg:.1f})"
    )
    print(
        "completion_tokens_estimate_total="
        f"[{comp_min_total}, {comp_max_total}] "
        f"(per_call=[{comp_min_per_call}, {comp_max_per_call}])"
    )
    print_estimated_cost_range(cost_min, cost_max)
    print("-" * 50)


def run_generation(
    input_file: str,
    output_file: str,
    model: str,
    batch_size: int,
    min_beliefs: int,
    max_beliefs: int,
    limit: int,
    max_retries: int,
    proposition_source: str,
    dry_run: bool = False,
) -> None:
    """Generate graphs and write them to output_file."""
    propositions = load_propositions(input_file=input_file)
    if not propositions:
        return

    propositions = select_pending_propositions(
        propositions=propositions,
        output_file=output_file,
        limit=limit,
    )

    print(f"Loaded {len(propositions)} propositions.")
    print(f"Using model: {model}")
    print(f"Generating between {min_beliefs} and {max_beliefs} beliefs per graph.")
    print(f"Batches of {batch_size} (Max retries: {max_retries})")
    print("-" * 50)

    if dry_run:
        run_dry_run(
            propositions=propositions,
            model=model,
            min_beliefs=min_beliefs,
            max_beliefs=max_beliefs,
            batch_size=batch_size,
            max_retries=max_retries,
        )
        return

    if os.path.exists(output_file):
        file_mode = "a"
    else:
        file_mode = "w"

    with open(output_file, file_mode, encoding="utf-8") as f_out:
        process_with_retries(
            propositions=propositions,
            model=model,
            batch_size=batch_size,
            min_beliefs=min_beliefs,
            max_beliefs=max_beliefs,
            max_retries=max_retries,
            f_out=f_out,
            proposition_source=proposition_source,
        )

    print("-" * 50)
    print(f"Finished generating graph structures. Results saved to {output_file}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate belief dependency graphs using LiteLLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vertex_ai/gemini-3-flash-preview",
        help="Model to use",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/data/participant_propositions.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/simulation/data/belief_structures.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Concurrent API calls"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit propositions (0=all)"
    )
    parser.add_argument(
        "--num-beliefs",
        type=int,
        default=None,
        help=(
            "Deprecated exact count alias. When provided, it overrides "
            "--min-beliefs/--max-beliefs with an exact value."
        ),
    )
    parser.add_argument(
        "--min-beliefs",
        type=int,
        default=4,
        help="Minimum belief nodes to generate (default: 4).",
    )
    parser.add_argument(
        "--max-beliefs",
        type=int,
        default=4,
        help="Maximum belief nodes to generate (default: 4).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for structurally invalid LLM responses (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Print planned calls and token/cost estimates without making API calls."),
    )
    parser.add_argument(
        "--proposition-source",
        type=str,
        choices=PROPOSITION_SOURCE_CHOICES,
        required=True,
        help="Source label to attach to generated propositions",
    )
    args = parser.parse_args()

    min_beliefs = int(args.min_beliefs)
    max_beliefs = int(args.max_beliefs)
    if args.num_beliefs is not None:
        min_beliefs = int(args.num_beliefs)
        max_beliefs = int(args.num_beliefs)
    if min_beliefs < 1:
        raise ValueError("--min-beliefs must be >= 1.")
    if max_beliefs < min_beliefs:
        raise ValueError("--max-beliefs must be >= --min-beliefs.")

    run_generation(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        batch_size=args.batch_size,
        min_beliefs=min_beliefs,
        max_beliefs=max_beliefs,
        limit=args.limit,
        max_retries=args.max_retries,
        proposition_source=args.proposition_source,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
