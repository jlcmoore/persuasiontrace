"""
Compute the full joint probability distribution over belief networks.
Uses tsor13/spectrum models for forced-completion scoring over all 2^N states.
"""

import argparse
import itertools
import json
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment.condition import PropositionSource
from simulation.io import read_jsonl_graphs
from simulation.scripts.utils import load_existing_ids, resolve_proposition_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROPOSITION_SOURCE_CHOICES = tuple(source.value for source in PropositionSource)


def get_device() -> str:
    """Determine the best available PyTorch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_messages(
    target: str, belief_nodes: List[str]
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Construct the messages payload formatted for Spectrum models.
    Also returns the ordered list of keys used in the JSON.
    """
    input_content = "Consider the following statements:\n"

    keys = []
    # Beliefs first
    for i, node in enumerate(belief_nodes):
        key = f"Belief_{i+1}"
        keys.append(key)
        input_content += f'"{key}": "{node}"\n'

    # Target last
    keys.append("Target")
    input_content += f'"Target": "{target}"\n'

    input_content += (
        "\nOutput exactly one of the possible JSON assignments indicating true/false "
        "for each statement. Do not explain. Do not add extra keys."
    )

    # Spectrum models use description, input, output roles
    messages = [
        {
            "role": "description",
            "content": (
                "The following are survey responses from one randomly selected adult American. "
                "Output exactly one JSON object giving that person's true/false responses."
            ),
        },
        {"role": "input", "content": input_content},
    ]
    return messages, keys


def _chat_template_token_ids(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> List[int]:
    """
    Normalize tokenizer.apply_chat_template output into a single token-id list.

    Different transformers/tokenizer versions may return a plain list[int],
    list[list[int]], torch.Tensor, or a BatchEncoding-like object.
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )

    if isinstance(encoded, list):
        if not encoded:
            return []
        if isinstance(encoded[0], list):
            return [int(token_id) for token_id in encoded[0]]
        return [int(token_id) for token_id in encoded]

    if torch.is_tensor(encoded):
        if encoded.ndim == 2:
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded.tolist()]

    if hasattr(encoded, "get"):
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise TypeError("Chat template output missing input_ids.")
        if torch.is_tensor(input_ids):
            if input_ids.ndim == 2:
                input_ids = input_ids[0]
            return [int(token_id) for token_id in input_ids.tolist()]
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [int(token_id) for token_id in input_ids[0]]
            return [int(token_id) for token_id in input_ids]

    raise TypeError(
        "Unsupported chat-template return type for token ids: "
        f"{type(encoded).__name__}"
    )


def get_probabilities_batched(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    completions: List[str],
    device: str,
    batch_size: int = 32,
) -> List[float]:
    """
    Perform forced completion scoring over a set of possible continuations using batched inference.
    """
    # Ensure pad token exists for batching
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # The length of the prompt determines where the completion begins
    prompt_ids = _chat_template_token_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )
    prompt_len = len(prompt_ids)

    all_logprobs = []

    for i in range(0, len(completions), batch_size):
        batch_comps = completions[i : i + batch_size]

        batch_input_ids = []
        for comp in batch_comps:
            full_messages = messages + [{"role": "output", "content": comp}]
            full_ids = _chat_template_token_ids(tokenizer, full_messages)
            batch_input_ids.append(full_ids)

        # Right-pad sequences in the batch
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        attention_mask = []

        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_tensor = torch.tensor(padded_input_ids).to(device)
        mask_tensor = torch.tensor(attention_mask).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)

        # Shift logits and labels for next-token prediction
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_tensor[..., 1:].contiguous()
        shift_mask = mask_tensor[..., 1:].contiguous()

        # Calculate log probabilities of the actual target tokens
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # We only want to sum over the completion tokens, ignoring prompt and padding.
        comp_mask = shift_mask.clone()
        comp_mask[:, : prompt_len - 1] = 0

        # Sum logprobs over the completion sequence
        seq_log_probs = (token_log_probs * comp_mask).sum(dim=1).cpu().tolist()
        all_logprobs.extend(seq_log_probs)

    return all_logprobs


def run_compute(
    input_file: str,
    output_file: str,
    model_name: str,
    limit: int,
    proposition_source: str | None,
    local_files_only: bool = False,
) -> None:
    """Load graphs, enumerate joint states, score them, and save."""
    graphs = read_jsonl_graphs(input_file)

    if limit > 0:
        graphs = graphs[:limit]

    if not graphs:
        logging.warning("No valid graphs found to process.")
        return

    device = get_device()
    logging.info("Loading model %s on %s...", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        local_files_only=local_files_only,
    )
    if device == "mps":
        model.to(device)
    model.eval()

    logging.info("Starting enumeration and scoring...")

    proposition_source = resolve_proposition_source(graphs, proposition_source)

    existing_targets = load_existing_ids(output_file)

    if existing_targets:
        graphs = [graph for graph in graphs if graph.get("id") not in existing_targets]

    file_mode = "a" if os.path.exists(output_file) else "w"

    with open(output_file, file_mode, encoding="utf-8") as f_out:
        for graph in tqdm(graphs, desc="Scoring Graphs"):
            if "bayesian_network" not in graph:
                raise ValueError(
                    "Input graphs must include a 'bayesian_network' field."
                )
            graph_data = graph.get("bayesian_network") or {}
            if "belief_nodes" not in graph_data or "target" not in graph_data:
                raise ValueError(
                    "Input graphs must include belief_nodes and target in bayesian_network."
                )

            existing_source = graph.get("proposition_source")
            if existing_source and existing_source != proposition_source:
                logging.error(
                    "Skipping graph with mismatched proposition_source %r (expected %r).",
                    existing_source,
                    proposition_source,
                )
                continue
            graph["proposition_source"] = proposition_source
            target = graph_data.get("target", "")
            belief_nodes = graph_data.get("belief_nodes", [])

            messages, keys = build_messages(target, belief_nodes)

            # Enumerate all 2^N states
            completions = []
            state_dicts = []
            for combo in itertools.product([True, False], repeat=len(keys)):
                state_dict = dict(zip(keys, combo))
                state_dicts.append(state_dict)
                completions.append(json.dumps(state_dict, separators=(",", ":")))

            try:
                logprobs = get_probabilities_batched(
                    model, tokenizer, messages, completions, device
                )
            except (ValueError, RuntimeError, TypeError) as e:
                logging.error("Failed to score graph: %s", e)
                continue

            # Calculate raw probability mass (Coverage)
            raw_probs = [math.exp(lp) for lp in logprobs]
            coverage = sum(raw_probs)

            # Calculate normalized probabilities
            if coverage > 0.0:
                normalized_probs = [rp / coverage for rp in raw_probs]
            else:
                max_lp = max(logprobs)
                exp_probs = [math.exp(lp - max_lp) for lp in logprobs]
                fallback_cov = sum(exp_probs)
                normalized_probs = [ep / fallback_cov for ep in exp_probs]

            # Combine results
            joint_distribution = []
            for state, prob, lp, rp in zip(
                state_dicts, normalized_probs, logprobs, raw_probs
            ):
                joint_distribution.append(
                    {
                        "state": state,
                        "probability": prob,
                        "logprob": lp,
                        "raw_probability": rp,
                    }
                )

            joint_distribution.sort(key=lambda x: x["probability"], reverse=True)

            graph_data["coverage"] = coverage
            graph_data["joint_distribution"] = joint_distribution
            graph["bayesian_network"] = graph_data

            f_out.write(json.dumps(graph) + "\n")
            f_out.flush()

    logging.info("Finished scoring. Results saved to %s", output_file)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute joint probabilities over belief networks using Spectrum models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tsor13/spectrum-Llama-3.1-8B-v1",
        help="HuggingFace model ID for the Spectrum model",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/simulation/data/belief_structures.jsonl",
        help="Input JSONL file containing graph structures",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/simulation/data/belief_distributions.jsonl",
        help="Output JSONL file for the distributions",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of graphs to process (0=all)"
    )
    parser.add_argument(
        "--proposition-source",
        type=str,
        choices=PROPOSITION_SOURCE_CHOICES,
        default=None,
        help="Source label to attach to scored propositions when missing in input",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Force use of local cached files only",
    )
    args = parser.parse_args()

    run_compute(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        limit=args.limit,
        proposition_source=args.proposition_source,
        local_files_only=args.local_only,
    )


if __name__ == "__main__":
    main()
