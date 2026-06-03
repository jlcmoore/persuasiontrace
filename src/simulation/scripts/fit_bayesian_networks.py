"""
Fit Conditional Probability Tables (CPTs) to empirical joint distributions
using the causal graph structures.
"""

import argparse
import itertools
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

from experiment.condition import PropositionSource
from simulation.io import read_jsonl_graphs
from simulation.scripts.utils import resolve_proposition_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROPOSITION_SOURCE_CHOICES = tuple(source.value for source in PropositionSource)


def marginalize_and_condition(
    joint_distribution: List[Dict[str, Any]],
    target_node: str,
    parent_nodes: List[str],
) -> Dict[str, Any]:
    """
    Compute P(target_node=True | parent_nodes) from the full joint distribution.
    Returns a CPT dictionary mapping stringified parent states to probabilities.
    """
    # Maps tuple(parent_values) -> total_prob
    parent_marginals: Dict[tuple, float] = defaultdict(float)

    # Maps tuple(parent_values) -> P(target=True and parents)
    joint_target_true: Dict[tuple, float] = defaultdict(float)

    for entry in joint_distribution:
        state = entry["state"]
        prob = entry["probability"]

        # Extract the boolean values of the parents for this state
        parent_vals = tuple(state[p] for p in parent_nodes)

        parent_marginals[parent_vals] += prob
        if state[target_node] is True:
            joint_target_true[parent_vals] += prob

    cpt = {}

    # Generate all possible boolean combinations for the parents
    for parent_combo in itertools.product([True, False], repeat=len(parent_nodes)):
        marg_prob = parent_marginals.get(parent_combo, 0.0)

        if marg_prob > 0:
            p_true = joint_target_true.get(parent_combo, 0.0) / marg_prob
        else:
            # Fallback to 0.5 (maximum entropy) to avoid NaNs
            p_true = 0.5

        # Create a string key like "Belief_1=True,Belief_2=False"
        condition_key = ",".join(f"{p}={v}" for p, v in zip(parent_nodes, parent_combo))

        # If there are no parents, the key is just "prior"
        if not parent_nodes:
            condition_key = "prior"

        cpt[condition_key] = p_true

    return cpt


def run_fitting(
    input_file: str, output_file: str, proposition_source: str | None
) -> None:
    """Read full joint distributions, fit CPTs based on graph edges, and save."""
    graphs = read_jsonl_graphs(input_file)

    fitted_graphs = []

    proposition_source = resolve_proposition_source(graphs, proposition_source)

    for item in graphs:
        if "bayesian_network" not in item:
            raise ValueError("Input graphs must include a 'bayesian_network' field.")
        graph_data = item.get("bayesian_network") or {}
        if "joint_distribution" not in graph_data:
            raise ValueError(
                "Input graphs must include joint_distribution in bayesian_network."
            )

        existing_source = item.get("proposition_source")
        if existing_source and existing_source != proposition_source:
            logging.error(
                "Skipping graph with mismatched proposition_source %r (expected %r).",
                existing_source,
                proposition_source,
            )
            continue

        joint = graph_data["joint_distribution"]
        edges = graph_data.get("edges", [])
        num_beliefs = len(graph_data.get("belief_nodes", []))

        # Map graph edges to child->parents relationships
        parents_map: Dict[str, List[str]] = defaultdict(list)

        for edge in edges:
            source_idx = edge.get("from")
            target_idx = edge.get("to")

            source_name = "Target" if source_idx == 0 else f"Belief_{source_idx}"
            target_name = "Target" if target_idx == 0 else f"Belief_{target_idx}"

            parents_map[target_name].append(source_name)

        all_nodes = ["Target"] + [f"Belief_{i+1}" for i in range(num_beliefs)]

        # Fit the CPT for every node in the graph based on the parents
        cpts = {}
        for node in all_nodes:
            parents = sorted(parents_map.get(node, []))
            cpt = marginalize_and_condition(joint, node, parents)

            cpts[node] = {"parents": parents, "probabilities": cpt}

        graph_data["bayesian_network"] = cpts

        # Package the entire original graph data inside the 'bayesian_network'
        # field so it matches the SQL Proposition model schema perfectly.
        db_item = {
            "id": item.get("id") or graph_data.get("target"),
            "factual_domain": item.get("factual_domain", False),
            "proposition_is_correct": item.get("proposition_is_correct"),
            "control_dialogue": item.get("control_dialogue", False),
            "original_text": item.get("original_text"),
            "proposition_source": proposition_source,
            "bayesian_network": graph_data,
        }

        fitted_graphs.append(db_item)
    with open(output_file, "w", encoding="utf-8") as f_out:
        for graph in fitted_graphs:
            f_out.write(json.dumps(graph) + "\n")

    logging.info("Fitted Bayesian Networks for %d graphs.", len(fitted_graphs))
    logging.info("Results saved to %s", output_file)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fit Bayesian Network CPTs to the empirical joint distribution."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/simulation/data/belief_distributions.jsonl",
        help="Input JSONL containing joint distributions and graph structures",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/simulation/data/fitted_bayesian_networks.jsonl",
        help="Output JSONL containing the fitted CPTs",
    )
    parser.add_argument(
        "--proposition-source",
        type=str,
        choices=PROPOSITION_SOURCE_CHOICES,
        default=None,
        help="Source label to attach to fitted propositions when missing in input",
    )
    args = parser.parse_args()

    run_fitting(args.input, args.output, args.proposition_source)


if __name__ == "__main__":
    main()
