"""
Auto-clean fitted Bayesian networks with edge cleanup and target-reachability pruning.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation.io import read_jsonl_records
from simulation.scripts.audit_bn_sign_consistency import (
    classify_context_direction,
    cpt_context_deltas,
)

SIGN_TOLERANCE = 1e-9
MIN_TOTAL_NODES_TO_KEEP = 3


@dataclass(frozen=True)
class CleanupSummary:
    """Summary payload for one cleaned proposition."""

    proposition_id: str
    proposition_source: str
    old_belief_count: int
    new_belief_count: int
    dropped_belief_count: int
    old_edge_count: int
    new_edge_count: int
    dropped_edge_count: int
    relabeled_edge_count: int
    dropped_beliefs: str
    dropped_edges: str
    relabeled_edges: str
    drop_reasons: str


def parse_args() -> argparse.Namespace:
    """Parse CLI args for BN cleanup.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Remove inconsistent edges, then drop belief nodes without a path to "
            "Target, then refit CPTs from the projected joint distribution."
        )
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="src/simulation/data/fitted_bayesian_networks_*.jsonl",
        help="Glob pattern or comma-separated file list.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/simulation/data/cleaned",
        help="Output directory for cleaned JSONL files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="analysis/data/bn_target_node_cleanup_summary.csv",
        help="Cleanup summary CSV path.",
    )
    return parser.parse_args()


def resolve_input_paths(inputs_arg: str) -> list[Path]:
    """Resolve input expression to existing files.

    Args:
        inputs_arg: Glob pattern or comma-separated file list.

    Returns:
        Existing file paths.
    """
    raw_value = inputs_arg.strip()
    if "," in raw_value:
        candidates = [
            Path(item.strip()) for item in raw_value.split(",") if item.strip()
        ]
    else:
        candidates = sorted(Path(".").glob(raw_value))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise ValueError(f"No files found for --inputs={inputs_arg!r}.")
    return existing


def node_name_from_index(index: int) -> str:
    """Map edge index to node identifier."""
    return "Target" if index == 0 else f"Belief_{index}"


def sign_of(value: float) -> int:
    """Map numeric value to sign bucket."""
    if value > SIGN_TOLERANCE:
        return 1
    if value < -SIGN_TOLERANCE:
        return -1
    return 0


def cpt_key_from_assignment(parents: list[str], assignment: dict[str, bool]) -> str:
    """Build a CPT key from ordered parent assignment."""
    if not parents:
        return "prior"
    return ",".join(f"{parent}={assignment[parent]}" for parent in parents)


def fit_cpt_for_node(
    *,
    joint_distribution: list[dict[str, Any]],
    target_node: str,
    parent_nodes: list[str],
) -> dict[str, float]:
    """Fit one child-node CPT from a joint distribution.

    Args:
        joint_distribution: Joint rows with state/probability.
        target_node: Child node id.
        parent_nodes: Ordered parent node ids.

    Returns:
        Fitted CPT map.
    """
    parent_mass: dict[tuple[bool, ...], float] = defaultdict(float)
    target_true_mass: dict[tuple[bool, ...], float] = defaultdict(float)

    for entry in joint_distribution:
        state = entry["state"]
        probability = float(entry["probability"])
        parent_values = tuple(bool(state[parent]) for parent in parent_nodes)
        parent_mass[parent_values] += probability
        if bool(state[target_node]):
            target_true_mass[parent_values] += probability

    cpt: dict[str, float] = {}
    for combo in itertools.product([True, False], repeat=len(parent_nodes)):
        total = parent_mass.get(combo, 0.0)
        p_true = target_true_mass.get(combo, 0.0) / total if total > 0 else 0.5
        assignment = dict(zip(parent_nodes, combo))
        cpt[cpt_key_from_assignment(parent_nodes, assignment)] = float(p_true)
    return cpt


def build_parents_map(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build child->parents mapping from edge list."""
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source_name = node_name_from_index(int(edge["from"]))
        target_name = node_name_from_index(int(edge["to"]))
        parents[target_name].append(source_name)
    return parents


def refit_cpts(
    *,
    joint_distribution: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    num_beliefs: int,
) -> dict[str, dict[str, Any]]:
    """Refit CPT map after cleanup.

    Args:
        joint_distribution: Projected joint distribution.
        edges: Cleaned edge list.
        num_beliefs: Number of retained beliefs.

    Returns:
        Refit CPT map.
    """
    parents_map = build_parents_map(edges)
    nodes = ["Target"] + [f"Belief_{index}" for index in range(1, num_beliefs + 1)]
    fitted: dict[str, dict[str, Any]] = {}
    for node in nodes:
        parents = sorted(parents_map.get(node, []))
        fitted[node] = {
            "parents": parents,
            "probabilities": fit_cpt_for_node(
                joint_distribution=joint_distribution,
                target_node=node,
                parent_nodes=parents,
            ),
        }
    return fitted


def edge_cpt_direction(
    *,
    edge: dict[str, Any],
    cpt_map: dict[str, Any],
) -> str:
    """Compute context-wise CPT direction for one edge.

    Args:
        edge: Edge payload with from/to.
        cpt_map: Node->CPT mapping.

    Returns:
        Direction label: positive/negative/mixed/neutral/missing.
    """
    target_idx = int(edge["to"])
    source_idx = int(edge["from"])
    target_node = node_name_from_index(target_idx)
    source_node = node_name_from_index(source_idx)
    target_spec = cpt_map.get(target_node, {})
    if not isinstance(target_spec, dict):
        return "missing"
    deltas = cpt_context_deltas(cpt_node_spec=target_spec, source_node=source_node)
    return classify_context_direction(deltas)


def sanitize_edges_by_cpt_direction(
    *,
    edges: list[dict[str, Any]],
    cpt_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, str], list[str]]:
    """Drop unresolved edges and relabel retained edges from fitted CPT direction.

    Args:
        edges: Original edge list.
        cpt_map: Node->CPT mapping.

    Returns:
        Tuple of sanitized edges, dropped-edge reason map keyed by original
        edge index, and relabel annotations.
    """
    sanitized: list[dict[str, Any]] = []
    dropped_reasons: dict[int, str] = {}
    relabel_annotations: list[str] = []
    for edge_index, edge in enumerate(edges):
        direction = edge_cpt_direction(edge=edge, cpt_map=cpt_map)
        if direction in {"mixed", "neutral", "missing"}:
            source_node = node_name_from_index(int(edge["from"]))
            target_node = node_name_from_index(int(edge["to"]))
            dropped_reasons[edge_index] = (
                f"{source_node}->{target_node}:cpt={direction}"
            )
            continue

        expected_sign = 1 if bool(edge.get("positive_influence", True)) else -1
        new_sign = 1 if direction == "positive" else -1
        if new_sign != expected_sign:
            source_node = node_name_from_index(int(edge["from"]))
            target_node = node_name_from_index(int(edge["to"]))
            old_sign_text = "positive" if expected_sign > 0 else "negative"
            new_sign_text = "positive" if new_sign > 0 else "negative"
            relabel_annotations.append(
                f"{source_node}->{target_node}:{old_sign_text}->{new_sign_text}"
            )

        sanitized.append(
            {
                "from": int(edge["from"]),
                "to": int(edge["to"]),
                "positive_influence": bool(new_sign > 0),
            }
        )
    return sanitized, dropped_reasons, relabel_annotations


def beliefs_reaching_target(
    *,
    num_beliefs: int,
    edges: list[dict[str, Any]],
) -> set[int]:
    """Return belief indices with a directed path to Target.

    Args:
        num_beliefs: Number of belief nodes.
        edges: Edge list.

    Returns:
        Old belief indices (1-based) that can reach Target.
    """
    reverse_adj: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        source = int(edge["from"])
        target = int(edge["to"])
        reverse_adj[target].add(source)

    reachable: set[int] = {0}
    changed = True
    while changed:
        changed = False
        for node in list(reachable):
            for source in reverse_adj.get(node, set()):
                if source not in reachable:
                    reachable.add(source)
                    changed = True

    return {index for index in range(1, num_beliefs + 1) if index in reachable}


def build_index_mapping(
    *,
    old_count: int,
    kept_old_indices: list[int],
) -> dict[int, int]:
    """Build old->new belief index mapping."""
    if len(kept_old_indices) > old_count:
        raise ValueError("kept_old_indices cannot exceed old_count.")
    return {
        old_index: new_index
        for new_index, old_index in enumerate(kept_old_indices, start=1)
    }


def project_joint_distribution(
    *,
    joint_distribution: list[dict[str, Any]],
    kept_old_indices: list[int],
) -> list[dict[str, Any]]:
    """Project joint onto retained beliefs + Target and renormalize."""
    aggregated: dict[tuple[bool, ...], float] = defaultdict(float)
    for entry in joint_distribution:
        state = entry.get("state", {})
        probability = float(entry.get("probability", 0.0))
        key_values = [
            bool(state.get(f"Belief_{index}", False)) for index in kept_old_indices
        ]
        key_values.append(bool(state.get("Target", False)))
        aggregated[tuple(key_values)] += probability

    rows: list[dict[str, Any]] = []
    for key_tuple, probability in aggregated.items():
        state: dict[str, bool] = {}
        for new_index, belief_value in enumerate(key_tuple[:-1], start=1):
            state[f"Belief_{new_index}"] = bool(belief_value)
        state["Target"] = bool(key_tuple[-1])
        rows.append({"state": state, "probability": float(probability)})

    total = sum(float(row["probability"]) for row in rows)
    if total > 0.0:
        for row in rows:
            row["probability"] = float(row["probability"]) / total
    rows.sort(key=lambda row: float(row["probability"]), reverse=True)
    return rows


def remap_edges(
    *,
    edges: list[dict[str, Any]],
    kept_old_indices: list[int],
    old_to_new: dict[int, int],
) -> list[dict[str, Any]]:
    """Remap retained edges to compact node indices.

    Args:
        edges: Edge list after inconsistency cleanup.
        kept_old_indices: Retained old belief indices.
        old_to_new: Old->new belief index map.

    Returns:
        Remapped edge list.
    """
    kept_set = set(kept_old_indices)
    remapped: list[dict[str, Any]] = []
    for edge in edges:
        source = int(edge["from"])
        target = int(edge["to"])
        if source > 0 and source not in kept_set:
            continue
        if target > 0 and target not in kept_set:
            continue
        new_source = 0 if source == 0 else old_to_new[source]
        new_target = 0 if target == 0 else old_to_new[target]
        remapped.append(
            {
                "from": int(new_source),
                "to": int(new_target),
                "positive_influence": bool(edge.get("positive_influence", True)),
            }
        )
    return remapped


def clean_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any], CleanupSummary | None]:
    """Clean one fitted proposition record.

    Args:
        record: Proposition record with bayesian_network payload.

    Returns:
        Tuple of cleaned record and summary object.
    """
    cleaned = deepcopy(record)
    payload = cleaned.get("bayesian_network", {})
    if not isinstance(payload, dict):
        return cleaned, None

    belief_nodes = payload.get("belief_nodes", [])
    edges = payload.get("edges", [])
    joint = payload.get("joint_distribution", [])
    cpt_map = payload.get("bayesian_network", {})
    if (
        not isinstance(belief_nodes, list)
        or not isinstance(edges, list)
        or not isinstance(joint, list)
        or not isinstance(cpt_map, dict)
    ):
        return cleaned, None

    edges_after_cleanup, bad_edge_reasons, relabel_annotations = (
        sanitize_edges_by_cpt_direction(edges=edges, cpt_map=cpt_map)
    )

    old_count = len(belief_nodes)
    reachable_beliefs = beliefs_reaching_target(
        num_beliefs=old_count,
        edges=edges_after_cleanup,
    )
    kept_old_indices = [
        index for index in range(1, old_count + 1) if index in reachable_beliefs
    ]
    dropped_belief_indices = {
        index for index in range(1, old_count + 1) if index not in reachable_beliefs
    }
    old_to_new = build_index_mapping(
        old_count=old_count, kept_old_indices=kept_old_indices
    )

    new_belief_nodes = [belief_nodes[index - 1] for index in kept_old_indices]
    new_edges = remap_edges(
        edges=edges_after_cleanup,
        kept_old_indices=kept_old_indices,
        old_to_new=old_to_new,
    )
    new_joint = project_joint_distribution(
        joint_distribution=joint,
        kept_old_indices=kept_old_indices,
    )
    new_cpts = refit_cpts(
        joint_distribution=new_joint,
        edges=new_edges,
        num_beliefs=len(new_belief_nodes),
    )

    payload["belief_nodes"] = new_belief_nodes
    payload["edges"] = new_edges
    payload["joint_distribution"] = new_joint
    payload["bayesian_network"] = new_cpts

    dropped_belief_texts = [
        belief_nodes[index - 1] for index in sorted(dropped_belief_indices)
    ]
    dropped_edge_texts = [
        bad_edge_reasons[index] for index in sorted(bad_edge_reasons.keys())
    ]
    drop_reasons = []
    if bad_edge_reasons:
        drop_reasons.append("inconsistent_edge_cleanup")
    if dropped_belief_indices:
        drop_reasons.append("no_path_to_target")
    if relabel_annotations:
        drop_reasons.append("edge_sign_relabel")

    summary = CleanupSummary(
        proposition_id=str(record.get("id", "")),
        proposition_source=str(record.get("proposition_source", "")),
        old_belief_count=old_count,
        new_belief_count=len(new_belief_nodes),
        dropped_belief_count=len(dropped_belief_indices),
        old_edge_count=len(edges),
        new_edge_count=len(new_edges),
        dropped_edge_count=len(bad_edge_reasons),
        relabeled_edge_count=len(relabel_annotations),
        dropped_beliefs=" || ".join(dropped_belief_texts),
        dropped_edges=" || ".join(dropped_edge_texts),
        relabeled_edges=" || ".join(relabel_annotations),
        drop_reasons=";".join(drop_reasons),
    )
    return cleaned, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write cleanup summary rows to CSV."""
    columns = [
        "file",
        "proposition_id",
        "proposition_source",
        "old_belief_count",
        "new_belief_count",
        "dropped_belief_count",
        "old_edge_count",
        "new_edge_count",
        "dropped_edge_count",
        "relabeled_edge_count",
        "dropped_beliefs",
        "dropped_edges",
        "relabeled_edges",
        "drop_reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_output_path(input_path: Path, output_dir: Path) -> Path:
    """Build cleaned output path for one input file."""
    return output_dir / f"{input_path.stem}_cleaned{input_path.suffix}"


def summary_to_row(file_label: str, summary: CleanupSummary) -> dict[str, Any]:
    """Convert summary dataclass to CSV dict row."""
    return {
        "file": file_label,
        "proposition_id": summary.proposition_id,
        "proposition_source": summary.proposition_source,
        "old_belief_count": summary.old_belief_count,
        "new_belief_count": summary.new_belief_count,
        "dropped_belief_count": summary.dropped_belief_count,
        "old_edge_count": summary.old_edge_count,
        "new_edge_count": summary.new_edge_count,
        "dropped_edge_count": summary.dropped_edge_count,
        "relabeled_edge_count": summary.relabeled_edge_count,
        "dropped_beliefs": summary.dropped_beliefs,
        "dropped_edges": summary.dropped_edges,
        "relabeled_edges": summary.relabeled_edges,
        "drop_reasons": summary.drop_reasons,
    }


def total_nodes_after_cleanup(record: dict[str, Any]) -> int | None:
    """Return total node count (Target + beliefs) for a cleaned record.

    Args:
        record: Cleaned proposition record.

    Returns:
        Total node count when BN payload is valid, otherwise None.
    """
    payload = record.get("bayesian_network", {})
    if not isinstance(payload, dict):
        return None
    belief_nodes = payload.get("belief_nodes", [])
    if not isinstance(belief_nodes, list):
        return None
    return len(belief_nodes) + 1


def main() -> None:
    """Run BN cleanup over one or more fitted BN files."""
    args = parse_args()
    input_paths = resolve_input_paths(args.inputs)
    output_dir = Path(args.output_dir)

    summary_rows: list[dict[str, Any]] = []
    total_records = 0
    total_dropped_beliefs = 0
    total_dropped_edges = 0
    total_relabeled_edges = 0
    total_min_node_filtered = 0
    all_node_count_value_counts: dict[int, int] = defaultdict(int)
    kept_node_count_value_counts: dict[int, int] = defaultdict(int)
    for input_path in input_paths:
        records = read_jsonl_records(file_path=input_path)
        cleaned_rows: list[dict[str, Any]] = []
        input_all_node_count_value_counts: dict[int, int] = defaultdict(int)
        input_kept_node_count_value_counts: dict[int, int] = defaultdict(int)
        input_min_node_filtered = 0
        for record in records:
            cleaned, summary = clean_record(record)
            total_nodes = total_nodes_after_cleanup(cleaned)
            if summary is not None:
                summary_rows.append(summary_to_row(str(input_path), summary))
                total_records += 1
                total_dropped_beliefs += int(summary.dropped_belief_count)
                total_dropped_edges += int(summary.dropped_edge_count)
                total_relabeled_edges += int(summary.relabeled_edge_count)

            if total_nodes is not None:
                input_all_node_count_value_counts[total_nodes] += 1
                all_node_count_value_counts[total_nodes] += 1
            if total_nodes is not None and total_nodes < MIN_TOTAL_NODES_TO_KEEP:
                input_min_node_filtered += 1
                total_min_node_filtered += 1
                continue

            cleaned_rows.append(cleaned)
            if total_nodes is not None:
                input_kept_node_count_value_counts[total_nodes] += 1
                kept_node_count_value_counts[total_nodes] += 1

        output_path = build_output_path(input_path, output_dir)
        write_jsonl(output_path, cleaned_rows)
        print(f"Wrote cleaned file: {output_path} ({len(cleaned_rows)} records)")
        print(
            "Node count value counts (all cleaned propositions): "
            f"{dict(sorted(input_all_node_count_value_counts.items()))}"
        )
        print(
            "Node count value counts (kept propositions): "
            f"{dict(sorted(input_kept_node_count_value_counts.items()))}"
        )
        print(
            "Dropped propositions with fewer than "
            f"{MIN_TOTAL_NODES_TO_KEEP} total nodes: {input_min_node_filtered}"
        )

    write_summary_csv(Path(args.summary_csv), summary_rows)
    print(f"Wrote cleanup summary: {args.summary_csv}")
    print(
        "Overall node count value counts (all cleaned propositions): "
        f"{dict(sorted(all_node_count_value_counts.items()))}"
    )
    print(
        "Overall node count value counts (kept propositions): "
        f"{dict(sorted(kept_node_count_value_counts.items()))}"
    )
    print(
        "Dropped propositions with fewer than "
        f"{MIN_TOTAL_NODES_TO_KEEP} total nodes overall: {total_min_node_filtered}"
    )
    print(
        "Processed "
        f"{total_records} propositions; dropped {total_dropped_beliefs} beliefs "
        f"and {total_dropped_edges} edges total; relabeled "
        f"{total_relabeled_edges} retained edges."
    )


if __name__ == "__main__":
    main()
