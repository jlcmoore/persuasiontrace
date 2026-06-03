"""
Audit fitted Bayesian-network edge directions for sign consistency.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation.io import read_jsonl_records

SIGN_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EdgeAuditContext:
    """Per-proposition context carried to per-edge auditing."""

    file_label: str
    proposition_source: str
    proposition_id: str
    joint_distribution: list[dict[str, Any]]
    cpt_map: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for BN sign-consistency auditing.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether each edge's expected sign (positive_influence) "
            "matches fitted CPT-implied direction and joint-marginal direction."
        )
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="src/simulation/data/fitted_bayesian_networks_*.jsonl",
        help=(
            "Glob pattern or comma-separated file paths for fitted BN JSONLs. "
            "Default scans all fitted BN files in src/simulation/data."
        ),
    )
    parser.add_argument(
        "--output-edges-csv",
        type=str,
        default="analysis/data/bn_sign_consistency_edges.csv",
        help="Per-edge output CSV path.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        default="analysis/data/bn_sign_consistency_summary.csv",
        help="Per-file summary CSV path.",
    )
    return parser.parse_args()


def resolve_input_paths(inputs_arg: str) -> list[Path]:
    """Resolve input argument into an ordered list of existing paths.

    Args:
        inputs_arg: Glob string or comma-separated file paths.

    Returns:
        Existing paths matching the input expression.
    """
    input_text = inputs_arg.strip()
    if "," in input_text:
        candidates = [
            Path(item.strip()) for item in input_text.split(",") if item.strip()
        ]
    else:
        candidates = sorted(Path(".").glob(input_text))

    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise ValueError(f"No input files found for --inputs={inputs_arg!r}.")
    return existing


def node_name_from_index(index: int) -> str:
    """Map edge index value to node identifier.

    Args:
        index: Edge node index where 0 denotes Target and 1..N are belief nodes.

    Returns:
        Node identifier string.
    """
    if index == 0:
        return "Target"
    return f"Belief_{index}"


def sign_of(value: float) -> int:
    """Return sign bucket for one value with tolerance.

    Args:
        value: Numeric value to sign.

    Returns:
        +1, -1, or 0 for positive, negative, or near-zero.
    """
    if value > SIGN_TOLERANCE:
        return 1
    if value < -SIGN_TOLERANCE:
        return -1
    return 0


def cpt_key_from_assignment(parents: list[str], assignment: dict[str, bool]) -> str:
    """Build CPT key string from a parent assignment.

    Args:
        parents: Ordered parent node ids.
        assignment: Parent truth assignment mapping.

    Returns:
        CPT key string.
    """
    if not parents:
        return "prior"
    return ",".join(f"{parent}={assignment[parent]}" for parent in parents)


def marginal_delta(
    joint_distribution: list[dict[str, Any]],
    source_node: str,
    target_node: str,
) -> float | None:
    """Compute pairwise marginal direction between source and target node.

    Args:
        joint_distribution: List of joint-distribution entries.
        source_node: Source node id.
        target_node: Target node id.

    Returns:
        Difference P(target=True|source=True)-P(target=True|source=False), or
        ``None`` when one side has zero support.
    """
    source_true_mass = 0.0
    source_false_mass = 0.0
    target_true_given_source_true = 0.0
    target_true_given_source_false = 0.0
    for entry in joint_distribution:
        state = entry.get("state", {})
        probability = float(entry.get("probability", 0.0))
        source_value = state.get(source_node)
        if source_value is True:
            source_true_mass += probability
            if state.get(target_node) is True:
                target_true_given_source_true += probability
        elif source_value is False:
            source_false_mass += probability
            if state.get(target_node) is True:
                target_true_given_source_false += probability

    if source_true_mass <= 0.0 or source_false_mass <= 0.0:
        return None
    p_true = target_true_given_source_true / source_true_mass
    p_false = target_true_given_source_false / source_false_mass
    return float(p_true - p_false)


def cpt_context_deltas(
    *,
    cpt_node_spec: dict[str, Any],
    source_node: str,
) -> list[float]:
    """Compute CPT deltas for one source->child edge over all parent contexts.

    Args:
        cpt_node_spec: Child-node CPT payload with parents and probabilities.
        source_node: Source parent node id.

    Returns:
        List of per-context deltas:
        P(child=True|source=True,others)-P(child=True|source=False,others).
    """
    parents = cpt_node_spec.get("parents", [])
    probabilities = cpt_node_spec.get("probabilities", {})
    if not isinstance(parents, list) or source_node not in parents:
        return []
    if not isinstance(probabilities, dict):
        return []

    other_parents = [parent for parent in parents if parent != source_node]
    deltas: list[float] = []
    for values in itertools.product([True, False], repeat=len(other_parents)):
        assignment = dict(zip(other_parents, values))
        true_assignment = dict(assignment)
        false_assignment = dict(assignment)
        true_assignment[source_node] = True
        false_assignment[source_node] = False

        key_true = cpt_key_from_assignment(parents, true_assignment)
        key_false = cpt_key_from_assignment(parents, false_assignment)
        if key_true not in probabilities or key_false not in probabilities:
            continue
        probability_true = float(probabilities[key_true])
        probability_false = float(probabilities[key_false])
        deltas.append(probability_true - probability_false)
    return deltas


def classify_context_direction(deltas: list[float]) -> str:
    """Classify context-wise CPT direction from per-context deltas.

    Args:
        deltas: Per-context source-toggle deltas.

    Returns:
        One of: ``positive``, ``negative``, ``neutral``, ``mixed``, ``missing``.
    """
    if not deltas:
        return "missing"
    signs = {sign_of(delta) for delta in deltas}
    if signs == {1}:
        return "positive"
    if signs == {-1}:
        return "negative"
    if signs == {0}:
        return "neutral"
    return "mixed"


def sign_label(sign_value: int | None) -> str:
    """Map numeric sign to textual direction label.

    Args:
        sign_value: Sign bucket (+1, -1, 0, or None).

    Returns:
        Direction label string.
    """
    if sign_value == 1:
        return "positive"
    if sign_value == -1:
        return "negative"
    if sign_value == 0:
        return "neutral"
    return "missing"


def cpt_consistency_for_direction(
    *,
    expected_sign: int,
    context_direction: str,
) -> bool | None:
    """Resolve CPT consistency flag for one edge direction comparison.

    Args:
        expected_sign: Expected edge sign (+1 or -1).
        context_direction: Context-wise CPT direction label.

    Returns:
        True/False consistency when resolvable, otherwise None.
    """
    if context_direction == "positive":
        return expected_sign > 0
    if context_direction == "negative":
        return expected_sign < 0
    if context_direction in {"mixed", "neutral"}:
        return False
    return None


def compute_context_metrics(
    *,
    expected_sign: int,
    source_node: str,
    target_node: str,
    cpt_map: dict[str, Any],
) -> tuple[str, float | None, int, float | None, bool | None]:
    """Compute context-conditioned CPT metrics for one source->target edge.

    Args:
        expected_sign: Expected edge sign (+1 or -1).
        source_node: Source node identifier.
        target_node: Target node identifier.
        cpt_map: Node->CPT mapping.

    Returns:
        Tuple of context direction, mean delta, context count, match fraction,
        and CPT-consistency flag.
    """
    cpt_node_spec = cpt_map.get(target_node, {})
    context_deltas: list[float] = []
    if isinstance(cpt_node_spec, dict):
        context_deltas = cpt_context_deltas(
            cpt_node_spec=cpt_node_spec,
            source_node=source_node,
        )
    context_direction = classify_context_direction(context_deltas)
    context_signs = [sign_of(delta) for delta in context_deltas]
    total_contexts = len(context_signs)
    matched_contexts = sum(1 for value in context_signs if value == expected_sign)
    context_match_fraction = (
        float(matched_contexts) / float(total_contexts) if total_contexts > 0 else None
    )
    cpt_consistent = cpt_consistency_for_direction(
        expected_sign=expected_sign,
        context_direction=context_direction,
    )
    mean_context_delta = (
        sum(context_deltas) / len(context_deltas) if context_deltas else None
    )
    return (
        context_direction,
        mean_context_delta,
        total_contexts,
        context_match_fraction,
        cpt_consistent,
    )


def compute_marginal_metrics(
    *,
    expected_sign: int,
    source_node: str,
    target_node: str,
    joint_distribution: list[dict[str, Any]],
) -> tuple[float | None, str, bool | None]:
    """Compute marginal sign metrics for one source->target pair.

    Args:
        expected_sign: Expected edge sign (+1 or -1).
        source_node: Source node identifier.
        target_node: Target node identifier.
        joint_distribution: Proposition joint distribution.

    Returns:
        Tuple of marginal delta, marginal direction label, and consistency flag.
    """
    marginal = marginal_delta(
        joint_distribution=joint_distribution,
        source_node=source_node,
        target_node=target_node,
    )
    marginal_sign = sign_of(marginal) if marginal is not None else None
    marginal_direction = sign_label(marginal_sign)
    marginal_consistent = (
        (marginal_sign == expected_sign) if marginal_sign in {-1, 1} else None
    )
    return marginal, marginal_direction, marginal_consistent


def audit_edge(
    *,
    context: EdgeAuditContext,
    edge_index: int,
    edge: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one audit row for a single edge.

    Args:
        context: Per-proposition audit context.
        edge_index: Edge index within proposition edge list.
        edge: Edge payload with ``from``, ``to``, and ``positive_influence``.

    Returns:
        Per-edge audit row, or None when edge payload is invalid.
    """
    source_idx = edge.get("from")
    target_idx = edge.get("to")
    positive_influence = bool(edge.get("positive_influence", True))
    if not isinstance(source_idx, int) or not isinstance(target_idx, int):
        return None

    source_node = node_name_from_index(source_idx)
    target_node = node_name_from_index(target_idx)
    expected_sign = 1 if positive_influence else -1

    context_metrics = compute_context_metrics(
        expected_sign=expected_sign,
        source_node=source_node,
        target_node=target_node,
        cpt_map=context.cpt_map,
    )
    marginal_metrics = compute_marginal_metrics(
        expected_sign=expected_sign,
        source_node=source_node,
        target_node=target_node,
        joint_distribution=context.joint_distribution,
    )

    return {
        "file": context.file_label,
        "proposition_source": context.proposition_source,
        "proposition_id": context.proposition_id,
        "edge_index": edge_index,
        "source_node": source_node,
        "target_node": target_node,
        "expected_direction": sign_label(expected_sign),
        "cpt_direction": context_metrics[0],
        "cpt_mean_delta": context_metrics[1],
        "cpt_context_count": context_metrics[2],
        "cpt_context_match_fraction": context_metrics[3],
        "cpt_consistent": context_metrics[4],
        "marginal_delta": marginal_metrics[0],
        "marginal_direction": marginal_metrics[1],
        "marginal_consistent": marginal_metrics[2],
    }


def audit_record(
    *,
    file_label: str,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Audit one proposition record and return one row per edge.

    Args:
        file_label: Source file label for reporting.
        record: Fitted proposition record.

    Returns:
        List of per-edge audit rows.
    """
    proposition_id = str(record.get("id", ""))
    proposition_source = str(record.get("proposition_source", ""))
    bayes_payload = record.get("bayesian_network", {})
    if not isinstance(bayes_payload, dict):
        return []

    edges = bayes_payload.get("edges", [])
    joint_distribution = bayes_payload.get("joint_distribution", [])
    cpt_map = bayes_payload.get("bayesian_network", {})
    if (
        not isinstance(edges, list)
        or not isinstance(joint_distribution, list)
        or not isinstance(cpt_map, dict)
    ):
        return []

    context = EdgeAuditContext(
        file_label=file_label,
        proposition_source=proposition_source,
        proposition_id=proposition_id,
        joint_distribution=joint_distribution,
        cpt_map=cpt_map,
    )
    rows: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        row = audit_edge(
            context=context,
            edge_index=edge_index,
            edge=edge,
        )
        if row is not None:
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate edge audit rows to one summary row per file.

    Args:
        rows: Per-edge audit rows.

    Returns:
        Summary rows.
    """
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        file_label = str(row["file"])
        by_file.setdefault(file_label, []).append(row)

    summaries: list[dict[str, Any]] = []
    for file_label, file_rows in sorted(by_file.items()):
        edge_count = len(file_rows)
        cpt_consistent_count = sum(
            1 for row in file_rows if row.get("cpt_consistent") is True
        )
        cpt_inconsistent_count = sum(
            1 for row in file_rows if row.get("cpt_consistent") is False
        )
        cpt_missing_count = sum(
            1 for row in file_rows if row.get("cpt_consistent") is None
        )
        cpt_mixed_count = sum(
            1 for row in file_rows if row.get("cpt_direction") == "mixed"
        )
        marginal_consistent_count = sum(
            1 for row in file_rows if row.get("marginal_consistent") is True
        )
        marginal_inconsistent_count = sum(
            1 for row in file_rows if row.get("marginal_consistent") is False
        )
        marginal_missing_count = sum(
            1 for row in file_rows if row.get("marginal_consistent") is None
        )
        summaries.append(
            {
                "file": file_label,
                "edge_count": edge_count,
                "cpt_consistent_count": cpt_consistent_count,
                "cpt_inconsistent_count": cpt_inconsistent_count,
                "cpt_missing_or_unresolved_count": cpt_missing_count,
                "cpt_mixed_count": cpt_mixed_count,
                "marginal_consistent_count": marginal_consistent_count,
                "marginal_inconsistent_count": marginal_inconsistent_count,
                "marginal_missing_or_unresolved_count": marginal_missing_count,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write dict rows to a CSV file.

    Args:
        path: Output path.
        rows: Rows to write.
        columns: Column ordering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the BN sign-consistency audit CLI."""
    args = parse_args()
    input_paths = resolve_input_paths(args.inputs)
    all_rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        records = read_jsonl_records(file_path=input_path)
        for record in records:
            all_rows.extend(
                audit_record(
                    file_label=str(input_path),
                    record=record,
                )
            )

    edge_columns = [
        "file",
        "proposition_source",
        "proposition_id",
        "edge_index",
        "source_node",
        "target_node",
        "expected_direction",
        "cpt_direction",
        "cpt_mean_delta",
        "cpt_context_count",
        "cpt_context_match_fraction",
        "cpt_consistent",
        "marginal_delta",
        "marginal_direction",
        "marginal_consistent",
    ]
    write_csv(Path(args.output_edges_csv), all_rows, edge_columns)

    summary_rows = summarize_rows(all_rows)
    summary_columns = [
        "file",
        "edge_count",
        "cpt_consistent_count",
        "cpt_inconsistent_count",
        "cpt_missing_or_unresolved_count",
        "cpt_mixed_count",
        "marginal_consistent_count",
        "marginal_inconsistent_count",
        "marginal_missing_or_unresolved_count",
    ]
    write_csv(Path(args.output_summary_csv), summary_rows, summary_columns)

    print(f"Audited {len(all_rows)} edges across {len(input_paths)} files.")
    print(f"Wrote per-edge audit to {args.output_edges_csv}")
    print(f"Wrote summary audit to {args.output_summary_csv}")


if __name__ == "__main__":
    main()
