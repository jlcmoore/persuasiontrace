"""Shared helpers for human/simulator round matching and rendering.

This module centralizes helpers used by:
- ``analysis.find_human_sim_overlap``
- ``analysis.render_human_sim_round_figure``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def belief_bin(value: float) -> str:
    """Map a belief value in ``[0, 1]`` to the simulator bin label.

    Args:
        value: Belief probability to map.

    Returns:
        Bin name for the value.

    Raises:
        ValueError: If ``value`` falls outside ``[0, 1]``.
    """
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Belief must be in [0,1], got {value!r}")
    if value < 0.10:
        return "very_low"
    if value < 0.35:
        return "low"
    if value < 0.65:
        return "mid"
    if value < 0.90:
        return "high"
    return "very_high"


def infer_stance(target_initial_belief: float) -> bool:
    """Infer persuader stance when round metadata omits it.

    Args:
        target_initial_belief: Initial target belief.

    Returns:
        ``True`` when the persuader supports the proposition and ``False``
        otherwise, following the paper convention that ``<= 0.5`` implies
        support.
    """
    return target_initial_belief <= 0.5


def parse_jsonl_dict_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Parse JSONL file and flatten object records with source line numbers.

    Args:
        path: JSONL source file.

    Returns:
        A list of ``(line_number, payload_dict)`` tuples. Lines containing JSON
        arrays are flattened to include object members.
    """
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        records.append((line_idx, item))
                continue
            if isinstance(payload, dict):
                records.append((line_idx, payload))
    return records


def extract_belief_node_names_from_bn(
    bn_payload: dict[str, Any], distribution: list[Any]
) -> list[str]:
    """Extract canonical ``Belief_*`` node names from BN metadata.

    Args:
        bn_payload: Bayesian-network metadata payload.
        distribution: Initial distribution rows, used as a fallback source.

    Returns:
        Sorted ``Belief_*`` node ids when available.
    """
    all_nodes = bn_payload.get("all_nodes")
    if isinstance(all_nodes, list):
        names = [item for item in all_nodes if isinstance(item, str)]
        belief_names = [item for item in names if item.startswith("Belief_")]
        if belief_names:
            return sorted(belief_names)

    node_to_text = bn_payload.get("node_to_text")
    if isinstance(node_to_text, dict):
        keys = [key for key in node_to_text.keys() if isinstance(key, str)]
        belief_keys = [key for key in keys if key.startswith("Belief_")]
        if belief_keys:
            return sorted(belief_keys)

    belief_nodes = bn_payload.get("belief_nodes")
    if isinstance(belief_nodes, list):
        return [f"Belief_{idx + 1}" for idx, _ in enumerate(belief_nodes)]

    if distribution and isinstance(distribution[0], dict):
        state = distribution[0].get("state")
        if isinstance(state, dict):
            keys = [key for key in state.keys() if isinstance(key, str)]
            belief_keys = [key for key in keys if key.startswith("Belief_")]
            if belief_keys:
                return sorted(belief_keys)
    return []


def _step_index(row: dict[str, Any]) -> int:
    """Return sortable step index for simulator rows.

    Args:
        row: One simulator step row.

    Returns:
        Integer step index, defaulting to ``0`` for malformed values.
    """
    raw = row.get("step_index", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _parse_round_trace_from_env_state(
    raw_state: Any,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Parse one environment state JSON string into round/trace payloads.

    Args:
        raw_state: Raw ``pre_step_env_state_json`` or ``post_step_env_state_json``
            value.

    Returns:
        ``(round_payload, trace_payload)`` if parsing succeeds, otherwise
        ``None``.
    """
    if not isinstance(raw_state, str) or not raw_state.strip():
        return None
    try:
        payload = json.loads(raw_state)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    round_payload = payload.get("round")
    if not isinstance(round_payload, dict):
        return None
    trace_payload = round_payload.get("simulated_target_trace")
    if isinstance(trace_payload, dict):
        return round_payload, trace_payload
    target_payload = payload.get("target")
    if isinstance(target_payload, dict):
        return round_payload, target_payload
    return None


def iter_round_trace_snapshots(
    step_rows: list[dict[str, Any]],
    *,
    reverse_steps: bool = False,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield parsed round/trace snapshots from simulator step rows.

    Args:
        step_rows: Step rows for a simulator episode.
        reverse_steps: Whether to scan later step indices first.

    Yields:
        Tuples of ``(round_payload, trace_payload)`` for each parseable
        environment-state snapshot.
    """
    ordered = sorted(step_rows, key=_step_index, reverse=reverse_steps)
    for row in ordered:
        for key in ("pre_step_env_state_json", "post_step_env_state_json"):
            parsed = _parse_round_trace_from_env_state(row.get(key))
            if parsed is not None:
                yield parsed
