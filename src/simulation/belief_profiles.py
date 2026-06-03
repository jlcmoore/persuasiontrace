"""Reusable utilities for node-level belief profile analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simulation.human_likeness import (
    RoundTrajectory,
    apply_proposition_matching,
    select_human_reference,
    select_simulator,
)

BELIEF_PROFILE_CORPUS_ORDER = [
    "human_reference",
    "vanilla_llm_target",
    "structure_target",
    "full_simulated_target",
    "full_no_rhetoric_target",
]


@dataclass(frozen=True)
class BeliefProfileRow:
    """One extracted round-level belief profile."""

    corpus: str
    proposition: str
    supports_proposition: bool
    pre_target: float
    post_target: float
    pre_nodes: dict[str, float]
    post_nodes: dict[str, float]

    @property
    def target_delta(self) -> float:
        """Compute the target belief change for this profile row."""
        return float(self.post_target - self.pre_target)


def belief_sort_key(node_id: str) -> tuple[int, str]:
    """Sort belief ids by numeric suffix when present."""
    if node_id.startswith("Belief_"):
        suffix = node_id.replace("Belief_", "", 1)
        if suffix.isdigit():
            return int(suffix), node_id
    return 10**9, node_id


def extract_node_payload(payload: Any) -> dict[str, float] | None:
    """Validate and normalize one node-belief mapping payload."""
    if not isinstance(payload, dict):
        return None
    normalized: dict[str, float] = {}
    for key_raw, value_raw in payload.items():
        key = str(key_raw).strip()
        if not key:
            continue
        if not isinstance(value_raw, (int, float)):
            return None
        value = float(value_raw)
        if not 0 <= value <= 1:
            return None
        normalized[key] = value
    return normalized or None


def profile_row_from_trajectory(
    corpus: str,
    row: RoundTrajectory,
) -> BeliefProfileRow | None:
    """Convert one trajectory row into a node-level profile row."""
    round_obj = row.round_obj
    pre_target = round_obj.target_initial_belief
    post_target = round_obj.target_final_belief
    supports = round_obj.persuader_supports_proposition
    if not isinstance(pre_target, (int, float)):
        return None
    if not isinstance(post_target, (int, float)):
        return None
    if not isinstance(supports, bool):
        return None

    pre_nodes = extract_node_payload(round_obj.target_initial_node_beliefs)
    post_nodes = extract_node_payload(round_obj.target_final_node_beliefs)
    if pre_nodes is None or post_nodes is None:
        return None
    if set(pre_nodes) != set(post_nodes):
        return None

    return BeliefProfileRow(
        corpus=corpus,
        proposition=row.proposition,
        supports_proposition=supports,
        pre_target=float(pre_target),
        post_target=float(post_target),
        pre_nodes=pre_nodes,
        post_nodes=post_nodes,
    )


def build_belief_profile_corpora(
    rows: list[RoundTrajectory],
    *,
    human_source: str,
    proposition_match: str,
    include_vanilla_llm_target: bool,
    selector_kwargs: dict[str, Any],
) -> list[tuple[str, list[RoundTrajectory]]]:
    """Select and proposition-align human/simulator corpora."""
    human_rows = select_human_reference(
        rows,
        human_source=human_source,
        **selector_kwargs,
    )
    structure_rows = select_simulator(
        rows,
        simulator_type="structure",
        **selector_kwargs,
    )
    full_rows = select_simulator(
        rows,
        simulator_type="full",
        **selector_kwargs,
    )
    vanilla_rows = (
        select_simulator(
            rows,
            simulator_type="vanilla",
            **selector_kwargs,
        )
        if include_vanilla_llm_target
        else []
    )
    no_rhet_rows = select_simulator(
        rows,
        simulator_type="full_no_rhetoric",
        **selector_kwargs,
    )

    (
        human_rows,
        structure_rows,
        full_rows,
        vanilla_rows,
    ) = apply_proposition_matching(
        human_rows,
        structure_rows,
        full_rows,
        vanilla_rows,
        mode=proposition_match,
    )

    corpora: list[tuple[str, list[RoundTrajectory]]] = [
        ("human_reference", human_rows),
    ]
    if include_vanilla_llm_target:
        corpora.append(("vanilla_llm_target", vanilla_rows))
    corpora.extend(
        [
            ("structure_target", structure_rows),
            ("full_simulated_target", full_rows),
            ("full_no_rhetoric_target", no_rhet_rows),
        ]
    )
    return corpora


def assign_bin_label(vector: np.ndarray, thresholds: np.ndarray) -> str:
    """Assign a low/mid/high bin label for each node and join them."""
    labels: list[str] = []
    for idx, value in enumerate(vector):
        low_hi = thresholds[idx]
        low = float(low_hi[0])
        high = float(low_hi[1])
        if value < low:
            tag = "L"
        elif value > high:
            tag = "H"
        else:
            tag = "M"
        labels.append(f"B{idx + 1}={tag}")
    return "|".join(labels)
