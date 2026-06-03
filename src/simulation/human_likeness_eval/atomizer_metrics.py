"""Atomizer-target alignment metrics for human-likeness evaluation."""

from __future__ import annotations

import numpy as np

from simulation.human_likeness import RoundTrajectory
from simulation.human_likeness_eval.corpus_variants import (
    _split_policy_variant_corpus,
)


def _atom_targets_target(atom: dict[str, object]) -> bool:
    """
    Determine whether one atom directly references the target proposition.

    Args:
        atom: Atom payload from ``simulated_target_trace.atom_history``.

    Returns:
        True when the atom targets ``Target`` in belief or edge targets.
    """
    belief_targets = atom.get("belief_targets")
    if isinstance(belief_targets, list):
        for target in belief_targets:
            if not isinstance(target, dict):
                continue
            if str(target.get("belief_id", "")) == "Target":
                return True

    edge_targets = atom.get("edge_targets")
    if isinstance(edge_targets, list):
        for edge in edge_targets:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source == "Target" or target == "Target":
                return True
    return False


def _atomizer_alignment_rows_for_corpus(
    *,
    corpus: str,
    rows: list[RoundTrajectory],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """
    Extract target-directed atomizer direction rows for one corpus.

    Args:
        corpus: Corpus label.
        rows: Round trajectory rows.

    Returns:
        Tuple of atom-level rows and corpus-level counts.
    """
    atom_rows: list[dict[str, object]] = []
    rounds_with_trace = 0
    rounds_with_target_atoms: set[int] = set()
    atoms_total = 0
    atoms_target_directed = 0

    for trajectory_index, row in enumerate(rows):
        trace_payload = row.round_obj.simulated_target_trace
        if not isinstance(trace_payload, dict):
            continue
        atom_history = trace_payload.get("atom_history")
        if not isinstance(atom_history, list) or not atom_history:
            continue
        rounds_with_trace += 1
        supports = row.round_obj.persuader_supports_proposition

        for turn_index, atom_list in enumerate(atom_history):
            if not isinstance(atom_list, list):
                continue
            for atom_index, atom in enumerate(atom_list):
                if not isinstance(atom, dict):
                    continue
                atoms_total += 1
                if not _atom_targets_target(atom):
                    continue

                try:
                    p_support = float(atom.get("p_support"))
                except (TypeError, ValueError):
                    continue
                atoms_target_directed += 1
                rounds_with_target_atoms.add(int(trajectory_index))

                is_neutral = bool(np.isclose(p_support, 0.5))
                is_aligned: bool | None = None
                if isinstance(supports, bool) and not is_neutral:
                    is_aligned = p_support > 0.5 if supports else p_support < 0.5

                atom_rows.append(
                    {
                        "corpus": corpus,
                        "trajectory_index": int(trajectory_index),
                        "turn_index": int(turn_index),
                        "atom_index": int(atom_index),
                        "proposition": row.proposition,
                        "supports_proposition": supports,
                        "p_support": p_support,
                        "is_neutral": int(is_neutral),
                        "is_aligned": (
                            "" if is_aligned is None else int(bool(is_aligned))
                        ),
                        "text_span": str(atom.get("text_span", "")),
                        "source_path": str(row.source_path),
                        "source_line_index": int(row.source_line_index),
                        "source_round_index": row.source_round_index,
                    }
                )

    counts = {
        "rounds_total": int(len(rows)),
        "rounds_with_trace": int(rounds_with_trace),
        "rounds_with_target_directed_atoms": int(len(rounds_with_target_atoms)),
        "atoms_total": int(atoms_total),
        "atoms_target_directed": int(atoms_target_directed),
    }
    return atom_rows, counts


def _atomizer_alignment_summary_row(
    *,
    corpus: str,
    atom_rows: list[dict[str, object]],
    counts: dict[str, int],
) -> dict[str, object]:
    """
    Summarize atomizer target-direction alignment for one corpus.

    Args:
        corpus: Corpus label.
        atom_rows: Atom rows from `_atomizer_alignment_rows_for_corpus`.
        counts: Corpus-level count dictionary.

    Returns:
        One summary row.
    """
    p_values = np.asarray(
        [float(item["p_support"]) for item in atom_rows],
        dtype=float,
    )
    neutral_mask = np.asarray(
        [bool(int(item["is_neutral"])) for item in atom_rows],
        dtype=bool,
    )
    non_neutral_rows = [
        item
        for item in atom_rows
        if not bool(int(item["is_neutral"])) and item.get("is_aligned") in (0, 1)
    ]
    aligned_values = np.asarray(
        [int(item["is_aligned"]) for item in non_neutral_rows],
        dtype=float,
    )
    supports_true_rows = [
        item for item in non_neutral_rows if item.get("supports_proposition") is True
    ]
    supports_false_rows = [
        item for item in non_neutral_rows if item.get("supports_proposition") is False
    ]
    supports_true_aligned = np.asarray(
        [int(item["is_aligned"]) for item in supports_true_rows],
        dtype=float,
    )
    supports_false_aligned = np.asarray(
        [int(item["is_aligned"]) for item in supports_false_rows],
        dtype=float,
    )
    supports_true_rate = (
        float(np.mean(supports_true_aligned))
        if supports_true_aligned.size > 0
        else float("nan")
    )
    supports_false_rate = (
        float(np.mean(supports_false_aligned))
        if supports_false_aligned.size > 0
        else float("nan")
    )
    balanced_aligned_rate = (
        float((supports_true_rate + supports_false_rate) / 2.0)
        if np.isfinite(supports_true_rate) and np.isfinite(supports_false_rate)
        else float("nan")
    )
    symmetry_gap = (
        float(supports_true_rate - supports_false_rate)
        if np.isfinite(supports_true_rate) and np.isfinite(supports_false_rate)
        else float("nan")
    )

    atoms_target_directed = int(counts.get("atoms_target_directed", 0))
    return {
        "corpus": corpus,
        "rounds_total": int(counts.get("rounds_total", 0)),
        "rounds_with_trace": int(counts.get("rounds_with_trace", 0)),
        "rounds_with_target_directed_atoms": int(
            counts.get("rounds_with_target_directed_atoms", 0)
        ),
        "atoms_total": int(counts.get("atoms_total", 0)),
        "atoms_target_directed": atoms_target_directed,
        "atoms_target_directed_non_neutral": int(aligned_values.size),
        "mean_p_support_target_directed": (
            float(np.mean(p_values)) if p_values.size > 0 else float("nan")
        ),
        "neutral_rate_target_directed": (
            float(np.mean(neutral_mask)) if atoms_target_directed > 0 else float("nan")
        ),
        "aligned_rate_target_directed_non_neutral": (
            float(np.mean(aligned_values)) if aligned_values.size > 0 else float("nan")
        ),
        "balanced_aligned_rate": balanced_aligned_rate,
        "symmetry_gap_supports_true_minus_false": symmetry_gap,
        "supports_true_n": int(supports_true_aligned.size),
        "supports_true_aligned_rate": supports_true_rate,
        "supports_false_n": int(supports_false_aligned.size),
        "supports_false_aligned_rate": supports_false_rate,
    }


def _atomizer_proposition_bias_rows(
    atom_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Aggregate target-directed atomizer evidence by proposition.

    Args:
        atom_rows: Atom rows from `_atomizer_alignment_rows_for_corpus`.

    Returns:
        Proposition-level rows containing average target-direction evidence.
    """
    grouped: dict[tuple[str, str, bool], list[dict[str, object]]] = {}
    for item in atom_rows:
        corpus = str(item.get("corpus", "")).strip()
        proposition = str(item.get("proposition", "")).strip()
        supports = item.get("supports_proposition")
        if not corpus or not proposition or not isinstance(supports, bool):
            continue
        grouped.setdefault((corpus, proposition, supports), []).append(item)

    rows: list[dict[str, object]] = []
    for corpus, proposition, supports in sorted(grouped):
        items = grouped[(corpus, proposition, supports)]
        p_support = np.asarray(
            [float(item["p_support"]) for item in items],
            dtype=float,
        )
        if p_support.size == 0:
            continue
        goal_alignment = p_support if supports else (1.0 - p_support)
        signed_goal_alignment = 2.0 * (goal_alignment - 0.5)
        aligned_non_neutral = np.asarray(
            [
                float(item["is_aligned"])
                for item in items
                if item.get("is_aligned") in (0, 1)
            ],
            dtype=float,
        )
        base_corpus, policy_model = _split_policy_variant_corpus(corpus)
        rows.append(
            {
                "corpus": corpus,
                "base_corpus": base_corpus,
                "policy_model": policy_model or "",
                "proposition": proposition,
                "supports_proposition": supports,
                "n_atoms": int(p_support.size),
                "mean_p_support": float(np.mean(p_support)),
                "mean_goal_alignment": float(np.mean(goal_alignment)),
                "mean_signed_goal_alignment": float(np.mean(signed_goal_alignment)),
                "aligned_rate_non_neutral": (
                    float(np.mean(aligned_non_neutral))
                    if aligned_non_neutral.size > 0
                    else float("nan")
                ),
            }
        )
    return rows


def _atomizer_proposition_bias_summary_rows(
    proposition_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Summarize proposition-level atomizer evidence spread by corpus and stance.

    Args:
        proposition_rows: Rows from `_atomizer_proposition_bias_rows`.

    Returns:
        Summary rows over proposition means.
    """
    grouped: dict[tuple[str, bool], list[float]] = {}
    metadata: dict[tuple[str, bool], tuple[str, str]] = {}
    for row in proposition_rows:
        corpus = str(row.get("corpus", ""))
        supports = row.get("supports_proposition")
        if not corpus or not isinstance(supports, bool):
            continue
        mean_alignment = float(row.get("mean_goal_alignment", float("nan")))
        if not np.isfinite(mean_alignment):
            continue
        grouped.setdefault((corpus, supports), []).append(mean_alignment)
        base_corpus = str(row.get("base_corpus", ""))
        policy_model = str(row.get("policy_model", ""))
        metadata[(corpus, supports)] = (base_corpus, policy_model)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        corpus, supports = key
        values = np.asarray(grouped[key], dtype=float)
        base_corpus, policy_model = metadata.get(key, ("", ""))
        summary_rows.append(
            {
                "corpus": corpus,
                "base_corpus": base_corpus,
                "policy_model": policy_model,
                "supports_proposition": supports,
                "n_propositions": int(values.size),
                "mean_of_prop_mean_goal_alignment": float(np.mean(values)),
                "std_of_prop_mean_goal_alignment": (
                    float(np.std(values)) if values.size > 1 else 0.0
                ),
                "min_prop_mean_goal_alignment": float(np.min(values)),
                "max_prop_mean_goal_alignment": float(np.max(values)),
            }
        )
    return summary_rows
