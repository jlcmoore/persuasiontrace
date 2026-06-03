"""Find overlapping human/simulator rounds under proposition and bin constraints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from analysis.human_sim_round_common import (
    belief_bin,
    extract_belief_node_names_from_bn,
    infer_stance,
    iter_round_trace_snapshots,
    parse_jsonl_dict_records,
)


@dataclass(frozen=True)
class HumanRound:
    """Human round candidate with BN node-level initial beliefs."""

    source_file: str
    source_line: int
    proposition: str
    persuader_supports_proposition: bool
    target_initial_belief: float
    target_final_belief: float
    target_initial_bin: str
    node_initial_beliefs: dict[str, float]
    node_initial_bins: dict[str, str]
    raw_delta: float
    persuader_relative_delta: float
    timed_out: bool
    persuader_turns: int


@dataclass(frozen=True)
class SimRound:
    """Simulator round candidate with reconstructed node-level initial beliefs."""

    episode_id: str
    source_file: str
    proposition: str
    persuader_supports_proposition: bool
    target_initial_belief: float
    target_final_belief: float
    target_initial_bin: str
    node_initial_beliefs: dict[str, float]
    node_initial_bins: dict[str, str]
    raw_delta: float
    persuader_relative_delta: float
    persona: str | None
    simulated_target_no_rhetoric: bool
    policy_model: str | None
    persuader_turns: int


@dataclass(frozen=True)
class MatchCandidate:
    """Matched human/simulator pair."""

    proposition: str
    human_source: str
    sim_episode_id: str
    stance_supports_proposition: bool
    human_initial: float
    sim_initial: float
    initial_abs_diff: float
    target_bin: str
    node_bins: dict[str, str]
    human_raw_delta: float
    sim_raw_delta: float
    human_persuader_relative_delta: float
    sim_persuader_relative_delta: float
    movement_score: float


def persuader_relative_delta(
    *, initial: float, final: float, supports_proposition: bool
) -> float:
    """Compute persuader-relative delta from initial/final target beliefs."""
    return (final - initial) if supports_proposition else (initial - final)


def count_persuader_turns(messages: Any) -> int:
    """Count persuader turns from message list payload."""
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "persuader":
            total += 1
    return total


def _to_float_map(payload: Any) -> dict[str, float] | None:
    """Convert payload to a map of finite float values."""
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, float] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str):
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        parsed[raw_key] = value
    return parsed


def _node_probs_from_distribution_history(
    trace_payload: dict[str, Any],
) -> dict[str, float] | None:
    """Compute initial node marginals from simulator distribution history."""
    history = trace_payload.get("distribution_history")
    bn_payload = trace_payload.get("bn")
    if not isinstance(history, list) or not history:
        return None
    if not isinstance(history[0], list):
        return None
    if not isinstance(bn_payload, dict):
        return None
    initial_distribution = history[0]
    node_names = extract_belief_node_names_from_bn(bn_payload, initial_distribution)
    if not node_names:
        return None

    probs: dict[str, float] = {node: 0.0 for node in node_names}
    for entry in initial_distribution:
        if not isinstance(entry, dict):
            return None
        state = entry.get("state")
        if not isinstance(state, dict):
            return None
        try:
            prob = float(entry.get("probability"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(prob):
            return None
        for node in node_names:
            if state.get(node) is True:
                probs[node] += prob
    return probs


def load_human_rounds(results_root: Path, min_persuader_turns: int) -> list[HumanRound]:
    """Load human BN-survey rounds with per-node initial beliefs."""
    candidates: list[HumanRound] = []
    seen: set[tuple[Any, ...]] = set()
    for jsonl_path in sorted(results_root.rglob("*.jsonl")):
        for line_idx, record in parse_jsonl_dict_records(jsonl_path):
            condition = record.get("condition")
            if not isinstance(condition, dict):
                continue
            roles = condition.get("roles")
            if not isinstance(roles, dict):
                continue
            if roles.get("human_target") is not True:
                continue
            if roles.get("simulated_target") is not None:
                continue

            proposition = record.get("proposition")
            if not isinstance(proposition, str) or not proposition.strip():
                continue
            try:
                initial = float(record.get("target_initial_belief"))
                final = float(record.get("target_final_belief"))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(initial) and math.isfinite(final)):
                continue
            if not (0.0 <= initial <= 1.0 and 0.0 <= final <= 1.0):
                continue

            node_beliefs = _to_float_map(record.get("target_initial_node_beliefs"))
            if not node_beliefs:
                continue
            node_keys = sorted(
                [key for key in node_beliefs if key.startswith("Belief_")]
            )
            if not node_keys:
                continue
            filtered_node_beliefs = {key: node_beliefs[key] for key in node_keys}
            if any(
                not (0.0 <= value <= 1.0) for value in filtered_node_beliefs.values()
            ):
                continue

            stance_raw = record.get("persuader_supports_proposition")
            if isinstance(stance_raw, bool):
                supports = stance_raw
            else:
                supports = infer_stance(initial)

            turns = count_persuader_turns(record.get("messages"))
            if turns < min_persuader_turns:
                continue

            node_bins = {
                key: belief_bin(value) for key, value in filtered_node_beliefs.items()
            }
            raw = final - initial
            pr = persuader_relative_delta(
                initial=initial,
                final=final,
                supports_proposition=supports,
            )
            timed_out = bool(record.get("timed_out", False))

            dedupe_key = (
                proposition.strip(),
                round(initial, 6),
                round(final, 6),
                supports,
                tuple(sorted(node_bins.items())),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            candidates.append(
                HumanRound(
                    source_file=str(jsonl_path),
                    source_line=line_idx,
                    proposition=proposition.strip(),
                    persuader_supports_proposition=supports,
                    target_initial_belief=initial,
                    target_final_belief=final,
                    target_initial_bin=belief_bin(initial),
                    node_initial_beliefs=filtered_node_beliefs,
                    node_initial_bins=node_bins,
                    raw_delta=raw,
                    persuader_relative_delta=pr,
                    timed_out=timed_out,
                    persuader_turns=turns,
                )
            )
    return candidates


def _pick_snapshot_trace(
    step_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Pick one step snapshot that includes round and trace payload."""
    for round_payload, trace_payload in iter_round_trace_snapshots(
        step_rows, reverse_steps=True
    ):
        return round_payload, trace_payload
    return None


def load_sim_rounds(
    episodes_path: Path,
    steps_path: Path,
    min_persuader_turns: int,
    sim_corpus_prefix: str = "",
    required_persuader_turn_mode: str = "",
    required_initialization_mode: str = "",
) -> list[SimRound]:
    """Load simulator rounds from episodes/steps artifacts with node initial beliefs.

    Args:
        episodes_path: Episodes JSONL path.
        steps_path: Steps JSONL path.
        min_persuader_turns: Minimum persuader turns required.
        sim_corpus_prefix: Optional corpus prefix filter.
        required_persuader_turn_mode: Optional required persuader turn mode.
        required_initialization_mode: Optional required initialization mode.

    Returns:
        Simulator rounds that satisfy the replay filters.
    """
    episode_rows = parse_jsonl_dict_records(episodes_path)
    step_rows = parse_jsonl_dict_records(steps_path)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in step_rows:
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str) and episode_id:
            by_episode[episode_id].append(row)

    out: list[SimRound] = []
    for _, episode in episode_rows:
        episode_id = episode.get("episode_id")
        proposition = episode.get("proposition_id")
        target_backend = episode.get("target_backend")
        if not isinstance(episode_id, str) or not episode_id:
            continue
        if not isinstance(proposition, str) or not proposition.strip():
            continue
        corpus_raw = episode.get("corpus")
        corpus = corpus_raw if isinstance(corpus_raw, str) else ""
        if sim_corpus_prefix and not corpus.startswith(sim_corpus_prefix):
            continue
        turn_mode_raw = episode.get("persuader_turn_mode")
        turn_mode = turn_mode_raw if isinstance(turn_mode_raw, str) else ""
        if required_persuader_turn_mode and turn_mode != required_persuader_turn_mode:
            continue
        init_mode_raw = episode.get("initialization_mode")
        init_mode = init_mode_raw if isinstance(init_mode_raw, str) else ""
        if required_initialization_mode and init_mode != required_initialization_mode:
            continue
        if target_backend not in {"simulated_target", "llm_target"}:
            continue
        try:
            initial = float(episode.get("target_initial_belief"))
            final = float(episode.get("target_final_belief"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(initial) and math.isfinite(final)):
            continue
        if not (0.0 <= initial <= 1.0 and 0.0 <= final <= 1.0):
            continue

        steps = by_episode.get(episode_id, [])
        if not steps:
            continue
        snapshot_pair = _pick_snapshot_trace(steps)
        if snapshot_pair is None:
            continue
        round_payload, trace_payload = snapshot_pair

        stance_raw = round_payload.get("persuader_supports_proposition")
        if isinstance(stance_raw, bool):
            supports = stance_raw
        else:
            supports = infer_stance(initial)

        node_map = _to_float_map(round_payload.get("target_initial_node_beliefs"))
        if not node_map:
            node_map = _node_probs_from_distribution_history(trace_payload)
        if not node_map:
            continue
        node_keys = sorted([key for key in node_map if key.startswith("Belief_")])
        if not node_keys:
            continue
        filtered_node_map = {key: node_map[key] for key in node_keys}
        if any(not (0.0 <= value <= 1.0) for value in filtered_node_map.values()):
            continue

        turns = count_persuader_turns(round_payload.get("messages"))
        if turns < min_persuader_turns:
            continue

        node_bins = {key: belief_bin(value) for key, value in filtered_node_map.items()}
        raw = final - initial
        pr = persuader_relative_delta(
            initial=initial,
            final=final,
            supports_proposition=supports,
        )
        out.append(
            SimRound(
                episode_id=episode_id,
                source_file=str(episodes_path),
                proposition=proposition.strip(),
                persuader_supports_proposition=supports,
                target_initial_belief=initial,
                target_final_belief=final,
                target_initial_bin=belief_bin(initial),
                node_initial_beliefs=filtered_node_map,
                node_initial_bins=node_bins,
                raw_delta=raw,
                persuader_relative_delta=pr,
                persona=(
                    str(episode.get("persona"))
                    if isinstance(episode.get("persona"), str)
                    else None
                ),
                simulated_target_no_rhetoric=bool(
                    episode.get("simulated_target_no_rhetoric", False)
                ),
                policy_model=(
                    str(episode.get("policy_model"))
                    if isinstance(episode.get("policy_model"), str)
                    else None
                ),
                persuader_turns=turns,
            )
        )
    return out


def build_matches(
    humans: list[HumanRound],
    sims: list[SimRound],
    *,
    require_node_bin_match: bool,
    min_abs_human_pr_delta: float,
    min_abs_sim_pr_delta: float,
    skip_timed_out_humans: bool,
) -> list[MatchCandidate]:
    """Build candidate matches under proposition/bin/stance constraints."""
    matches: list[MatchCandidate] = []
    sims_by_prop: dict[str, list[SimRound]] = defaultdict(list)
    for sim in sims:
        sims_by_prop[sim.proposition].append(sim)

    for human in humans:
        if skip_timed_out_humans and human.timed_out:
            continue
        if abs(human.persuader_relative_delta) < min_abs_human_pr_delta:
            continue
        prop_sims = sims_by_prop.get(human.proposition, [])
        for sim in prop_sims:
            if (
                sim.persuader_supports_proposition
                != human.persuader_supports_proposition
            ):
                continue
            if sim.target_initial_bin != human.target_initial_bin:
                continue
            if abs(sim.persuader_relative_delta) < min_abs_sim_pr_delta:
                continue
            if require_node_bin_match:
                if sim.node_initial_bins.keys() != human.node_initial_bins.keys():
                    continue
                if any(
                    sim.node_initial_bins[key] != human.node_initial_bins[key]
                    for key in human.node_initial_bins.keys()
                ):
                    continue

            movement_score = min(
                abs(human.persuader_relative_delta),
                abs(sim.persuader_relative_delta),
            )
            matches.append(
                MatchCandidate(
                    proposition=human.proposition,
                    human_source=f"{human.source_file}:{human.source_line}",
                    sim_episode_id=sim.episode_id,
                    stance_supports_proposition=human.persuader_supports_proposition,
                    human_initial=human.target_initial_belief,
                    sim_initial=sim.target_initial_belief,
                    initial_abs_diff=abs(
                        human.target_initial_belief - sim.target_initial_belief
                    ),
                    target_bin=human.target_initial_bin,
                    node_bins=dict(human.node_initial_bins),
                    human_raw_delta=human.raw_delta,
                    sim_raw_delta=sim.raw_delta,
                    human_persuader_relative_delta=human.persuader_relative_delta,
                    sim_persuader_relative_delta=sim.persuader_relative_delta,
                    movement_score=movement_score,
                )
            )
    return matches


def parse_args() -> argparse.Namespace:
    """Parse CLI args for overlap search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Results root containing human round exports.",
    )
    parser.add_argument(
        "--sim-episodes",
        type=Path,
        default=Path("results/rl_baseline/sim_compare/episodes.jsonl"),
        help="Simulator episodes JSONL.",
    )
    parser.add_argument(
        "--sim-steps",
        type=Path,
        default=Path("results/rl_baseline/sim_compare/steps.jsonl"),
        help="Simulator steps JSONL.",
    )
    parser.add_argument(
        "--min-persuader-turns",
        type=int,
        default=3,
        help="Minimum persuader turns required in both human and simulator rounds.",
    )
    parser.add_argument(
        "--require-node-bin-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require per-node initial bin match across all BN beliefs.",
    )
    parser.add_argument(
        "--min-abs-human-pr-delta",
        type=float,
        default=0.0,
        help="Minimum abs(persuader-relative delta) for human rounds.",
    )
    parser.add_argument(
        "--min-abs-sim-pr-delta",
        type=float,
        default=0.0,
        help="Minimum abs(persuader-relative delta) for simulator rounds.",
    )
    parser.add_argument(
        "--skip-timed-out-humans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip human rounds flagged as timed out.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of matches to print.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full matches as JSON.",
    )
    parser.add_argument(
        "--sim-corpus-prefix",
        type=str,
        default="",
        help=(
            "Optional simulator corpus prefix filter, "
            "for example 'full_simulated_target__'."
        ),
    )
    parser.add_argument(
        "--required-persuader-turn-mode",
        type=str,
        default="",
        help=(
            "Optional required simulator persuader turn mode "
            "(for example 'human_first_then_policy')."
        ),
    )
    parser.add_argument(
        "--required-initialization-mode",
        type=str,
        default="",
        help=(
            "Optional required simulator initialization mode " "(for example 'exact')."
        ),
    )
    return parser.parse_args()


def _print_summary(
    *,
    humans: list[HumanRound],
    sims: list[SimRound],
    matches: list[MatchCandidate],
    top_k: int,
) -> None:
    """Print concise human-readable summary to stdout."""
    human_props = sorted({row.proposition for row in humans})
    sim_props = sorted({row.proposition for row in sims})
    overlap_props = sorted(set(human_props).intersection(sim_props))

    print(f"Human candidates: {len(humans)}")
    print(f"Simulator candidates: {len(sims)}")
    print(f"Human propositions with node beliefs: {len(human_props)}")
    print(f"Simulator propositions: {len(sim_props)}")
    print(f"Overlapping propositions: {len(overlap_props)}")
    for proposition in overlap_props:
        print(f"  - {proposition}")
    print("")
    print(f"Match candidates: {len(matches)}")
    if not matches:
        return

    by_init = sorted(
        matches, key=lambda row: (row.initial_abs_diff, -row.movement_score)
    )
    by_movement = sorted(
        matches,
        key=lambda row: (-row.movement_score, row.initial_abs_diff),
    )
    print("Best by initial-belief closeness:")
    best_init = by_init[0]
    print(json.dumps(asdict(best_init), indent=2))
    print("")
    print("Best by movement score:")
    best_move = by_movement[0]
    print(json.dumps(asdict(best_move), indent=2))
    print("")
    print(f"Top {min(top_k, len(by_init))} matches by initial-belief closeness:")
    for index, match in enumerate(by_init[:top_k], start=1):
        print(
            f"{index:02d}. "
            f"init_diff={match.initial_abs_diff:.4f} "
            f"movement={match.movement_score:.4f} "
            f"human_pr_delta={match.human_persuader_relative_delta:+.4f} "
            f"sim_pr_delta={match.sim_persuader_relative_delta:+.4f} "
            f"sim_episode={match.sim_episode_id}"
        )


def main() -> None:
    """Run overlap search and print ranked results."""
    args = parse_args()
    humans = load_human_rounds(
        results_root=args.results_root,
        min_persuader_turns=args.min_persuader_turns,
    )
    sims = load_sim_rounds(
        episodes_path=args.sim_episodes,
        steps_path=args.sim_steps,
        min_persuader_turns=args.min_persuader_turns,
        sim_corpus_prefix=str(args.sim_corpus_prefix or ""),
        required_persuader_turn_mode=str(args.required_persuader_turn_mode or ""),
        required_initialization_mode=str(args.required_initialization_mode or ""),
    )
    matches = build_matches(
        humans,
        sims,
        require_node_bin_match=bool(args.require_node_bin_match),
        min_abs_human_pr_delta=float(args.min_abs_human_pr_delta),
        min_abs_sim_pr_delta=float(args.min_abs_sim_pr_delta),
        skip_timed_out_humans=bool(args.skip_timed_out_humans),
    )
    _print_summary(humans=humans, sims=sims, matches=matches, top_k=args.top_k)

    if args.output_json is not None:
        by_init = sorted(
            matches, key=lambda row: (row.initial_abs_diff, -row.movement_score)
        )
        payload = [asdict(match) for match in by_init]
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print("")
        print(f"Wrote {len(payload)} matches to {args.output_json}")


if __name__ == "__main__":
    main()
