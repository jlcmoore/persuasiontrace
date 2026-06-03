"""Export RL rollout JSONL artifacts into round-style condition JSONL outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiment.condition import Condition, ContinuousMeasure, Roles
from experiment.round import Round
from simulation.io import read_jsonl_records
from simulation.target import (
    TargetPersona,
    node_beliefs_from_trace_payload,
    susceptibilities_for_persona,
)


def _float_or_none(value: Any) -> float | None:
    """
    Convert a value to float when possible.

    Args:
        value: Raw input value.

    Returns:
        Parsed float, or None when conversion fails.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    """
    Convert a value to int when possible.

    Args:
        value: Raw input value.

    Returns:
        Parsed integer, or None when conversion fails.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_from_env_snapshot(snapshot_json: str) -> dict[str, Any] | None:
    """
    Extract compact simulated-target trace fields from one env snapshot JSON.

    Args:
        snapshot_json: Serialized environment snapshot payload.

    Returns:
        Trace dictionary with atom/belief fields, or None when unavailable.
    """
    try:
        payload = json.loads(snapshot_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        return None

    trace: dict[str, Any] = {}
    atom_history = target_payload.get("atom_history")
    if isinstance(atom_history, list):
        trace["atom_history"] = atom_history
    belief_history = target_payload.get("belief_history")
    if isinstance(belief_history, list):
        trace["belief_history"] = belief_history

    susceptibilities = target_payload.get("susceptibilities")
    if isinstance(susceptibilities, dict):
        trace["susceptibilities"] = susceptibilities

    use_rhetorical = target_payload.get("use_rhetorical_dimensions")
    if isinstance(use_rhetorical, bool):
        trace["use_rhetorical_dimensions"] = use_rhetorical

    effect_scale = _float_or_none(target_payload.get("belief_update_scale"))
    if effect_scale is not None:
        trace["belief_update_scale"] = float(effect_scale)

    verbalize_beliefs = target_payload.get("verbalize_beliefs")
    if isinstance(verbalize_beliefs, bool):
        trace["verbalize_beliefs"] = verbalize_beliefs

    bn_payload = target_payload.get("bn")
    if isinstance(bn_payload, dict):
        trace["bn"] = bn_payload
    distribution_history = target_payload.get("distribution_history")
    if isinstance(distribution_history, list):
        trace["distribution_history"] = distribution_history

    return trace or None


def _node_beliefs_from_episode_row(
    episode_row: dict[str, Any],
    *,
    initial: bool,
) -> dict[str, float] | None:
    """
    Read node-level beliefs directly from one episode row payload.

    Args:
        episode_row: Episode row dictionary.
        initial: Whether to read initial or final node beliefs.

    Returns:
        Mapping from node id to belief in [0,1], or None when unavailable/invalid.
    """
    key = "target_initial_node_beliefs" if initial else "target_final_node_beliefs"
    payload = episode_row.get(key)
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, float] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str) or not raw_key.startswith("Belief_"):
            return None
        try:
            belief_value = float(raw_value)
        except (TypeError, ValueError):
            return None
        if not 0 <= belief_value <= 1:
            return None
        parsed[raw_key] = belief_value
    if not parsed:
        return None
    return parsed


def _build_condition_from_episode_row(episode_row: dict[str, Any]) -> Condition:
    """
    Reconstruct a Condition from one rollout episode row.

    Args:
        episode_row: Serialized episode row.

    Returns:
        Reconstructed Condition.
    """
    target_backend = str(episode_row.get("target_backend") or "simulated_target")
    policy_model = str(episode_row.get("policy_model") or "")
    simulated_target_model = str(episode_row.get("simulated_target_model") or "")
    llm_target_model = str(
        episode_row.get("llm_target_model") or simulated_target_model or ""
    )
    persona = str(episode_row.get("persona") or "")

    if target_backend == "llm_target":
        roles = Roles(
            llm_persuader=policy_model,
            llm_target=llm_target_model or None,
        )
    else:
        roles = Roles(
            llm_persuader=policy_model,
            simulated_target=simulated_target_model or None,
            simulated_target_persona=persona or None,
        )
    has_llm_target = bool(roles.llm_target)
    has_simulated_target = bool(roles.simulated_target)

    factual_domain = bool(episode_row.get("factual_domain", False))
    proposition_is_correct = episode_row.get("proposition_is_correct")
    if factual_domain and not isinstance(proposition_is_correct, bool):
        factual_domain = False
        proposition_is_correct = None

    continuous_measure_raw = episode_row.get("continuous_measure")
    if isinstance(continuous_measure_raw, str) and continuous_measure_raw:
        continuous_measure = continuous_measure_raw
    else:
        continuous_measure = ContinuousMeasure.SERIAL_QUESTIONS

    llm_target_use_bayes_structure = (
        bool(episode_row.get("llm_target_use_bayes_structure", False))
        if has_llm_target
        else False
    )
    enable_node_belief_survey = bool(
        episode_row.get("enable_node_belief_survey", False)
    )
    simulated_target_no_rhetoric = (
        bool(episode_row.get("simulated_target_no_rhetoric", False))
        if has_simulated_target
        else False
    )
    simulated_target_effect_scale = (
        _float_or_none(episode_row.get("simulated_target_effect_scale"))
        if has_simulated_target
        else None
    )
    simulated_target_verbalize_beliefs = (
        bool(episode_row.get("simulated_target_verbalize_beliefs", False))
        if has_simulated_target
        else False
    )
    participant_proposition_raw = episode_row.get("participant_proposition")
    participant_proposition = (
        bool(participant_proposition_raw)
        if isinstance(participant_proposition_raw, bool)
        else False
    )

    return Condition(
        roles=roles,
        factual_domain=factual_domain,
        proposition_is_correct=proposition_is_correct,
        continuous_measure=continuous_measure,
        enable_node_belief_survey=enable_node_belief_survey,
        use_audio=False,
        show_transcript=False,
        control_dialogue=False,
        participant_proposition=participant_proposition,
        proposition_source=(
            str(episode_row.get("source") or "")
            if has_simulated_target
            or (has_llm_target and llm_target_use_bayes_structure)
            or enable_node_belief_survey
            else None
        ),
        turn_limit=_int_or_none(episode_row.get("turn_limit")),
        no_early_end=True,
        max_message_chars=300,
        max_audio_seconds=30,
        llm_target_use_bayes_structure=llm_target_use_bayes_structure,
        simulated_target_no_rhetoric=simulated_target_no_rhetoric,
        simulated_target_effect_scale=(
            float(simulated_target_effect_scale)
            if simulated_target_effect_scale is not None
            else (1.0 if has_simulated_target else None)
        ),
        simulated_target_verbalize_beliefs=simulated_target_verbalize_beliefs,
    )


def _build_round_from_episode_and_steps(
    episode_row: dict[str, Any],
    steps: list[dict[str, Any]],
) -> Round:
    """
    Reconstruct one Round from an episode row and its aligned step rows.

    Args:
        episode_row: Serialized episode row.
        steps: Step rows for the same episode.

    Returns:
        Reconstructed round model.
    """
    condition = _build_condition_from_episode_row(episode_row)

    sorted_steps = sorted(steps, key=lambda row: int(row.get("step_index", 0)))
    messages: list[dict[str, str]] = []
    chains_of_thought: list[dict[str, str | None]] = []
    reasoning_traces: list[dict[str, str | None]] = []
    serial_questions: list[float] = []
    for step in sorted_steps:
        persuader_text = str(step.get("persuader_text") or "")
        target_text = str(step.get("target_text") or "")
        messages.append({"role": "persuader", "content": persuader_text})
        messages.append({"role": "target", "content": target_text})
        chains_of_thought.append(
            {"role": "persuader", "content": step.get("persuader_thought")}
        )
        chains_of_thought.append({"role": "target", "content": None})
        reasoning_traces.append(
            {
                "role": "persuader",
                "content": step.get("persuader_reasoning_trace"),
            }
        )
        reasoning_traces.append({"role": "target", "content": None})
        belief_after = _float_or_none(step.get("belief_after"))
        if belief_after is not None:
            serial_questions.append(float(belief_after))

    initial_belief = _float_or_none(episode_row.get("target_initial_belief"))
    final_belief = _float_or_none(episode_row.get("target_final_belief"))

    supports_raw = episode_row.get("persuader_supports_proposition")
    if isinstance(supports_raw, bool):
        persuader_supports_proposition: bool | None = supports_raw
    elif initial_belief is not None:
        persuader_supports_proposition = bool(initial_belief < 0.5)
    else:
        persuader_supports_proposition = None

    policy_instruction_enabled_raw = episode_row.get("extra_policy_instruction_enabled")
    policy_instruction_enabled: bool | None = (
        bool(policy_instruction_enabled_raw)
        if isinstance(policy_instruction_enabled_raw, bool)
        else None
    )
    policy_instruction_hash_raw = episode_row.get("extra_policy_instruction_hash")
    policy_instruction_hash: str | None = (
        str(policy_instruction_hash_raw).strip()
        if isinstance(policy_instruction_hash_raw, str)
        and str(policy_instruction_hash_raw).strip()
        else None
    )

    simulated_target_trace: dict[str, Any] | None = None
    if condition.roles.simulated_target:
        persona = str(episode_row.get("persona") or "")
        no_rhetoric = bool(episode_row.get("simulated_target_no_rhetoric", False))
        effect_scale = _float_or_none(episode_row.get("simulated_target_effect_scale"))
        if effect_scale is None:
            effect_scale = 1.0
        verbalize_beliefs = bool(
            episode_row.get("simulated_target_verbalize_beliefs", False)
        )
        if persona:
            try:
                persona_enum = TargetPersona(persona)
                if no_rhetoric:
                    simulated_target_trace = {"use_rhetorical_dimensions": False}
                else:
                    simulated_target_trace = {
                        "susceptibilities": susceptibilities_for_persona(persona_enum)
                    }
                if effect_scale is not None:
                    simulated_target_trace["belief_update_scale"] = float(effect_scale)
                simulated_target_trace["verbalize_beliefs"] = verbalize_beliefs
            except ValueError:
                simulated_target_trace = None

        for step in reversed(sorted_steps):
            snapshot_values: list[str] = []
            post_snapshot_raw = step.get("post_step_env_state_json")
            if isinstance(post_snapshot_raw, str) and post_snapshot_raw.strip():
                snapshot_values.append(post_snapshot_raw)
            pre_snapshot_raw = step.get("pre_step_env_state_json")
            if isinstance(pre_snapshot_raw, str) and pre_snapshot_raw.strip():
                snapshot_values.append(pre_snapshot_raw)
            snapshot_trace: dict[str, Any] | None = None
            for snapshot_raw in snapshot_values:
                snapshot_trace = _trace_from_env_snapshot(snapshot_raw)
                if snapshot_trace is not None:
                    break
            if snapshot_trace is None:
                continue
            if simulated_target_trace is None:
                simulated_target_trace = dict(snapshot_trace)
            else:
                simulated_target_trace.update(snapshot_trace)
            break

    if policy_instruction_enabled is not None or policy_instruction_hash is not None:
        if simulated_target_trace is None:
            simulated_target_trace = {}
        if policy_instruction_enabled is not None:
            simulated_target_trace["extra_policy_instruction_enabled"] = (
                policy_instruction_enabled
            )
        if policy_instruction_hash is not None:
            simulated_target_trace["extra_policy_instruction_hash"] = (
                policy_instruction_hash
            )

    initial_node_beliefs = node_beliefs_from_trace_payload(
        simulated_target_trace,
        initial=True,
    )
    if initial_node_beliefs is None:
        initial_node_beliefs = _node_beliefs_from_episode_row(
            episode_row,
            initial=True,
        )
    final_node_beliefs = node_beliefs_from_trace_payload(
        simulated_target_trace,
        initial=False,
    )
    if final_node_beliefs is None:
        final_node_beliefs = _node_beliefs_from_episode_row(
            episode_row,
            initial=False,
        )

    return Round(
        condition=condition,
        proposition=str(episode_row.get("proposition_id") or ""),
        proposition_during_round=(
            str(episode_row.get("proposition_during_round"))
            if episode_row.get("proposition_during_round")
            else None
        ),
        target_initial_belief=initial_belief,
        target_final_belief=final_belief,
        target_initial_node_beliefs=initial_node_beliefs,
        target_final_node_beliefs=final_node_beliefs,
        persuader_supports_proposition=persuader_supports_proposition,
        messages=messages,
        chains_of_thought=chains_of_thought,
        reasoning_traces=reasoning_traces,
        serial_questions=serial_questions or None,
        serial_questions_sentence=None,
        message_highlights=None,
        mouse_traces=None,
        simulated_target_trace=simulated_target_trace,
    )


def export_rollouts_to_rounds(
    *,
    episodes_jsonl: Path,
    steps_jsonl: Path,
    results_root: Path,
    output_date: str,
    append: bool = False,
) -> dict[str, Any]:
    """
    Export rollout artifacts to condition-folder round JSONL files.

    Args:
        episodes_jsonl: Path to baseline-runner episodes JSONL.
        steps_jsonl: Path to baseline-runner steps JSONL.
        results_root: Root output directory that will contain condition folders.
        output_date: Date tag for output files (YYYY-MM-DD).
        append: Whether to append instead of overwrite per condition file.

    Returns:
        Summary dictionary containing counts and output paths.
    """
    episode_rows = read_jsonl_records(episodes_jsonl)
    step_rows = read_jsonl_records(steps_jsonl)

    steps_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step_row in step_rows:
        episode_id = step_row.get("episode_id")
        if isinstance(episode_id, str) and episode_id:
            steps_by_episode[episode_id].append(step_row)

    grouped_rounds: dict[str, list[Round]] = defaultdict(list)
    skipped_missing_steps = 0
    for episode_row in episode_rows:
        episode_id = episode_row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            continue
        step_group = steps_by_episode.get(episode_id, [])
        if not step_group:
            skipped_missing_steps += 1
            continue

        round_obj = _build_round_from_episode_and_steps(episode_row, step_group)
        condition_dir = round_obj.get_condition().to_dir()
        grouped_rounds[condition_dir].append(round_obj)

    results_root.mkdir(parents=True, exist_ok=True)
    written_files: list[str] = []
    total_rounds = 0
    for condition_dir, rounds in grouped_rounds.items():
        output_dir = results_root / condition_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_date}.jsonl"
        mode = "a" if append else "w"
        with output_path.open(mode, encoding="utf-8") as handle:
            for round_obj in rounds:
                payload = [round_obj.model_dump(mode="json")]
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                total_rounds += 1
        written_files.append(str(output_path))

    return {
        "episodes_total": len(episode_rows),
        "steps_total": len(step_rows),
        "rounds_written": total_rounds,
        "conditions_written": len(grouped_rounds),
        "episodes_skipped_missing_steps": skipped_missing_steps,
        "written_files": written_files,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for rollout export."""
    parser = argparse.ArgumentParser(
        description="Export RL rollout artifacts to round-style condition JSONL files."
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        required=True,
        help="Path to episodes.jsonl emitted by rl.baseline_runner.",
    )
    parser.add_argument(
        "--steps-jsonl",
        type=Path,
        required=True,
        help="Path to steps.jsonl emitted by rl.baseline_runner.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "rl_exported_rounds",
        help="Destination root directory for condition folders.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=dt.date.today().isoformat(),
        help="Output filename date tag (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing date-stamped files instead of overwriting.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for rollout export."""
    args = _parse_args()
    result = export_rollouts_to_rounds(
        episodes_jsonl=args.episodes_jsonl,
        steps_jsonl=args.steps_jsonl,
        results_root=args.results_root,
        output_date=args.date,
        append=bool(args.append),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
