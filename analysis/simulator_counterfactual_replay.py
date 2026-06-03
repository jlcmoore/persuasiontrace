"""Replay human rounds against simulator backends from matched initial beliefs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import litellm
from tqdm import tqdm

from experiment.condition import ContinuousMeasure, Roles
from experiment.llm_batch import (
    _reasoning_effort_override_for_model,
    batch_chat,
    extract_text_from_response,
)
from experiment.llm_utils import atomizer_temperature_for_model, disable_litellm_logging
from experiment.persuader_policies import (
    is_naive_persuader_model,
    naive_persuader_action_for_round,
)
from experiment.round import Round
from rl.baseline_runner import _policy_action_from_text, _policy_messages_for_round
from rl.plan_io import write_plan_and_config_snapshot_to_paths
from rl.rollout_rows import build_episode_row, build_step_row
from rl.sim_target_env import (
    LLMTargetTurnOutput,
    SimTargetEnv,
    SimTargetEnvConfig,
)
from simulation.counterfactual_replay.aggregation import (
    build_summary_rows,
    load_round_error_rows,
    summary_table_rows,
    write_csv_rows,
)
from simulation.counterfactual_replay.exec import dry_run_rows
from simulation.counterfactual_replay.math import (
    ipf_match_marginals,
    marginal_true_probability,
    mean_abs_error,
    node_delta_mae,
    node_mae,
)
from simulation.human_likeness import (
    RoundTrajectory,
    load_serial_trajectories,
    parse_min_date,
)
from simulation.target import BayesianNetwork, MessageAnalysis, TargetPersona
from simulation.target_bins import TARGET_BELIEF_BIN_RANGES

from .simulator_common import (
    add_common_human_simulator_filter_args,
    select_human_rows,
    selector_kwargs_from_args,
)
from .tables import print_table

disable_litellm_logging()

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_PREFIX = Path("analysis/data/simulator_counterfactual_replay")
PERSUADER_TURN_MODES = (
    "human_replay",
    "human_first_then_policy",
    "policy",
)


@dataclass(frozen=True)
class ReplayCorpusSpec:
    """Configuration for one replay corpus variant."""

    corpus: str
    target_backend: Literal["simulated_target", "llm_target"]
    target_model: str
    persona: TargetPersona
    llm_target_use_bayes_structure: bool
    simulated_target_no_rhetoric: bool


@dataclass(frozen=True)
class IPFConfig:
    """Configuration for simulated-target initial-state fitting."""

    max_iter: int
    tol: float


@dataclass(frozen=True)
class ReplayRoundContext:
    """Inputs required to construct one round-level replay error row."""

    spec: ReplayCorpusSpec
    source_row: RoundTrajectory
    human_round: Round
    replay_round: Round
    replay_initial_nodes: dict[str, float] | None
    replay_final_nodes: dict[str, float] | None
    n_persuader_turns: int
    replay_index: int
    initialization_mode: str
    same_bin_scheme: str
    persuader_turn_mode: str


@dataclass(frozen=True)
class ReplayJob:
    """A single replay execution unit."""

    spec: ReplayCorpusSpec
    row: RoundTrajectory
    replay_index: int
    persuader_messages: tuple[str, ...]
    policy_model: str | None
    persuader_turn_mode: str


@dataclass
class ReplayRolloutBuffer:
    """Mutable replay rollout buffers persisted as rollout-style artifacts."""

    episode_id: str
    pair_id: str
    step_rows: list[dict[str, Any]]
    reward_sum: float = 0.0


@dataclass
class ReplayActiveState:
    """Mutable execution state for one pending replay job."""

    job: ReplayJob
    env: SimTargetEnv
    rollout: ReplayRolloutBuffer
    turn_index: int = 0
    initialization_mode: str = "exact"
    same_bin_scheme: str = "fixed3"


ReplayCacheKey = tuple[str, str, str, str, str, str, str, str]

FIXED3_BELIEF_BIN_RANGES: dict[str, tuple[float, float]] = {
    "low": (0.0, 0.35),
    "mid": (0.35, 0.65),
    "high": (0.65, 1.0),
}


def _parse_csv_list(raw: str) -> list[str]:
    """Parse a comma-separated CLI string into non-empty values."""
    values = [part.strip() for part in str(raw).split(",")]
    return [value for value in values if value]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for counterfactual replay."""
    parser = argparse.ArgumentParser(
        description=(
            "Run matched-initialization counterfactual evaluations against "
            "simulator targets with configurable persuader turn sourcing."
        )
    )
    add_common_human_simulator_filter_args(
        parser,
        include_results_dir=True,
        default_results_dir=DEFAULT_RESULTS_DIR,
        include_proposition_match=False,
    )
    parser.add_argument(
        "--target-models",
        type=str,
        default="openai/gpt-5.4-mini",
        help="Comma-separated target models to evaluate.",
    )
    parser.add_argument(
        "--simulated-target-personas",
        type=str,
        default="logical,emotional,authoritarian",
        help=(
            "Comma-separated personas for full simulated-target replay "
            "(logical, emotional, authoritarian, balanced, random)."
        ),
    )
    parser.add_argument(
        "--include-structure-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include llm_target with Bayes-structure context.",
    )
    parser.add_argument(
        "--include-vanilla-llm-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include llm_target without Bayes-structure context.",
    )
    parser.add_argument(
        "--include-full-simulated-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include full simulated target (rhetorical dimensions on).",
    )
    parser.add_argument(
        "--include-full-no-rhetoric-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include simulated target with rhetoric dimensions disabled.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional cap on replayable human rounds.",
    )
    parser.add_argument(
        "--replays-per-round",
        type=int,
        default=1,
        help=(
            "Number of replay rollouts per source human round and corpus. "
            "Use >1 for profile/divergence estimation."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used when sampling capped rounds.",
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("same-bin", "exact"),
        default="same-bin",
        help=(
            "How to initialize replay from human pre-survey beliefs. "
            "`same-bin` samples within each human belief bin (default); "
            "`exact` copies exact numeric beliefs."
        ),
    )
    parser.add_argument(
        "--same-bin-scheme",
        choices=("fixed5", "fixed3"),
        default="fixed3",
        help=(
            "Bin granularity used when --initialization-mode=same-bin. "
            "`fixed5`=very_low/low/mid/high/very_high; `fixed3`=low/mid/high."
        ),
    )
    parser.add_argument(
        "--persuader-turn-mode",
        choices=PERSUADER_TURN_MODES,
        default="policy",
        help=(
            "How to source persuader turns in replay. "
            "`policy` (default) generates all persuader turns from the policy model; "
            "`human_first_then_policy` reuses turn 1 then generates policy turns; "
            "`human_replay` reuses all human persuader turns; "
            "`policy` should be used for matched-initialization-only analyses."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan replay jobs and estimated model-call volume without executing calls.",
    )
    parser.add_argument(
        "--reuse-existing-round-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse matching rows from <output-prefix>_round_errors.csv and run only "
            "missing jobs. Disable with --no-reuse-existing-round-errors."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of replay jobs to schedule per batch.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent replay jobs inside each batch.",
    )
    parser.add_argument(
        "--sim-target-timeout-s",
        type=int,
        default=120,
        help="Timeout passed to target model calls.",
    )
    parser.add_argument(
        "--sim-target-max-retries",
        type=int,
        default=2,
        help="Retries passed to target model calls.",
    )
    parser.add_argument(
        "--ipf-max-iter",
        type=int,
        default=200,
        help="Maximum IPF iterations for simulated-target initialization.",
    )
    parser.add_argument(
        "--ipf-tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for simulated-target initialization.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix for replay artifacts.",
    )
    return parser.parse_args()


def _parse_persona(persona_name: str) -> TargetPersona:
    """Parse one persona name to a TargetPersona enum."""
    lowered = persona_name.strip().lower()
    for persona in TargetPersona:
        if persona.value == lowered:
            return persona
    raise ValueError(f"Unknown persona '{persona_name}'.")


def _build_corpus_specs(args: argparse.Namespace) -> list[ReplayCorpusSpec]:
    """Build replay corpus specs from CLI settings."""
    models = _parse_csv_list(args.target_models)
    personas = [
        _parse_persona(value)
        for value in _parse_csv_list(args.simulated_target_personas)
    ]

    specs: list[ReplayCorpusSpec] = []
    for model in models:
        if args.include_structure_target:
            specs.append(
                ReplayCorpusSpec(
                    corpus=f"structure_target__sim={model}",
                    target_backend="llm_target",
                    target_model=model,
                    persona=TargetPersona.LOGICAL,
                    llm_target_use_bayes_structure=True,
                    simulated_target_no_rhetoric=False,
                )
            )
        if args.include_vanilla_llm_target:
            specs.append(
                ReplayCorpusSpec(
                    corpus=f"vanilla_llm_target__sim={model}",
                    target_backend="llm_target",
                    target_model=model,
                    persona=TargetPersona.LOGICAL,
                    llm_target_use_bayes_structure=False,
                    simulated_target_no_rhetoric=False,
                )
            )
        for persona in personas:
            persona_suffix = persona.value
            if args.include_full_simulated_target:
                specs.append(
                    ReplayCorpusSpec(
                        corpus=(
                            f"full_simulated_target__sim={model}"
                            f"__persona={persona_suffix}"
                        ),
                        target_backend="simulated_target",
                        target_model=model,
                        persona=persona,
                        llm_target_use_bayes_structure=False,
                        simulated_target_no_rhetoric=False,
                    )
                )
        if args.include_full_no_rhetoric_target:
            specs.append(
                ReplayCorpusSpec(
                    corpus=f"full_no_rhetoric_target__sim={model}",
                    target_backend="simulated_target",
                    target_model=model,
                    persona=TargetPersona.LOGICAL,
                    llm_target_use_bayes_structure=False,
                    simulated_target_no_rhetoric=True,
                )
            )
    return specs


def _bin_ranges_for_scheme(same_bin_scheme: str) -> dict[str, tuple[float, float]]:
    """Return bin ranges for the requested same-bin scheme."""

    if same_bin_scheme == "fixed5":
        return TARGET_BELIEF_BIN_RANGES
    if same_bin_scheme == "fixed3":
        return FIXED3_BELIEF_BIN_RANGES
    raise ValueError(f"Unknown same-bin scheme: {same_bin_scheme!r}.")


def _belief_bin_from_value(value: float, *, same_bin_scheme: str) -> str:
    """Map one belief value to a same-bin scheme label."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Belief must be in [0,1], got {value!r}.")
    ranges = _bin_ranges_for_scheme(same_bin_scheme)
    for index, (bin_name, (low, high)) in enumerate(ranges.items()):
        is_last = index == len(ranges) - 1
        if is_last:
            if low <= value <= high:
                return str(bin_name)
            continue
        if low <= value < high:
            return str(bin_name)
    raise ValueError(f"Could not map belief value to bin: {value!r}.")


def _sample_value_from_bin(
    bin_name: str, rng: random.Random, *, same_bin_scheme: str
) -> float:
    """Sample one value inside a named belief bin."""

    ranges = _bin_ranges_for_scheme(same_bin_scheme)
    low, high = ranges[bin_name]
    if low == high:
        return float(low)
    sampled = float(rng.uniform(float(low), float(high)))
    if same_bin_scheme == "fixed5" and bin_name == "very_high":
        return max(0.0, min(1.0, sampled))
    epsilon = 1e-9
    return max(float(low), min(float(high) - epsilon, sampled))


def _seeded_replay_rng(seed: int, job: ReplayJob) -> random.Random:
    """Create a deterministic RNG keyed by replay-job identity."""

    source_round = (
        ""
        if job.row.source_round_index is None
        else str(int(job.row.source_round_index))
    )
    identity = "|".join(
        [
            str(int(seed)),
            str(job.spec.corpus),
            str(job.row.source_path),
            str(int(job.row.source_line_index)),
            source_round,
            str(int(job.replay_index)),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _resolve_initial_state_for_mode(
    *,
    mode: str,
    same_bin_scheme: str,
    seed: int,
    job: ReplayJob,
    initial_belief: float,
    initial_nodes: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Resolve initial belief state according to exact- vs same-bin mode."""

    if mode == "exact":
        return float(initial_belief), dict(initial_nodes)
    if mode != "same-bin":
        raise ValueError(f"Unknown initialization mode: {mode!r}.")

    rng = _seeded_replay_rng(seed, job)
    target_bin = _belief_bin_from_value(
        float(initial_belief),
        same_bin_scheme=same_bin_scheme,
    )
    sampled_target = _sample_value_from_bin(
        target_bin,
        rng,
        same_bin_scheme=same_bin_scheme,
    )
    sampled_nodes: dict[str, float] = {}
    for node_name in sorted(initial_nodes):
        node_bin = _belief_bin_from_value(
            float(initial_nodes[node_name]),
            same_bin_scheme=same_bin_scheme,
        )
        sampled_nodes[str(node_name)] = _sample_value_from_bin(
            node_bin,
            rng,
            same_bin_scheme=same_bin_scheme,
        )
    return sampled_target, sampled_nodes


def _extract_persuader_messages(round_obj: Round) -> list[str]:
    """Return ordered persuader-message text from a round."""
    messages: list[str] = []
    for msg in round_obj.messages:
        role = msg.get("role")
        content = msg.get("content")
        if role != "persuader":
            continue
        if not isinstance(content, str):
            continue
        messages.append(content)
    return messages


def _is_replayable_human_round(row: RoundTrajectory) -> bool:
    """Check whether one human row has the fields required for replay evaluation."""
    round_obj = row.round_obj
    return bool(
        isinstance(round_obj.bayesian_network, dict)
        and isinstance(round_obj.target_initial_belief, (int, float))
        and isinstance(round_obj.target_final_belief, (int, float))
        and isinstance(round_obj.target_initial_node_beliefs, dict)
        and isinstance(round_obj.target_final_node_beliefs, dict)
        and _extract_persuader_messages(round_obj)
    )


def _replay_persuader_role_kwargs(human_round: Round) -> dict[str, Any]:
    """Choose a persuader role for replay condition construction."""
    roles = human_round.get_condition().roles
    if roles.llm_persuader:
        return {"llm_persuader": str(roles.llm_persuader)}
    return {"human_persuader": True}


def _build_replay_condition(
    human_round: Round,
    spec: ReplayCorpusSpec,
    *,
    turn_limit: int,
) -> Any:
    """Build replay condition from a human round and a simulator spec."""
    base_condition = human_round.get_condition()
    persuader_kwargs = _replay_persuader_role_kwargs(human_round)

    if spec.target_backend == "llm_target":
        roles = Roles(llm_target=spec.target_model, **persuader_kwargs)
    else:
        roles = Roles(
            simulated_target=spec.target_model,
            simulated_target_persona=spec.persona.value,
            **persuader_kwargs,
        )

    return base_condition.model_copy(
        update={
            "roles": roles,
            "continuous_measure": ContinuousMeasure.SERIAL_QUESTIONS,
            "turn_limit": int(turn_limit),
            "minimum_turns": None,
            "no_early_end": True,
            "use_audio": False,
            "show_transcript": False,
            "control_dialogue": False,
            "enable_node_belief_survey": True,
            "llm_target_use_bayes_structure": (
                bool(spec.llm_target_use_bayes_structure)
                if spec.target_backend == "llm_target"
                else False
            ),
            "simulated_target_no_rhetoric": (
                bool(spec.simulated_target_no_rhetoric)
                if spec.target_backend == "simulated_target"
                else False
            ),
            "simulated_target_effect_scale": (
                1.0 if spec.target_backend == "simulated_target" else None
            ),
            "simulated_target_verbalize_beliefs": (
                False if spec.target_backend == "simulated_target" else False
            ),
        }
    )


def _safe_env_state_json(env: SimTargetEnv) -> str:
    """Serialize one environment snapshot, returning an empty string on failure."""
    try:
        return json.dumps(env.export_state(), ensure_ascii=True)
    except (AttributeError, RuntimeError, ValueError, OSError, TypeError):
        return ""


def _fallback_episode_id(job: ReplayJob) -> str:
    """Build a stable fallback episode id for one replay job."""
    source_round = (
        ""
        if job.row.source_round_index is None
        else str(int(job.row.source_round_index))
    )
    identity = "|".join(
        [
            str(job.spec.corpus),
            str(job.row.source_path),
            str(int(job.row.source_line_index)),
            source_round,
            str(int(job.replay_index)),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (
        f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-"
        f"{digest[16:20]}-{digest[20:32]}"
    )


def _episode_id_from_env_or_fallback(env: SimTargetEnv, job: ReplayJob) -> str:
    """Resolve replay episode id from env state, with deterministic fallback."""
    try:
        snapshot = env.export_state()
    except (AttributeError, RuntimeError, ValueError, OSError, TypeError):
        snapshot = None
    if isinstance(snapshot, dict):
        raw_episode_id = snapshot.get("episode_id")
        if isinstance(raw_episode_id, str) and raw_episode_id.strip():
            return raw_episode_id.strip()
    return _fallback_episode_id(job)


def _pair_id_from_job(job: ReplayJob) -> str:
    """Build a stable pair id derived from one replay job."""
    source_round = (
        ""
        if job.row.source_round_index is None
        else str(int(job.row.source_round_index))
    )
    identity = "|".join(
        [
            str(job.row.source_path),
            str(int(job.row.source_line_index)),
            source_round,
            str(job.row.proposition),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"pair_{digest[:16]}"


def _policy_model_label(human_round: Round) -> str:
    """Return a policy-model label for replay metadata rows."""
    roles = human_round.get_condition().roles
    if isinstance(roles.llm_persuader, str) and roles.llm_persuader.strip():
        return str(roles.llm_persuader)
    return "human_persuader"


def _policy_model_for_round(human_round: Round) -> str | None:
    """Return policy model for generated persuader turns, if available."""
    roles = human_round.get_condition().roles
    if isinstance(roles.llm_persuader, str) and roles.llm_persuader.strip():
        return str(roles.llm_persuader)
    return None


def _should_generate_policy_turn(*, mode: str, turn_index: int) -> bool:
    """Return whether one replay turn should be policy-generated."""
    if mode == "policy":
        return True
    if mode == "human_first_then_policy":
        return turn_index > 0
    if mode == "human_replay":
        return False
    raise ValueError(f"Unsupported persuader-turn mode: {mode!r}")


def _belief_bin_fixed5(value: float | None) -> str:
    """Map one belief value to fixed5 bin labels used by rollout metadata."""
    if value is None or not math.isfinite(value):
        return "unknown"
    clipped = max(0.0, min(1.0, float(value)))
    return _belief_bin_from_value(clipped, same_bin_scheme="fixed5")


def _replay_common_rollout_metadata(
    *,
    state: ReplayActiveState,
    proposition_during_round: str | None,
    continuous_measure: str,
    matched_initial_belief: float | None,
) -> dict[str, Any]:
    """Build shared rollout metadata for replay step/episode artifact rows.

    Args:
        state: Active replay state.
        proposition_during_round: Proposition text used during this round.
        continuous_measure: Continuous-measure mode label.
        matched_initial_belief: Initial target belief for matching metadata.

    Returns:
        Common metadata dictionary consumed by rollout row builder helpers.
    """
    source_round = state.job.row.round_obj
    source_condition = source_round.get_condition()
    return {
        "experiment_name": "simulator_counterfactual_replay",
        "episode_id": str(state.rollout.episode_id),
        "pair_id": str(state.rollout.pair_id),
        "split": "replay",
        "source": "counterfactual_replay",
        "proposition_id": str(state.job.row.proposition),
        "policy_model": _policy_model_label(source_round),
        "simulated_target_model": str(state.job.spec.target_model),
        "target_backend": str(state.job.spec.target_backend),
        "llm_target_use_bayes_structure": bool(
            state.job.spec.llm_target_use_bayes_structure
        ),
        "simulated_target_no_rhetoric": bool(
            state.job.spec.simulated_target_no_rhetoric
        ),
        "simulated_target_effect_scale": (
            1.0 if state.job.spec.target_backend == "simulated_target" else None
        ),
        "simulated_target_verbalize_beliefs": False,
        "persona": str(state.job.spec.persona.value),
        "persona_holdout": False,
        "reward_style": "dense",
        "turn_limit": int(len(state.job.persuader_messages)),
        "persuader_turn_mode": str(state.job.persuader_turn_mode),
        "init_belief_mode": str(state.initialization_mode),
        "init_belief_bin": _belief_bin_fixed5(matched_initial_belief),
        "extra_policy_instruction_enabled": False,
        "extra_policy_instruction_hash": "",
        "proposition_during_round": proposition_during_round,
        "factual_domain": bool(source_condition.factual_domain),
        "proposition_is_correct": source_condition.proposition_is_correct,
        "participant_proposition": bool(source_condition.participant_proposition),
        "continuous_measure": continuous_measure,
        "enable_node_belief_survey": True,
        "matched_initial_belief": matched_initial_belief,
    }


def _inject_human_initial_state(
    env: SimTargetEnv,
    human_round: Round,
    job: ReplayJob,
    *,
    initialization_mode: str,
    same_bin_scheme: str,
    seed: int,
    ipf_max_iter: int,
    ipf_tol: float,
) -> None:
    """Overwrite env initial state with one human round's pre-survey beliefs."""
    if not isinstance(human_round.target_initial_belief, (int, float)):
        raise ValueError("Human round is missing numeric target_initial_belief.")
    if not isinstance(human_round.target_initial_node_beliefs, dict):
        raise ValueError("Human round is missing target_initial_node_beliefs.")

    exact_initial_belief = float(human_round.target_initial_belief)
    exact_initial_nodes = {
        str(key): float(value)
        for key, value in human_round.target_initial_node_beliefs.items()
        if isinstance(value, (int, float))
    }
    initial_belief, initial_nodes = _resolve_initial_state_for_mode(
        mode=str(initialization_mode),
        same_bin_scheme=str(same_bin_scheme),
        seed=int(seed),
        job=job,
        initial_belief=exact_initial_belief,
        initial_nodes=exact_initial_nodes,
    )
    ipf_config = IPFConfig(max_iter=int(ipf_max_iter), tol=float(ipf_tol))

    snapshot = env.export_state()
    round_payload = snapshot.get("round")
    target_payload = snapshot.get("target")
    if not isinstance(round_payload, dict):
        raise ValueError("Env snapshot is missing round payload.")
    if not isinstance(target_payload, dict):
        raise ValueError("Env snapshot is missing target payload.")

    _reset_round_payload(
        round_payload,
        initial_belief=initial_belief,
        initial_nodes=initial_nodes,
        supports=bool(human_round.persuader_supports_proposition),
    )

    if job.spec.target_backend == "simulated_target":
        _inject_simulated_target_initial_state(
            target_payload,
            round_payload,
            initial_belief=initial_belief,
            initial_nodes=initial_nodes,
            ipf_config=ipf_config,
        )
    else:
        _inject_llm_target_initial_state(
            target_payload,
            round_payload,
            target_model=job.spec.target_model,
            initial_belief=initial_belief,
            initial_nodes=initial_nodes,
        )

    snapshot["round"] = round_payload
    snapshot["target"] = target_payload
    snapshot["steps"] = []
    snapshot["done"] = False
    snapshot["truncated"] = False
    env.load_state(snapshot)


def _reset_round_payload(
    round_payload: dict[str, Any],
    *,
    initial_belief: float,
    initial_nodes: dict[str, float],
    supports: bool,
) -> None:
    """Reset round payload so replay starts from a fresh state."""
    round_payload["target_initial_belief"] = float(initial_belief)
    round_payload["target_final_belief"] = None
    round_payload["target_initial_node_beliefs"] = dict(initial_nodes)
    round_payload["target_final_node_beliefs"] = None
    round_payload["persuader_supports_proposition"] = bool(supports)
    round_payload["messages"] = []
    round_payload["chains_of_thought"] = []
    round_payload["reasoning_traces"] = []
    round_payload["serial_questions"] = []
    round_payload["target_ended_round"] = None


def _inject_simulated_target_initial_state(
    target_payload: dict[str, Any],
    round_payload: dict[str, Any],
    *,
    initial_belief: float,
    initial_nodes: dict[str, float],
    ipf_config: IPFConfig,
) -> None:
    """Inject initial state for simulated-target replay."""
    distribution_history = target_payload.get("distribution_history")
    if (
        not isinstance(distribution_history, list)
        or not distribution_history
        or not isinstance(distribution_history[0], list)
    ):
        raise ValueError("Simulated-target snapshot is missing distribution history.")

    target_marginals = {"Target": initial_belief}
    target_marginals.update(initial_nodes)
    fitted = ipf_match_marginals(
        distribution_history[0],
        target_marginals,
        max_iter=int(ipf_config.max_iter),
        tol=float(ipf_config.tol),
    )
    distribution_history[0] = fitted
    target_payload["distribution_history"] = distribution_history

    marginal_target = marginal_true_probability(fitted, "Target")
    belief_history = target_payload.get("belief_history", [])
    if not isinstance(belief_history, list) or not belief_history:
        target_payload["belief_history"] = [marginal_target]
    else:
        belief_history[0] = marginal_target
        target_payload["belief_history"] = belief_history

    round_payload["target_initial_belief"] = float(marginal_target)
    round_payload["simulated_target_trace"] = dict(target_payload)


def _inject_llm_target_initial_state(
    target_payload: dict[str, Any],
    round_payload: dict[str, Any],
    *,
    target_model: str,
    initial_belief: float,
    initial_nodes: dict[str, float],
) -> None:
    """Inject initial state for llm-target replay."""
    target_payload["belief_history"] = [float(initial_belief)]
    target_payload["node_belief_history"] = [dict(initial_nodes)]
    round_payload["simulated_target_trace"] = {
        "llm_model": str(target_payload.get("llm_model") or target_model),
        "llm_target_belief_history": [float(initial_belief)],
        "llm_target_node_belief_history": [dict(initial_nodes)],
    }


def _extract_distribution_node_beliefs(
    target_obj: Any,
    *,
    index: int,
    node_ids: list[str],
) -> dict[str, float]:
    """Read node marginals from a SimulatedTarget distribution snapshot."""
    if not hasattr(target_obj, "distribution_history"):
        raise ValueError("Target object is missing distribution_history.")
    distribution_history = target_obj.distribution_history
    if not isinstance(distribution_history, list) or not distribution_history:
        raise ValueError("Target object has empty distribution_history.")
    distribution = distribution_history[index]
    if not isinstance(distribution, list):
        raise ValueError("Target distribution snapshot is invalid.")

    bn = BayesianNetwork(**target_obj.bn.model_dump(mode="json"))
    beliefs: dict[str, float] = {}
    for node_id in node_ids:
        beliefs[node_id] = float(bn.marginal_node_probability(node_id, distribution))
    return beliefs


def _load_replayable_rows(args: argparse.Namespace) -> list[RoundTrajectory]:
    """Load and filter human rows that are eligible for counterfactual replay."""
    min_date = parse_min_date(args.min_date)
    all_rows = load_serial_trajectories(args.results_dir, min_date=min_date)
    human_rows = select_human_rows(
        all_rows,
        human_source=args.human_source,
        selector_kwargs=selector_kwargs_from_args(args),
    )
    replayable_rows = [row for row in human_rows if _is_replayable_human_round(row)]
    if args.max_rounds is not None:
        if args.max_rounds <= 0:
            raise ValueError("--max-rounds must be positive when provided.")
        if len(replayable_rows) > args.max_rounds:
            rng = random.Random(int(args.seed))
            replayable_rows = rng.sample(replayable_rows, k=int(args.max_rounds))
    if not replayable_rows:
        raise ValueError(
            "No replayable human rounds found for the selected filters. "
            "Require bayesian_network, pre/post node beliefs, and persuader messages."
        )
    return replayable_rows


def _build_replay_jobs(
    specs: list[ReplayCorpusSpec],
    rows: list[RoundTrajectory],
    *,
    replays_per_round: int,
    persuader_turn_mode: str,
) -> list[ReplayJob]:
    """Construct replay jobs from corpora specs and human source rows."""
    jobs: list[ReplayJob] = []
    if replays_per_round <= 0:
        raise ValueError("--replays-per-round must be positive.")
    if persuader_turn_mode not in PERSUADER_TURN_MODES:
        raise ValueError(
            f"--persuader-turn-mode must be one of {PERSUADER_TURN_MODES}, "
            f"got {persuader_turn_mode!r}."
        )

    skipped_missing_policy_model = 0
    for spec in specs:
        for row in rows:
            persuader_messages = tuple(_extract_persuader_messages(row.round_obj))
            if not persuader_messages:
                continue
            policy_model = _policy_model_for_round(row.round_obj)
            needs_policy_model = persuader_turn_mode == "policy" or (
                persuader_turn_mode == "human_first_then_policy"
                and len(persuader_messages) > 1
            )
            if needs_policy_model and policy_model is None:
                skipped_missing_policy_model += 1
                continue
            for replay_index in range(int(replays_per_round)):
                jobs.append(
                    ReplayJob(
                        spec=spec,
                        row=row,
                        replay_index=int(replay_index),
                        persuader_messages=persuader_messages,
                        policy_model=policy_model,
                        persuader_turn_mode=str(persuader_turn_mode),
                    )
                )
    if skipped_missing_policy_model > 0:
        print(
            "Replay skip:",
            f"missing_policy_model_rows={int(skipped_missing_policy_model)}",
            f"persuader_turn_mode={persuader_turn_mode}",
        )
    return jobs


def _effective_same_bin_scheme(
    *, initialization_mode: str, same_bin_scheme: str
) -> str:
    """Return cache token for same-bin scheme given current init mode."""

    if initialization_mode == "same-bin":
        return str(same_bin_scheme)
    return "na"


def _job_cache_key(
    job: ReplayJob,
    *,
    initialization_mode: str,
    same_bin_scheme: str,
) -> ReplayCacheKey:
    """Build a stable cache key for one replay job.

    Args:
        job: Replay job with corpus and source-round identity.

    Returns:
        Tuple key used to match replay jobs against existing round-error rows.
    """
    source_round_index = (
        ""
        if job.row.source_round_index is None
        else str(int(job.row.source_round_index))
    )
    return (
        str(job.spec.corpus),
        str(job.row.source_path),
        str(int(job.row.source_line_index)),
        source_round_index,
        str(int(job.replay_index)),
        str(initialization_mode),
        _effective_same_bin_scheme(
            initialization_mode=initialization_mode,
            same_bin_scheme=same_bin_scheme,
        ),
        str(job.persuader_turn_mode),
    )


def _round_row_cache_key(row: dict[str, Any]) -> ReplayCacheKey | None:
    """Build the cache key for one existing round-error row.

    Args:
        row: Existing round-error CSV row.

    Returns:
        Tuple cache key when required identity columns are present; otherwise
        ``None``.
    """
    corpus = row.get("corpus")
    source_path = row.get("source_path")
    source_line_index = row.get("source_line_index")
    if corpus is None or source_path is None or source_line_index is None:
        return None

    line_index_text = str(source_line_index).strip()
    if not line_index_text:
        return None

    source_round_index = row.get("source_round_index")
    source_round_text = (
        "" if source_round_index is None else str(source_round_index).strip()
    )
    if source_round_text.lower() == "none":
        source_round_text = ""
    replay_index = row.get("replay_index")
    replay_index_text = "0" if replay_index is None else str(replay_index).strip()
    if replay_index_text.lower() in {"", "none", "nan"}:
        replay_index_text = "0"
    initialization_mode = row.get("initialization_mode")
    init_mode_text = (
        "exact" if initialization_mode is None else str(initialization_mode)
    )
    same_bin_scheme = row.get("same_bin_scheme")
    if same_bin_scheme is None:
        scheme_text = "fixed3" if init_mode_text == "same-bin" else "na"
    else:
        scheme_text = str(same_bin_scheme)
    persuader_turn_mode = row.get("persuader_turn_mode")
    turn_mode_text = (
        "human_replay"
        if persuader_turn_mode is None
        else str(persuader_turn_mode).strip() or "human_replay"
    )

    return (
        str(corpus),
        str(source_path),
        line_index_text,
        source_round_text,
        replay_index_text,
        init_mode_text,
        scheme_text,
        turn_mode_text,
    )


def _split_jobs_by_cache(
    jobs: list[ReplayJob],
    cached_rows: list[dict[str, Any]],
    *,
    initialization_mode: str,
    same_bin_scheme: str,
) -> tuple[list[dict[str, Any]], list[ReplayJob]]:
    """Split replay jobs into cache hits and jobs requiring execution.

    Args:
        jobs: Planned replay jobs for the current CLI filters.
        cached_rows: Existing round-error rows loaded from disk.

    Returns:
        A tuple of ``(reused_rows, pending_jobs)`` where reused rows are ordered
        by the current job order and pending jobs are the jobs still requiring
        model calls.
    """
    cached_by_key: dict[ReplayCacheKey, dict[str, Any]] = {}
    for row in cached_rows:
        key = _round_row_cache_key(row)
        if key is None:
            continue
        cached_by_key[key] = row

    reused_rows: list[dict[str, Any]] = []
    pending_jobs: list[ReplayJob] = []
    for job in jobs:
        key = _job_cache_key(
            job,
            initialization_mode=initialization_mode,
            same_bin_scheme=same_bin_scheme,
        )
        cached_row = cached_by_key.get(key)
        if cached_row is None:
            pending_jobs.append(job)
            continue
        reused_rows.append(cached_row)
    return reused_rows, pending_jobs


def _build_replay_env(
    args: argparse.Namespace,
    *,
    job: ReplayJob,
) -> SimTargetEnv:
    """Construct and initialize one replay environment."""
    spec = job.spec
    human_round = job.row.round_obj
    persuader_messages = list(job.persuader_messages)
    replay_condition = _build_replay_condition(
        human_round,
        spec,
        turn_limit=len(persuader_messages),
    )
    env_config = SimTargetEnvConfig(
        condition=replay_condition,
        proposition=human_round.proposition,
        proposition_during_round=human_round.proposition_during_round,
        bayesian_network=dict(human_round.bayesian_network or {}),
        simulated_target_model=spec.target_model,
        target_backend=spec.target_backend,
        llm_target_use_bayes_structure=bool(spec.llm_target_use_bayes_structure),
        simulated_target_no_rhetoric=bool(spec.simulated_target_no_rhetoric),
        simulated_target_effect_scale=1.0,
        simulated_target_verbalize_beliefs=False,
        simulated_target_persona=spec.persona,
        max_persuader_turns=len(persuader_messages),
        reward_style="dense",
        dense_reward_weight=1.0,
        terminal_reward_weight=1.0,
        init_belief_mode="prior",
        seed=int(args.seed),
        sim_target_timeout_s=int(args.sim_target_timeout_s),
        sim_target_max_retries=int(args.sim_target_max_retries),
    )
    env = SimTargetEnv(env_config)
    env.reset()
    _inject_human_initial_state(
        env,
        human_round,
        job,
        initialization_mode=str(args.initialization_mode),
        same_bin_scheme=str(args.same_bin_scheme),
        seed=int(args.seed),
        ipf_max_iter=int(args.ipf_max_iter),
        ipf_tol=float(args.ipf_tol),
    )
    return env


def _assert_batched_env_methods(env: SimTargetEnv, target_backend: str) -> None:
    """Validate that the environment exposes required batched-step methods."""
    if target_backend == "simulated_target":
        required = (
            "build_batched_step_input",
            "build_batched_response_messages",
            "complete_batched_step",
            "get_target",
        )
    elif target_backend == "llm_target":
        required = (
            "build_batched_step_input",
            "complete_batched_step",
            "parse_llm_target_turn_payload",
        )
    else:
        raise ValueError(f"Unsupported target backend: {target_backend}")
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise RuntimeError(
            "SimTargetEnv missing required batched methods: "
            f"backend={target_backend} missing={missing}"
        )


def _log_replay_skip(
    *,
    state: ReplayActiveState,
    stage: str,
    error: str,
) -> None:
    """Log one skipped replay job with compact context."""

    job = state.job
    print(
        "Replay skip:",
        stage,
        f"corpus={job.spec.corpus}",
        f"source_path={job.row.source_path}",
        f"source_line_index={int(job.row.source_line_index)}",
        f"source_round_index={job.row.source_round_index}",
        f"replay_index={int(job.replay_index)}",
        f"error={error}",
    )


def _call_batched_target(
    *,
    args: argparse.Namespace,
    model: str,
    messages: list[list[dict[str, str]]],
    response_format: Any | None = None,
    temperature: float | None = None,
) -> tuple[list[Any] | None, str | None]:
    """Call batched target completions with replay timeout/retry settings."""

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": args.sim_target_timeout_s,
            "max_workers": args.max_workers,
            "num_retries": args.sim_target_max_retries,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        kwargs.update(
            _reasoning_effort_override_for_model(
                model,
                disable_reasoning=True,
                reasoning_effort=None,
            )
        )
        responses = litellm.batch_completion(**kwargs)
        if hasattr(responses, "__iter__") and not isinstance(responses, list):
            return list(responses), None
        if isinstance(responses, list):
            return responses, None
        return None, "batch call returned non-list response container"
    except (RuntimeError, ValueError, OSError, litellm.OpenAIError) as error:
        return None, f"{type(error).__name__}: {error}"


def _call_batched_policy(
    *,
    args: argparse.Namespace,
    model_list: list[str],
    messages_list: list[list[dict[str, str]]],
) -> tuple[list[dict[str, Any] | Exception] | None, str | None]:
    """Call batched policy generation using replay timeout/retry settings."""
    try:
        responses = batch_chat(
            model=model_list,
            messages_list=messages_list,
            timeout=args.sim_target_timeout_s,
            max_workers=args.max_workers,
            temperature=None,
            num_retries=args.sim_target_max_retries,
        )
        return responses, None
    except (RuntimeError, ValueError, OSError, litellm.OpenAIError) as error:
        return None, f"{type(error).__name__}: {error}"


def _finalize_replay_state_row(
    *,
    state: ReplayActiveState,
) -> dict[str, Any] | None:
    """Build final round-error row for one completed replay state."""

    job = state.job
    replay_round = state.env.get_round()
    replay_initial_nodes, replay_final_nodes = _replay_node_beliefs(
        state.env,
        spec=job.spec,
        human_round=job.row.round_obj,
        replay_round=replay_round,
    )
    context = ReplayRoundContext(
        spec=job.spec,
        source_row=job.row,
        human_round=job.row.round_obj,
        replay_round=replay_round,
        replay_initial_nodes=replay_initial_nodes,
        replay_final_nodes=replay_final_nodes,
        n_persuader_turns=len(_extract_persuader_messages(replay_round)),
        replay_index=int(job.replay_index),
        initialization_mode=str(state.initialization_mode),
        same_bin_scheme=str(state.same_bin_scheme),
        persuader_turn_mode=str(job.persuader_turn_mode),
    )
    return _build_round_error_row(context)


def _build_replay_step_row(
    *,
    state: ReplayActiveState,
    step_info: dict[str, Any] | None,
    policy_messages: list[dict[str, str]],
    pre_step_env_state_json: str,
    post_step_env_state_json: str,
) -> dict[str, Any]:
    """Build one rollout-style step row from a replay transition."""
    step_record = step_info.get("step_record") if isinstance(step_info, dict) else None
    if isinstance(step_record, dict):
        step_row = dict(step_record)
    else:
        step_row = {
            "episode_id": str(state.rollout.episode_id),
            "step_index": int(state.turn_index),
            "persuader_text": str(state.job.persuader_messages[state.turn_index]),
            "persuader_thought": None,
            "persuader_reasoning_trace": None,
            "target_text": "",
            "belief_before": None,
            "belief_after": None,
            "done": False,
            "truncated": False,
            "reward": {},
            "turns_left_for_persuader": max(
                0, len(state.job.persuader_messages) - int(state.turn_index) - 1
            ),
            "target_can_end_round": False,
            "metadata": {},
        }

    source_round = state.job.row.round_obj
    initial_belief = (
        float(source_round.target_initial_belief)
        if isinstance(source_round.target_initial_belief, (int, float))
        else None
    )
    turns_left_before_step = max(
        0, len(state.job.persuader_messages) - int(step_row.get("step_index", 0))
    )
    common_metadata = _replay_common_rollout_metadata(
        state=state,
        proposition_during_round=source_round.proposition_during_round,
        continuous_measure=str(ContinuousMeasure.SERIAL_QUESTIONS),
        matched_initial_belief=initial_belief,
    )
    serialized = build_step_row(
        step_record=step_row,
        common_metadata=common_metadata,
        policy_messages=policy_messages,
        turns_left_before_step=int(turns_left_before_step),
        env_state_json=(pre_step_env_state_json, post_step_env_state_json),
    )
    serialized.update(
        {
            "source_path": str(state.job.row.source_path),
            "source_line_index": int(state.job.row.source_line_index),
            "source_round_index": (
                int(state.job.row.source_round_index)
                if state.job.row.source_round_index is not None
                else None
            ),
            "replay_index": int(state.job.replay_index),
            "corpus": str(state.job.spec.corpus),
            "initialization_mode": str(state.initialization_mode),
            "same_bin_scheme": (
                str(state.same_bin_scheme)
                if str(state.initialization_mode) == "same-bin"
                else "na"
            ),
            "persuader_turn_mode": str(state.job.persuader_turn_mode),
        }
    )
    return serialized


def _build_replay_episode_row(
    *,
    state: ReplayActiveState,
    replay_round: Round,
) -> dict[str, Any]:
    """Build one rollout-style episode row from a finalized replay state."""
    initial_belief = (
        float(replay_round.target_initial_belief)
        if isinstance(replay_round.target_initial_belief, (int, float))
        else None
    )
    common_metadata = _replay_common_rollout_metadata(
        state=state,
        proposition_during_round=replay_round.proposition_during_round,
        continuous_measure=str(ContinuousMeasure.SERIAL_QUESTIONS),
        matched_initial_belief=initial_belief,
    )
    serialized = build_episode_row(
        round_obj=replay_round,
        common_metadata=common_metadata,
        replay_count=int(state.job.replay_index),
        steps_taken=int(len(state.rollout.step_rows)),
        episode_reward_sum=float(state.rollout.reward_sum),
    )
    serialized.update(
        {
            "source_path": str(state.job.row.source_path),
            "source_line_index": int(state.job.row.source_line_index),
            "source_round_index": (
                int(state.job.row.source_round_index)
                if state.job.row.source_round_index is not None
                else None
            ),
            "replay_index": int(state.job.replay_index),
            "corpus": str(state.job.spec.corpus),
            "initialization_mode": str(state.initialization_mode),
            "same_bin_scheme": (
                str(state.same_bin_scheme)
                if str(state.initialization_mode) == "same-bin"
                else "na"
            ),
            "persuader_turn_mode": str(state.job.persuader_turn_mode),
        }
    )
    return serialized


def _capture_step_and_reward(
    *,
    state: ReplayActiveState,
    step_info: dict[str, Any] | None,
    policy_messages: list[dict[str, str]],
    reward_value: float | int | None,
    pre_step_env_state_json: str,
    post_step_env_state_json: str,
) -> None:
    """Append one serialized step row and accumulate reward for one state."""
    if isinstance(reward_value, (int, float)):
        state.rollout.reward_sum += float(reward_value)
    step_row = _build_replay_step_row(
        state=state,
        step_info=step_info,
        policy_messages=policy_messages,
        pre_step_env_state_json=pre_step_env_state_json,
        post_step_env_state_json=post_step_env_state_json,
    )
    state.rollout.step_rows.append(step_row)


def _replay_node_beliefs(
    env: SimTargetEnv,
    *,
    spec: ReplayCorpusSpec,
    human_round: Round,
    replay_round: Round,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Extract replay initial/final node beliefs for evaluation."""
    if spec.target_backend != "simulated_target":
        return (
            replay_round.target_initial_node_beliefs,
            replay_round.target_final_node_beliefs,
        )

    sim_target = env.get_target()
    node_ids = [item["id"] for item in human_round.belief_survey_items()]
    replay_initial_nodes = _extract_distribution_node_beliefs(
        sim_target,
        index=0,
        node_ids=node_ids,
    )
    replay_final_nodes = _extract_distribution_node_beliefs(
        sim_target,
        index=-1,
        node_ids=node_ids,
    )
    return replay_initial_nodes, replay_final_nodes


def _build_round_error_row(context: ReplayRoundContext) -> dict[str, Any]:
    """Build one output row containing replay error metrics."""
    spec = context.spec
    source_row = context.source_row
    human_round = context.human_round
    replay_round = context.replay_round
    human_initial = float(human_round.target_initial_belief)
    human_final = float(human_round.target_final_belief)
    replay_initial = (
        float(replay_round.target_initial_belief)
        if isinstance(replay_round.target_initial_belief, (int, float))
        else math.nan
    )
    replay_final = (
        float(replay_round.target_final_belief)
        if isinstance(replay_round.target_final_belief, (int, float))
        else math.nan
    )
    final_target_abs_error = (
        abs(replay_final - human_final) if math.isfinite(replay_final) else math.nan
    )
    human_serial = human_round.get_serial_questions() or []
    replay_serial = replay_round.get_serial_questions() or []

    return {
        "corpus": spec.corpus,
        "target_backend": spec.target_backend,
        "target_model": spec.target_model,
        "persona": spec.persona.value,
        "llm_target_use_bayes_structure": bool(spec.llm_target_use_bayes_structure),
        "simulated_target_no_rhetoric": bool(spec.simulated_target_no_rhetoric),
        "source_path": str(source_row.source_path),
        "source_line_index": int(source_row.source_line_index),
        "source_round_index": (
            int(source_row.source_round_index)
            if source_row.source_round_index is not None
            else None
        ),
        "replay_index": int(context.replay_index),
        "initialization_mode": str(context.initialization_mode),
        "same_bin_scheme": (
            str(context.same_bin_scheme)
            if str(context.initialization_mode) == "same-bin"
            else "na"
        ),
        "persuader_turn_mode": str(context.persuader_turn_mode),
        "proposition": source_row.proposition,
        "n_persuader_turns": int(context.n_persuader_turns),
        "human_initial_belief": human_initial,
        "human_final_belief": human_final,
        "replay_initial_belief": replay_initial,
        "replay_final_belief": replay_final,
        "final_target_abs_error": final_target_abs_error,
        "serial_trajectory_mae": mean_abs_error(human_serial, replay_serial),
        "initial_node_mae": node_mae(
            human_round.target_initial_node_beliefs,
            context.replay_initial_nodes,
        ),
        "final_node_mae": node_mae(
            human_round.target_final_node_beliefs,
            context.replay_final_nodes,
        ),
        "node_delta_mae": node_delta_mae(
            human_round.target_initial_node_beliefs,
            human_round.target_final_node_beliefs,
            context.replay_initial_nodes,
            context.replay_final_nodes,
        ),
    }


def _run_replay_jobs_batched(
    args: argparse.Namespace,
    jobs: list[ReplayJob],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute replay jobs with the same batched target-query pipeline."""

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")

    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    with tqdm(
        total=int(len(jobs)),
        desc="Counterfactual Replay Jobs",
        unit="job",
        leave=True,
    ) as progress:
        for start in range(0, len(jobs), int(args.batch_size)):
            chunk = jobs[start : start + int(args.batch_size)]
            active_states: list[ReplayActiveState] = []
            for job in chunk:
                try:
                    env = _build_replay_env(args, job=job)
                    _assert_batched_env_methods(env, str(job.spec.target_backend))
                except (RuntimeError, ValueError, OSError) as error:
                    print(
                        "Replay skip:",
                        "init_failed",
                        f"corpus={job.spec.corpus}",
                        f"source_path={job.row.source_path}",
                        f"source_line_index={int(job.row.source_line_index)}",
                        f"source_round_index={job.row.source_round_index}",
                        f"replay_index={int(job.replay_index)}",
                        f"error={type(error).__name__}: {error}",
                    )
                    progress.update(1)
                    continue
                active_states.append(
                    ReplayActiveState(
                        job=job,
                        env=env,
                        rollout=ReplayRolloutBuffer(
                            episode_id=_episode_id_from_env_or_fallback(env, job),
                            pair_id=_pair_id_from_job(job),
                            step_rows=[],
                            reward_sum=0.0,
                        ),
                        turn_index=0,
                        initialization_mode=str(args.initialization_mode),
                        same_bin_scheme=str(args.same_bin_scheme),
                    )
                )

            while active_states:
                next_states: list[ReplayActiveState] = []
                llm_items: list[dict[str, Any]] = []
                sim_items: list[dict[str, Any]] = []
                step_seed_items: list[dict[str, Any]] = []
                policy_generation_items: list[dict[str, Any]] = []

                for state in active_states:
                    job = state.job
                    if state.turn_index >= len(job.persuader_messages):
                        try:
                            row = _finalize_replay_state_row(state=state)
                        except (RuntimeError, ValueError, OSError) as error:
                            _log_replay_skip(
                                state=state,
                                stage="finalize_failed",
                                error=f"{type(error).__name__}: {error}",
                            )
                        else:
                            if row is not None:
                                rows.append(row)
                                episode_rows.append(
                                    _build_replay_episode_row(
                                        state=state,
                                        replay_round=state.env.get_round(),
                                    )
                                )
                                step_rows.extend(state.rollout.step_rows)
                        progress.update(1)
                        continue

                    mode = str(job.persuader_turn_mode)
                    if not _should_generate_policy_turn(
                        mode=mode,
                        turn_index=int(state.turn_index),
                    ):
                        action_text = str(job.persuader_messages[state.turn_index])
                        step_seed_items.append(
                            {
                                "state": state,
                                "action_text": action_text,
                                "persuader_thought": None,
                                "persuader_reasoning_trace": None,
                                "policy_messages": [],
                            }
                        )
                        continue

                    policy_model = job.policy_model
                    if not isinstance(policy_model, str) or not policy_model.strip():
                        _log_replay_skip(
                            state=state,
                            stage="policy_model_missing",
                            error="No llm_persuader configured for policy mode.",
                        )
                        progress.update(1)
                        continue
                    round_obj = state.env.get_round()
                    if is_naive_persuader_model(policy_model):
                        action_text, thought, reasoning = (
                            naive_persuader_action_for_round(round_obj)
                        )
                        step_seed_items.append(
                            {
                                "state": state,
                                "action_text": action_text,
                                "persuader_thought": thought,
                                "persuader_reasoning_trace": reasoning,
                                "policy_messages": [],
                            }
                        )
                        continue

                    policy_messages = _policy_messages_for_round(
                        round_obj,
                        include_chain_of_thought=True,
                    )
                    policy_generation_items.append(
                        {
                            "state": state,
                            "policy_model": policy_model,
                            "policy_messages": policy_messages,
                        }
                    )

                if policy_generation_items:
                    policy_responses, policy_batch_error = _call_batched_policy(
                        args=args,
                        model_list=[
                            str(item["policy_model"])
                            for item in policy_generation_items
                        ],
                        messages_list=[
                            list(item["policy_messages"])
                            for item in policy_generation_items
                        ],
                    )
                    if policy_responses is None or len(policy_responses) != len(
                        policy_generation_items
                    ):
                        for item in policy_generation_items:
                            _log_replay_skip(
                                state=item["state"],
                                stage="policy_batch_call_failed",
                                error=str(
                                    policy_batch_error
                                    or "unknown policy batch call error"
                                ),
                            )
                            progress.update(1)
                    else:
                        for item, response in zip(
                            policy_generation_items, policy_responses, strict=True
                        ):
                            state = item["state"]
                            if isinstance(response, Exception) or not hasattr(
                                response, "get"
                            ):
                                _log_replay_skip(
                                    state=state,
                                    stage="policy_response_invalid",
                                    error=f"type={type(response).__name__}",
                                )
                                progress.update(1)
                                continue
                            try:
                                response_text = extract_text_from_response(response)
                            except (AttributeError, TypeError, ValueError) as error:
                                _log_replay_skip(
                                    state=state,
                                    stage="policy_response_extract_failed",
                                    error=f"{type(error).__name__}: {error}",
                                )
                                progress.update(1)
                                continue
                            action_text, thought, reasoning = _policy_action_from_text(
                                response_text
                            )
                            action_text = str(action_text).strip()
                            if not action_text:
                                _log_replay_skip(
                                    state=state,
                                    stage="policy_empty_action",
                                    error="Parsed policy action text was empty.",
                                )
                                progress.update(1)
                                continue
                            step_seed_items.append(
                                {
                                    "state": state,
                                    "action_text": action_text,
                                    "persuader_thought": thought,
                                    "persuader_reasoning_trace": reasoning,
                                    "policy_messages": list(
                                        item.get("policy_messages") or []
                                    ),
                                }
                            )

                for item in step_seed_items:
                    state = item["state"]
                    action_text = str(item["action_text"])
                    pre_step_env_state_json = _safe_env_state_json(state.env)
                    try:
                        prepared_step = state.env.build_batched_step_input(
                            action_text=action_text,
                            persuader_thought=item.get("persuader_thought"),
                            persuader_reasoning_trace=item.get(
                                "persuader_reasoning_trace"
                            ),
                        )
                    except (RuntimeError, ValueError, OSError) as error:
                        _log_replay_skip(
                            state=state,
                            stage="build_batched_step_input_failed",
                            error=f"{type(error).__name__}: {error}",
                        )
                        progress.update(1)
                        continue

                    entry = {
                        "state": state,
                        "prepared_step": prepared_step,
                        "pre_step_env_state_json": pre_step_env_state_json,
                        "policy_messages": list(item.get("policy_messages") or []),
                    }
                    if state.job.spec.target_backend == "llm_target":
                        llm_items.append(entry)
                    else:
                        sim_items.append(entry)

                for model in sorted(
                    {
                        str(item["state"].job.spec.target_model)
                        for item in llm_items
                        if isinstance(item.get("state"), ReplayActiveState)
                    }
                ):
                    model_items = [
                        item
                        for item in llm_items
                        if str(item["state"].job.spec.target_model) == model
                    ]
                    valid_items: list[dict[str, Any]] = []
                    messages_batch: list[list[dict[str, str]]] = []
                    for item in model_items:
                        target_messages = item["prepared_step"].get("target_messages")
                        if not isinstance(target_messages, list):
                            _log_replay_skip(
                                state=item["state"],
                                stage="llm_target_messages_invalid",
                                error="prepared_step.target_messages was not a list",
                            )
                            progress.update(1)
                            continue
                        valid_items.append(item)
                        messages_batch.append(target_messages)
                    if not valid_items:
                        continue

                    responses, batch_error = _call_batched_target(
                        args=args,
                        model=model,
                        messages=messages_batch,
                        response_format=LLMTargetTurnOutput,
                    )
                    if responses is None or len(responses) != len(valid_items):
                        for item in valid_items:
                            _log_replay_skip(
                                state=item["state"],
                                stage="llm_target_batch_call_failed",
                                error=str(
                                    batch_error or "unknown llm target batch error"
                                ),
                            )
                            progress.update(1)
                        continue

                    for item, response in zip(valid_items, responses, strict=True):
                        state = item["state"]
                        if isinstance(response, Exception) or not hasattr(
                            response, "get"
                        ):
                            _log_replay_skip(
                                state=state,
                                stage="llm_target_response_invalid",
                                error=f"type={type(response).__name__}",
                            )
                            progress.update(1)
                            continue
                        try:
                            response_text = extract_text_from_response(response)
                        except (AttributeError, TypeError, ValueError) as error:
                            _log_replay_skip(
                                state=state,
                                stage="llm_target_response_extract_failed",
                                error=f"{type(error).__name__}: {error}",
                            )
                            progress.update(1)
                            continue

                        survey_node_ids = item["prepared_step"].get("survey_node_ids")
                        parsed_payload = state.env.parse_llm_target_turn_payload(
                            response_text,
                            (
                                survey_node_ids
                                if isinstance(survey_node_ids, list)
                                else []
                            ),
                        )
                        if parsed_payload is None:
                            _log_replay_skip(
                                state=state,
                                stage="llm_target_parse_failed",
                                error="failed to parse llm target payload",
                            )
                            progress.update(1)
                            continue
                        target_reply, target_belief, target_node_beliefs = (
                            parsed_payload
                        )
                        complete_step_kwargs: dict[str, Any] = {
                            "prepared_step": item["prepared_step"],
                            "target_reply": target_reply,
                            "target_belief": float(target_belief),
                        }
                        if isinstance(target_node_beliefs, dict):
                            complete_step_kwargs["target_node_beliefs"] = dict(
                                target_node_beliefs
                            )
                        try:
                            (
                                _,
                                reward_value,
                                done,
                                truncated,
                                step_info,
                            ) = state.env.complete_batched_step(**complete_step_kwargs)
                        except (RuntimeError, ValueError, OSError) as error:
                            _log_replay_skip(
                                state=state,
                                stage="llm_target_complete_step_failed",
                                error=f"{type(error).__name__}: {error}",
                            )
                            progress.update(1)
                            continue

                        post_step_env_state_json = _safe_env_state_json(state.env)
                        _capture_step_and_reward(
                            state=state,
                            step_info=(
                                step_info if isinstance(step_info, dict) else None
                            ),
                            policy_messages=list(item.get("policy_messages") or []),
                            reward_value=(
                                float(reward_value)
                                if isinstance(reward_value, (int, float))
                                else None
                            ),
                            pre_step_env_state_json=str(
                                item.get("pre_step_env_state_json", "")
                            ),
                            post_step_env_state_json=post_step_env_state_json,
                        )
                        state.turn_index += 1
                        if (
                            bool(done)
                            or bool(truncated)
                            or (state.turn_index >= len(state.job.persuader_messages))
                        ):
                            try:
                                row = _finalize_replay_state_row(state=state)
                            except (RuntimeError, ValueError, OSError) as error:
                                _log_replay_skip(
                                    state=state,
                                    stage="finalize_failed",
                                    error=f"{type(error).__name__}: {error}",
                                )
                            else:
                                if row is not None:
                                    rows.append(row)
                                    episode_rows.append(
                                        _build_replay_episode_row(
                                            state=state,
                                            replay_round=state.env.get_round(),
                                        )
                                    )
                                    step_rows.extend(state.rollout.step_rows)
                            progress.update(1)
                        else:
                            next_states.append(state)

                for model in sorted(
                    {
                        str(item["state"].job.spec.target_model)
                        for item in sim_items
                        if isinstance(item.get("state"), ReplayActiveState)
                    }
                ):
                    model_items = [
                        item
                        for item in sim_items
                        if str(item["state"].job.spec.target_model) == model
                    ]
                    valid_items: list[dict[str, Any]] = []
                    atom_messages_batch: list[list[dict[str, str]]] = []
                    for item in model_items:
                        atom_messages = item["prepared_step"].get(
                            "atomization_messages"
                        )
                        if not isinstance(atom_messages, list):
                            _log_replay_skip(
                                state=item["state"],
                                stage="sim_target_atomization_messages_invalid",
                                error="prepared_step.atomization_messages was not a list",
                            )
                            progress.update(1)
                            continue
                        valid_items.append(item)
                        atom_messages_batch.append(atom_messages)
                    if not valid_items:
                        continue

                    atom_responses, atom_batch_error = _call_batched_target(
                        args=args,
                        model=model,
                        messages=atom_messages_batch,
                        response_format=MessageAnalysis,
                        temperature=atomizer_temperature_for_model(model),
                    )
                    if atom_responses is None or len(atom_responses) != len(
                        valid_items
                    ):
                        for item in valid_items:
                            _log_replay_skip(
                                state=item["state"],
                                stage="sim_target_atomization_batch_failed",
                                error=str(
                                    atom_batch_error or "unknown atomization error"
                                ),
                            )
                            progress.update(1)
                        continue

                    response_prompt_items: list[dict[str, Any]] = []
                    response_messages_batch: list[list[dict[str, str]]] = []
                    for item, atom_response in zip(
                        valid_items, atom_responses, strict=True
                    ):
                        state = item["state"]
                        if isinstance(atom_response, Exception) or not hasattr(
                            atom_response, "get"
                        ):
                            _log_replay_skip(
                                state=state,
                                stage="sim_target_atomization_response_invalid",
                                error=f"type={type(atom_response).__name__}",
                            )
                            progress.update(1)
                            continue
                        try:
                            atom_text = extract_text_from_response(atom_response)
                            atoms = state.env.get_target().parse_atomization_content(
                                atom_text
                            )
                            response_prompt = state.env.build_batched_response_messages(
                                prepared_step=item["prepared_step"],
                                atoms=atoms,
                            )
                        except (RuntimeError, ValueError, OSError, TypeError) as error:
                            _log_replay_skip(
                                state=state,
                                stage="sim_target_atomization_parse_or_build_failed",
                                error=f"{type(error).__name__}: {error}",
                            )
                            progress.update(1)
                            continue
                        response_prompt_items.append(item)
                        response_messages_batch.append(response_prompt)

                    if not response_prompt_items:
                        continue

                    target_responses, target_batch_error = _call_batched_target(
                        args=args,
                        model=model,
                        messages=response_messages_batch,
                    )
                    if target_responses is None or len(target_responses) != len(
                        response_prompt_items
                    ):
                        for item in response_prompt_items:
                            _log_replay_skip(
                                state=item["state"],
                                stage="sim_target_response_batch_failed",
                                error=str(
                                    target_batch_error or "unknown response error"
                                ),
                            )
                            progress.update(1)
                        continue

                    for item, target_response in zip(
                        response_prompt_items, target_responses, strict=True
                    ):
                        state = item["state"]
                        if isinstance(target_response, Exception) or not hasattr(
                            target_response, "get"
                        ):
                            _log_replay_skip(
                                state=state,
                                stage="sim_target_response_invalid",
                                error=f"type={type(target_response).__name__}",
                            )
                            progress.update(1)
                            continue
                        try:
                            target_reply = extract_text_from_response(target_response)
                            (
                                _,
                                reward_value,
                                done,
                                truncated,
                                step_info,
                            ) = state.env.complete_batched_step(
                                prepared_step=item["prepared_step"],
                                target_reply=target_reply,
                            )
                        except (RuntimeError, ValueError, OSError, TypeError) as error:
                            _log_replay_skip(
                                state=state,
                                stage="sim_target_complete_step_failed",
                                error=f"{type(error).__name__}: {error}",
                            )
                            progress.update(1)
                            continue

                        post_step_env_state_json = _safe_env_state_json(state.env)
                        _capture_step_and_reward(
                            state=state,
                            step_info=(
                                step_info if isinstance(step_info, dict) else None
                            ),
                            policy_messages=list(item.get("policy_messages") or []),
                            reward_value=(
                                float(reward_value)
                                if isinstance(reward_value, (int, float))
                                else None
                            ),
                            pre_step_env_state_json=str(
                                item.get("pre_step_env_state_json", "")
                            ),
                            post_step_env_state_json=post_step_env_state_json,
                        )
                        state.turn_index += 1
                        if (
                            bool(done)
                            or bool(truncated)
                            or (state.turn_index >= len(state.job.persuader_messages))
                        ):
                            try:
                                row = _finalize_replay_state_row(state=state)
                            except (RuntimeError, ValueError, OSError) as error:
                                _log_replay_skip(
                                    state=state,
                                    stage="finalize_failed",
                                    error=f"{type(error).__name__}: {error}",
                                )
                            else:
                                if row is not None:
                                    rows.append(row)
                                    episode_rows.append(
                                        _build_replay_episode_row(
                                            state=state,
                                            replay_round=state.env.get_round(),
                                        )
                                    )
                                    step_rows.extend(state.rollout.step_rows)
                            progress.update(1)
                        else:
                            next_states.append(state)

                active_states = next_states

    return rows, episode_rows, step_rows


def _load_round_error_rows_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load round-error rows from JSONL and coerce numeric error fields."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            for key in (
                "final_target_abs_error",
                "serial_trajectory_mae",
                "final_node_mae",
                "node_delta_mae",
            ):
                value = row.get(key)
                if value is None or value == "":
                    row[key] = math.nan
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = math.nan
            rows.append(row)
    return rows


def _write_round_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write round rows as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load arbitrary dictionary JSONL rows, skipping malformed lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _merge_episode_rows(
    existing_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge episode rows by episode_id, preferring fresh rows."""
    merged: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str) and episode_id:
            merged[episode_id] = row
    for row in fresh_rows:
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str) and episode_id:
            merged[episode_id] = row
    return [merged[key] for key in sorted(merged)]


def _merge_step_rows(
    existing_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge step rows by (episode_id, step_index), preferring fresh rows."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for row in existing_rows:
        episode_id = row.get("episode_id")
        step_index = row.get("step_index")
        if not isinstance(episode_id, str) or not episode_id:
            continue
        try:
            parsed_step_index = int(step_index)
        except (TypeError, ValueError):
            continue
        merged[(episode_id, parsed_step_index)] = row
    for row in fresh_rows:
        episode_id = row.get("episode_id")
        step_index = row.get("step_index")
        if not isinstance(episode_id, str) or not episode_id:
            continue
        try:
            parsed_step_index = int(step_index)
        except (TypeError, ValueError):
            continue
        merged[(episode_id, parsed_step_index)] = row
    ordered_keys = sorted(merged, key=lambda item: (item[0], item[1]))
    return [merged[key] for key in ordered_keys]


def _build_dry_run_plan(
    *,
    args: argparse.Namespace,
    dry_rows: list[dict[str, Any]],
    reused_count: int,
    pending_count: int,
) -> dict[str, Any]:
    """Build a baseline-runner-style dry-run plan payload.

    Args:
        args: Parsed CLI arguments.
        dry_rows: Per-corpus dry-run summary rows.
        reused_count: Number of replay jobs found in cache.
        pending_count: Number of replay jobs pending execution.

    Returns:
        A JSON-serializable dry-run plan dictionary.
    """
    turns_remaining = sum(int(row["turns"]) for row in dry_rows)
    estimated_calls = sum(int(row["estimated_calls"]) for row in dry_rows)
    return {
        "experiment_name": "simulator_counterfactual_replay",
        "num_cells": int(len(dry_rows)),
        "planned_total_jobs": int(reused_count + pending_count),
        "resume": bool(args.reuse_existing_round_errors),
        "existing_jobs": int(reused_count),
        "dry_run": True,
        "replays_per_round": int(args.replays_per_round),
        "initialization_mode": str(args.initialization_mode),
        "same_bin_scheme": str(args.same_bin_scheme),
        "persuader_turn_mode": str(args.persuader_turn_mode),
        "estimation": {
            "jobs_remaining": int(pending_count),
            "turns_remaining": int(turns_remaining),
            "estimated_calls_remaining": int(estimated_calls),
            "cells": dry_rows,
        },
    }


def _write_dry_run_artifacts(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> tuple[Path, Path]:
    """Write dry-run plan and config snapshot artifacts.

    Args:
        args: Parsed CLI arguments.
        plan: Dry-run plan payload.

    Returns:
        Tuple of written `(plan_path, config_snapshot_path)`.
    """
    plan_path = Path(f"{args.output_prefix}_plan.json")
    config_snapshot_path = Path(f"{args.output_prefix}_config_snapshot.json")
    return write_plan_and_config_snapshot_to_paths(
        plan_path=plan_path,
        config_snapshot_path=config_snapshot_path,
        plan=plan,
        config_snapshot=dict(vars(args)),
    )


def main() -> None:
    """Run replay-based simulator-vs-human evaluation."""
    args = parse_args()
    round_jsonl_path = Path(f"{args.output_prefix}_round_errors.jsonl")
    episodes_jsonl_path = Path(f"{args.output_prefix}_episodes.jsonl")
    steps_jsonl_path = Path(f"{args.output_prefix}_steps.jsonl")
    legacy_round_csv_path = Path(f"{args.output_prefix}_round_errors.csv")
    summary_csv_path = Path(f"{args.output_prefix}_summary.csv")

    specs = _build_corpus_specs(args)
    if not specs:
        raise ValueError("No replay corpora selected. Enable at least one corpus flag.")
    replayable_rows = _load_replayable_rows(args)
    jobs = _build_replay_jobs(
        specs,
        replayable_rows,
        replays_per_round=int(args.replays_per_round),
        persuader_turn_mode=str(args.persuader_turn_mode),
    )
    if not jobs:
        raise ValueError("No replay jobs were constructed from current filters.")

    reused_round_rows: list[dict[str, Any]] = []
    pending_jobs = jobs
    if bool(args.reuse_existing_round_errors) and (
        round_jsonl_path.exists() or legacy_round_csv_path.exists()
    ):
        if round_jsonl_path.exists():
            existing_rows = _load_round_error_rows_jsonl(round_jsonl_path)
            cache_path_used = round_jsonl_path
        else:
            existing_rows = load_round_error_rows(legacy_round_csv_path)
            cache_path_used = legacy_round_csv_path
        reused_round_rows, pending_jobs = _split_jobs_by_cache(
            jobs,
            existing_rows,
            initialization_mode=str(args.initialization_mode),
            same_bin_scheme=str(args.same_bin_scheme),
        )
        print(
            "Replay cache:",
            f"reused={len(reused_round_rows)}",
            f"pending={len(pending_jobs)}",
            f"path={cache_path_used}",
        )

    if args.dry_run:
        dry_rows = dry_run_rows(pending_jobs)
        plan = _build_dry_run_plan(
            args=args,
            dry_rows=dry_rows,
            reused_count=len(reused_round_rows),
            pending_count=len(pending_jobs),
        )
        plan_path, config_snapshot_path = _write_dry_run_artifacts(
            args=args,
            plan=plan,
        )
        print(json.dumps(plan, indent=2))
        print("Wrote outputs:", plan_path, config_snapshot_path)
        return

    fresh_round_rows: list[dict[str, Any]] = []
    fresh_episode_rows: list[dict[str, Any]] = []
    fresh_step_rows: list[dict[str, Any]] = []
    if pending_jobs:
        fresh_round_rows, fresh_episode_rows, fresh_step_rows = (
            _run_replay_jobs_batched(args, pending_jobs)
        )
    round_rows = [*reused_round_rows, *fresh_round_rows]
    if not round_rows:
        raise ValueError("Replay execution produced no rows.")
    for row in round_rows:
        row.setdefault("replay_index", 0)
        row.setdefault("initialization_mode", "exact")
        if str(row.get("initialization_mode")) == "same-bin":
            row.setdefault("same_bin_scheme", "fixed3")
        else:
            row.setdefault("same_bin_scheme", "na")
        row.setdefault("persuader_turn_mode", "human_replay")

    summary_rows = build_summary_rows(round_rows)
    table_rows = summary_table_rows(summary_rows)
    print_table(
        table_rows,
        columns=["corpus", "n", "target_err", "node_err", "score"],
        title="Counterfactual Replay Human-Likeness",
        aligns={"n": "right", "target_err": "right", "node_err": "right"},
    )

    if round_rows:
        _write_round_rows_jsonl(round_jsonl_path, round_rows)
    if fresh_episode_rows:
        existing_episode_rows = (
            _load_jsonl_rows(episodes_jsonl_path)
            if episodes_jsonl_path.exists()
            else []
        )
        merged_episode_rows = _merge_episode_rows(
            existing_rows=existing_episode_rows,
            fresh_rows=fresh_episode_rows,
        )
        _write_round_rows_jsonl(episodes_jsonl_path, merged_episode_rows)
    if fresh_step_rows:
        existing_step_rows = (
            _load_jsonl_rows(steps_jsonl_path) if steps_jsonl_path.exists() else []
        )
        merged_step_rows = _merge_step_rows(
            existing_rows=existing_step_rows,
            fresh_rows=fresh_step_rows,
        )
        _write_round_rows_jsonl(steps_jsonl_path, merged_step_rows)
    if summary_rows:
        write_csv_rows(summary_csv_path, summary_rows, list(summary_rows[0].keys()))

    print(
        "Wrote outputs:",
        round_jsonl_path,
        episodes_jsonl_path,
        steps_jsonl_path,
        summary_csv_path,
        f"(reused={len(reused_round_rows)} fresh={len(fresh_round_rows)})",
    )


if __name__ == "__main__":
    main()
