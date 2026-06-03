"""Export proposition-susceptibility inputs from explicit corpus selections."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from simulation.human_likeness import (
    RoundTrajectory,
    load_serial_trajectories,
    parse_min_date,
)
from simulation.human_likeness_eval.trajectory_metrics import (
    _proposition_stance_delta_rows,
    _round_dynamics_row,
)

from .utils import resolve_repo_path

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_PREFIX = Path("analysis/data/rl_human_match_sim_compare")
_GPT5_DATED_MODEL_RE = re.compile(r"^gpt-5-\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class CorpusSlice:
    """Store one selected corpus with metadata.

    Attributes:
        corpus: Corpus key used in downstream exports.
        rows: Selected trajectory rows for the corpus.
    """

    corpus: str
    rows: list[RoundTrajectory]


@dataclass(frozen=True)
class CommonFilters:
    """Store shared filter settings for row selection.

    Attributes:
        include_control: Whether control rounds are allowed.
        include_audio: Whether audio rounds are allowed.
        turn_limit: Optional exact turn limit filter.
        participant_proposition: Tri-state participant proposition selector.
        include_bn_survey: Whether BN-survey rows are allowed.
    """

    include_control: bool
    include_audio: bool
    turn_limit: int | None
    participant_proposition: str
    include_bn_survey: bool


@dataclass(frozen=True)
class ExportPaths:
    """Store output paths for proposition-bias exports.

    Attributes:
        baseline_prop: Baseline proposition stance CSV.
        baseline_round: Baseline round dynamics CSV.
        by_policy_prop: Policy-split proposition stance CSV.
        by_policy_round: Policy-split round dynamics CSV.
        selection_summary: Selection metadata CSV.
    """

    baseline_prop: Path
    baseline_round: Path
    by_policy_prop: Path
    by_policy_round: Path
    selection_summary: Path


@dataclass(frozen=True)
class ExportPayload:
    """Store row payloads for proposition-bias CSV exports.

    Attributes:
        baseline_round_rows: Baseline round dynamics rows.
        baseline_prop_rows: Baseline proposition stance rows.
        policy_round_rows: Policy-split round dynamics rows.
        policy_prop_rows: Policy-split proposition stance rows.
        selection_rows: Selection metadata rows.
    """

    baseline_round_rows: list[dict[str, object]]
    baseline_prop_rows: list[dict[str, object]]
    policy_round_rows: list[dict[str, object]]
    policy_prop_rows: list[dict[str, object]]
    selection_rows: list[dict[str, object]]


ROUND_DYNAMICS_COLUMNS = [
    "corpus",
    "trajectory_index",
    "n_turns",
    "total_delta",
    "abs_total_delta",
    "raw_belief_delta",
    "initial_belief",
    "final_belief",
    "supports_proposition",
    "stance",
    "has_up_step",
    "has_down_step",
    "both_directions",
    "sign_changes",
    "max_up_step",
    "max_down_step",
    "abs_max_step",
    "source_path",
    "source_line_index",
    "source_round_index",
    "proposition",
]
PROPOSITION_STANCE_COLUMNS = [
    "corpus",
    "proposition",
    "stance",
    "n_rounds",
    "mean_total_delta",
    "median_total_delta",
    "mean_abs_total_delta",
    "toward_round_rate",
    "away_round_rate",
    "near_zero_round_rate",
]
SELECTION_COLUMNS = [
    "setting",
    "value",
    "corpus",
    "rows",
    "unique_propositions",
    "policy_models",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for proposition-bias export.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build proposition susceptibility source CSVs with explicit human and "
            "simulator filtering."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing exported JSONL round data.",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Minimum file date (YYYY-MM-DD) for results ingestion.",
    )
    parser.add_argument(
        "--human-source",
        choices=["llm-human-target", "human-human", "all-human-target"],
        default="llm-human-target",
        help="Human reference selector.",
    )
    parser.add_argument(
        "--human-persuader-model",
        type=str,
        default="openai/gpt-5",
        help="llm_persuader model id required for human-reference rounds.",
    )
    parser.add_argument(
        "--include-human-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include human_reference rows in exported proposition CSVs.",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Non-naive llm_persuader model id for simulator corpora.",
    )
    parser.add_argument(
        "--sim-baseline-policy-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--include-naive-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to also export naive-policy simulator slices.",
    )
    parser.add_argument(
        "--naive-policy-model",
        type=str,
        default="naive",
        help="Naive llm_persuader model id.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=4,
        help="Exact turn limit filter.",
    )
    parser.add_argument(
        "--participant-proposition",
        choices=["any", "true", "false"],
        default="false",
        help="Participant proposition filter.",
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        help="Include control-dialogue rounds.",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Include audio rounds.",
    )
    parser.add_argument(
        "--exclude-bn-survey",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude rounds where enable_node_belief_survey is true.",
    )
    parser.add_argument(
        "--allow-bn-survey-fallback",
        action="store_true",
        help=(
            "If non-BN filtering is too sparse, retry once including BN-survey "
            "rounds."
        ),
    )
    parser.add_argument(
        "--min-human-propositions",
        type=int,
        default=2,
        help="Minimum number of unique human propositions required.",
    )
    parser.add_argument(
        "--min-simulator-propositions",
        type=int,
        default=2,
        help="Minimum number of unique simulator propositions required.",
    )
    parser.add_argument(
        "--movement-epsilon",
        type=float,
        default=0.01,
        help="Near-zero threshold used by proposition stance aggregation.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix path for generated CSV files.",
    )
    return parser.parse_args()


def _matches_common_filters(
    row: RoundTrajectory,
    *,
    common_filters: CommonFilters,
) -> bool:
    """Apply shared selection filters.

    Args:
        row: Candidate trajectory row.
        common_filters: Shared filtering settings.

    Returns:
        True when row passes the shared filters.
    """
    condition = row.condition
    participant_filter = str(common_filters.participant_proposition)
    participant_ok = (
        participant_filter == "any"
        or (participant_filter == "true" and condition.participant_proposition)
        or (participant_filter == "false" and not condition.participant_proposition)
    )
    turn_ok = (
        common_filters.turn_limit is None
        or condition.turn_limit == common_filters.turn_limit
    )
    control_ok = common_filters.include_control or not condition.control_dialogue
    audio_ok = common_filters.include_audio or not condition.use_audio
    bn_ok = common_filters.include_bn_survey or not condition.enable_node_belief_survey
    return bool(participant_ok and turn_ok and control_ok and audio_ok and bn_ok)


def _canonical_policy_model_id(model_id: str | None) -> str:
    """Normalize policy model ids for stable matching across aliases.

    Args:
        model_id: Raw policy model id.

    Returns:
        Canonical policy model id.
    """
    model = str(model_id or "").strip()
    if not model:
        return ""
    if model == "naive":
        return "naive"
    normalized = model
    if normalized.startswith("openai/"):
        normalized = normalized[len("openai/") :]
    if normalized == "gpt-5" or _GPT5_DATED_MODEL_RE.fullmatch(normalized):
        return "openai/gpt-5"
    return model


def _selected_policy_model(args: argparse.Namespace) -> str:
    """Resolve the non-naive policy model id from CLI args.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Effective policy model id.
    """
    preferred = str(args.policy_model or "").strip()
    if preferred:
        return preferred
    legacy = str(args.sim_baseline_policy_model or "").strip()
    if legacy:
        return legacy
    return "gpt-5-2025-08-07"


def _is_llm_human_target(row: RoundTrajectory, human_source: str) -> bool:
    """Check whether a row belongs to the requested human source.

    Args:
        row: Candidate trajectory row.
        human_source: Human source selector.

    Returns:
        True when the row matches the human-source selector.
    """
    roles = row.condition.roles
    if human_source == "llm-human-target":
        return bool(roles.human_target and roles.llm_persuader)
    if human_source == "human-human":
        return bool(roles.human_target and roles.human_persuader)
    if human_source == "all-human-target":
        return bool(roles.human_target)
    raise ValueError(f"Unknown human source: {human_source}")


def _select_human_rows(
    rows: list[RoundTrajectory],
    *,
    human_source: str,
    human_persuader_model: str | None,
    common_filters: CommonFilters,
) -> list[RoundTrajectory]:
    """Select the human-reference slice.

    Args:
        rows: Candidate trajectory rows.
        human_source: Human source selector.
        human_persuader_model: Optional exact llm_persuader model filter.
        common_filters: Shared filtering settings.

    Returns:
        Filtered human-reference rows.
    """
    selected: list[RoundTrajectory] = []
    canonical_filter = (
        _canonical_policy_model_id(human_persuader_model)
        if human_persuader_model is not None
        else None
    )
    for row in rows:
        if not _matches_common_filters(
            row,
            common_filters=common_filters,
        ):
            continue
        if not _is_llm_human_target(row, human_source):
            continue
        if canonical_filter is not None:
            candidate = _canonical_policy_model_id(row.condition.roles.llm_persuader)
            if candidate != canonical_filter:
                continue
        selected.append(row)
    return selected


def _select_sim_rows(
    rows: list[RoundTrajectory],
    *,
    simulator_type: str,
    policy_model: str,
    common_filters: CommonFilters,
) -> list[RoundTrajectory]:
    """Select one simulator slice by type and persuader model.

    Args:
        rows: Candidate trajectory rows.
        simulator_type: One of ``structure``, ``full``, ``vanilla``.
        policy_model: Exact llm_persuader value required.
        common_filters: Shared filtering settings.

    Returns:
        Filtered simulator rows.
    """
    selected: list[RoundTrajectory] = []
    canonical_policy_model = _canonical_policy_model_id(policy_model)
    for row in rows:
        if not _matches_common_filters(
            row,
            common_filters=common_filters,
        ):
            continue
        candidate = _canonical_policy_model_id(row.condition.roles.llm_persuader)
        if candidate != canonical_policy_model:
            continue

        roles = row.condition.roles
        condition = row.condition
        if simulator_type == "structure":
            if not roles.llm_target or not condition.llm_target_use_bayes_structure:
                continue
        elif simulator_type == "full":
            if not roles.simulated_target or condition.simulated_target_no_rhetoric:
                continue
        elif simulator_type == "vanilla":
            if not roles.llm_target or condition.llm_target_use_bayes_structure:
                continue
        else:
            raise ValueError(f"Unknown simulator type: {simulator_type}")
        selected.append(row)
    return selected


def _unique_prop_count(rows: list[RoundTrajectory]) -> int:
    """Count unique proposition strings in a row list.

    Args:
        rows: Selected trajectory rows.

    Returns:
        Number of unique proposition values.
    """
    return len(
        {str(row.proposition).strip() for row in rows if str(row.proposition).strip()}
    )


def _build_round_dynamics_rows(
    *,
    slices: list[CorpusSlice],
    epsilon: float,
) -> list[dict[str, object]]:
    """Build round-level movement rows for selected corpora.

    Args:
        slices: Selected corpus slices.
        epsilon: Near-zero threshold for movement features.

    Returns:
        Round dynamics rows.
    """
    output: list[dict[str, object]] = []
    for corpus_slice in slices:
        for index, row in enumerate(corpus_slice.rows):
            output.append(
                _round_dynamics_row(
                    corpus=corpus_slice.corpus,
                    trajectory_index=index,
                    row=row,
                    epsilon=epsilon,
                )
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Write row dictionaries to CSV.

    Args:
        path: Output CSV path.
        rows: Row dictionaries.
        columns: Column order.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _policy_suffix(corpus: str, policy_model: str) -> str:
    """Build a policy-suffixed corpus key.

    Args:
        corpus: Base corpus key.
        policy_model: Policy model label.

    Returns:
        Policy-suffixed corpus key.
    """
    return f"{corpus}__policy={policy_model}"


def _build_selection(
    *,
    all_rows: list[RoundTrajectory],
    args: argparse.Namespace,
    include_bn_survey: bool,
) -> tuple[list[CorpusSlice], list[CorpusSlice]]:
    """Build baseline and policy-split corpus slices.

    Args:
        all_rows: Ingested trajectory rows.
        args: Parsed CLI arguments.
        include_bn_survey: Whether BN-survey rows are allowed.

    Returns:
        Tuple of ``(baseline_slices, policy_split_slices)``.
    """
    common_filters = CommonFilters(
        include_control=bool(args.include_control),
        include_audio=bool(args.include_audio),
        turn_limit=args.turn_limit,
        participant_proposition=str(args.participant_proposition),
        include_bn_survey=include_bn_survey,
    )
    include_human_reference = bool(args.include_human_reference)
    human_rows: list[RoundTrajectory] = []
    if include_human_reference:
        human_rows = _select_human_rows(
            all_rows,
            human_source=str(args.human_source),
            human_persuader_model=(
                str(args.human_persuader_model)
                if args.human_persuader_model is not None
                else None
            ),
            common_filters=common_filters,
        )

    policy_model = _selected_policy_model(args)
    naive_model = str(args.naive_policy_model)

    structure_baseline = _select_sim_rows(
        all_rows,
        simulator_type="structure",
        policy_model=policy_model,
        common_filters=common_filters,
    )
    full_baseline = _select_sim_rows(
        all_rows,
        simulator_type="full",
        policy_model=policy_model,
        common_filters=common_filters,
    )
    vanilla_baseline = _select_sim_rows(
        all_rows,
        simulator_type="vanilla",
        policy_model=policy_model,
        common_filters=common_filters,
    )

    baseline_slices = [
        CorpusSlice(corpus="vanilla_llm_target", rows=vanilla_baseline),
        CorpusSlice(corpus="structure_target", rows=structure_baseline),
        CorpusSlice(corpus="full_simulated_target", rows=full_baseline),
    ]
    policy_slices = [
        CorpusSlice(
            corpus=_policy_suffix("vanilla_llm_target", policy_model),
            rows=vanilla_baseline,
        ),
        CorpusSlice(
            corpus=_policy_suffix("structure_target", policy_model),
            rows=structure_baseline,
        ),
        CorpusSlice(
            corpus=_policy_suffix("full_simulated_target", policy_model),
            rows=full_baseline,
        ),
    ]
    if include_human_reference:
        baseline_slices.insert(
            0, CorpusSlice(corpus="human_reference", rows=human_rows)
        )
        policy_slices.insert(0, CorpusSlice(corpus="human_reference", rows=human_rows))

    if bool(args.include_naive_policy):
        structure_naive = _select_sim_rows(
            all_rows,
            simulator_type="structure",
            policy_model=naive_model,
            common_filters=common_filters,
        )
        full_naive = _select_sim_rows(
            all_rows,
            simulator_type="full",
            policy_model=naive_model,
            common_filters=common_filters,
        )
        vanilla_naive = _select_sim_rows(
            all_rows,
            simulator_type="vanilla",
            policy_model=naive_model,
            common_filters=common_filters,
        )
        policy_slices.extend(
            [
                CorpusSlice(
                    corpus=_policy_suffix("vanilla_llm_target", naive_model),
                    rows=vanilla_naive,
                ),
                CorpusSlice(
                    corpus=_policy_suffix("structure_target", naive_model),
                    rows=structure_naive,
                ),
                CorpusSlice(
                    corpus=_policy_suffix("full_simulated_target", naive_model),
                    rows=full_naive,
                ),
            ]
        )
    return baseline_slices, policy_slices


def _slice_stats_row(corpus_slice: CorpusSlice) -> dict[str, object]:
    """Build one slice statistics row.

    Args:
        corpus_slice: Selected corpus slice.

    Returns:
        Summary row.
    """
    models = Counter(
        _canonical_policy_model_id(row.condition.roles.llm_persuader)
        for row in corpus_slice.rows
        if row.condition.roles.llm_persuader
    )
    model_summary = ";".join(
        f"{model}:{count}"
        for model, count in sorted(models.items(), key=lambda item: item[0])
    )
    return {
        "corpus": corpus_slice.corpus,
        "rows": int(len(corpus_slice.rows)),
        "unique_propositions": int(_unique_prop_count(corpus_slice.rows)),
        "policy_models": model_summary,
    }


def _selection_is_usable(
    *,
    baseline_slices: list[CorpusSlice],
    min_human_props: int,
    min_simulator_props: int,
) -> tuple[bool, str]:
    """Validate minimum data requirements.

    Args:
        baseline_slices: Baseline corpus slices.
        min_human_props: Minimum unique human propositions.
        min_simulator_props: Minimum unique simulator propositions.

    Returns:
        Tuple ``(is_usable, reason)``.
    """
    by_name = {item.corpus: item for item in baseline_slices}
    structure = by_name["structure_target"]
    full = by_name["full_simulated_target"]
    vanilla = by_name["vanilla_llm_target"]

    if min_human_props > 0:
        human = by_name.get("human_reference")
        if human is None:
            return False, "human_reference corpus missing"
        human_props = _unique_prop_count(human.rows)
        if human_props < min_human_props:
            return False, (
                "insufficient human proposition coverage: "
                f"{human_props} < {min_human_props}"
            )
    if _unique_prop_count(structure.rows) < min_simulator_props:
        return False, "insufficient structured simulator proposition coverage"
    if _unique_prop_count(full.rows) < min_simulator_props:
        return False, "insufficient full simulator proposition coverage"
    if _unique_prop_count(vanilla.rows) < min_simulator_props:
        return False, "insufficient vanilla simulator proposition coverage"
    if not structure.rows or not full.rows or not vanilla.rows:
        return False, "missing simulator rows for one or more corpora"
    return True, ""


def _export_paths_from_prefix(prefix: Path) -> ExportPaths:
    """Build output paths from one prefix.

    Args:
        prefix: Output prefix path.

    Returns:
        Structured output paths.
    """
    return ExportPaths(
        baseline_prop=prefix.with_name(prefix.name + "_proposition_stance_deltas.csv"),
        baseline_round=prefix.with_name(prefix.name + "_round_dynamics.csv"),
        by_policy_prop=prefix.with_name(
            prefix.name + "_proposition_stance_deltas_by_policy.csv"
        ),
        by_policy_round=prefix.with_name(prefix.name + "_round_dynamics_by_policy.csv"),
        selection_summary=prefix.with_name(
            prefix.name + "_proposition_data_selection.csv"
        ),
    )


def _selection_summary_rows(
    *,
    args: argparse.Namespace,
    selected_bn_mode: bool,
    selected_policy: list[CorpusSlice],
) -> list[dict[str, object]]:
    """Build selection metadata rows for CSV export.

    Args:
        args: Parsed CLI namespace.
        selected_bn_mode: Whether BN-survey rows were included.
        selected_policy: Selected policy-split corpus slices.

    Returns:
        Summary rows.
    """
    return [
        {"setting": "include_bn_survey", "value": str(bool(selected_bn_mode))},
        {"setting": "human_source", "value": str(args.human_source)},
        {"setting": "human_persuader_model", "value": str(args.human_persuader_model)},
        {
            "setting": "policy_model",
            "value": _selected_policy_model(args),
        },
        {
            "setting": "include_human_reference",
            "value": str(bool(args.include_human_reference)),
        },
        {"setting": "naive_policy_model", "value": str(args.naive_policy_model)},
        {"setting": "turn_limit", "value": str(args.turn_limit)},
        {
            "setting": "participant_proposition",
            "value": str(args.participant_proposition),
        },
        {"setting": "include_control", "value": str(bool(args.include_control))},
        {"setting": "include_audio", "value": str(bool(args.include_audio))},
        {"setting": "min_date", "value": str(args.min_date)},
    ] + [_slice_stats_row(item) for item in selected_policy]


def _write_exports(
    *,
    paths: ExportPaths,
    payload: ExportPayload,
) -> None:
    """Write all proposition-bias export CSV files.

    Args:
        paths: Output paths.
        payload: Row payloads to persist.

    Returns:
        None.
    """
    _write_csv(
        paths.baseline_round, payload.baseline_round_rows, ROUND_DYNAMICS_COLUMNS
    )
    _write_csv(
        paths.baseline_prop, payload.baseline_prop_rows, PROPOSITION_STANCE_COLUMNS
    )
    _write_csv(paths.by_policy_round, payload.policy_round_rows, ROUND_DYNAMICS_COLUMNS)
    _write_csv(
        paths.by_policy_prop, payload.policy_prop_rows, PROPOSITION_STANCE_COLUMNS
    )
    _write_csv(paths.selection_summary, payload.selection_rows, SELECTION_COLUMNS)


def _build_export_payload(
    *,
    selected_baseline: list[CorpusSlice],
    selected_policy: list[CorpusSlice],
    epsilon: float,
    args: argparse.Namespace,
    selected_bn_mode: bool,
) -> ExportPayload:
    """Build all row payloads needed for output CSVs.

    Args:
        selected_baseline: Baseline corpus slices.
        selected_policy: Policy-split corpus slices.
        epsilon: Near-zero movement threshold.
        args: Parsed CLI namespace.
        selected_bn_mode: Whether BN-survey rows were included.

    Returns:
        Export payload.
    """
    baseline_round_rows = _build_round_dynamics_rows(
        slices=selected_baseline,
        epsilon=epsilon,
    )
    policy_round_rows = _build_round_dynamics_rows(
        slices=selected_policy,
        epsilon=epsilon,
    )
    return ExportPayload(
        baseline_round_rows=baseline_round_rows,
        baseline_prop_rows=_proposition_stance_delta_rows(
            round_dynamics_rows=baseline_round_rows,
            epsilon=epsilon,
        ),
        policy_round_rows=policy_round_rows,
        policy_prop_rows=_proposition_stance_delta_rows(
            round_dynamics_rows=policy_round_rows,
            epsilon=epsilon,
        ),
        selection_rows=_selection_summary_rows(
            args=args,
            selected_bn_mode=selected_bn_mode,
            selected_policy=selected_policy,
        ),
    )


def _select_bn_mode(
    *,
    all_rows: list[RoundTrajectory],
    args: argparse.Namespace,
) -> tuple[list[CorpusSlice], list[CorpusSlice], bool]:
    """Select usable baseline and policy slices across BN modes.

    Args:
        all_rows: Ingested trajectory rows.
        args: Parsed CLI namespace.

    Returns:
        Tuple of ``(baseline_slices, policy_slices, include_bn_mode)``.
    """
    allow_bn_fallback = bool(args.allow_bn_survey_fallback) and bool(
        args.exclude_bn_survey
    )
    bn_modes = [False] if bool(args.exclude_bn_survey) else [True]
    if allow_bn_fallback:
        bn_modes.append(True)

    reject_reasons: list[str] = []
    for include_bn in bn_modes:
        baseline_slices, policy_slices = _build_selection(
            all_rows=all_rows,
            args=args,
            include_bn_survey=include_bn,
        )
        is_usable, reason = _selection_is_usable(
            baseline_slices=baseline_slices,
            min_human_props=max(0, int(args.min_human_propositions)),
            min_simulator_props=max(1, int(args.min_simulator_propositions)),
        )
        if is_usable:
            return baseline_slices, policy_slices, include_bn
        reject_reasons.append(f"include_bn_survey={include_bn}: {reason}")

    reason_text = "\n".join(reject_reasons) if reject_reasons else "no valid selection"
    raise SystemExit(
        "Could not build proposition-bias export from the requested data slice.\n"
        + reason_text
    )


def main() -> None:
    """Execute proposition-bias CSV export workflow."""
    args = parse_args()
    reference_file = Path(__file__).resolve()
    results_dir = resolve_repo_path(args.results_dir, reference_file=reference_file)
    output_prefix = resolve_repo_path(args.output_prefix, reference_file=reference_file)

    min_date = parse_min_date(args.min_date)
    all_rows = load_serial_trajectories(results_dir, min_date=min_date)
    selected_baseline, selected_policy, selected_bn_mode = _select_bn_mode(
        all_rows=all_rows,
        args=args,
    )

    epsilon = max(0.0, float(args.movement_epsilon))
    payload = _build_export_payload(
        selected_baseline=selected_baseline,
        selected_policy=selected_policy,
        epsilon=epsilon,
        args=args,
        selected_bn_mode=selected_bn_mode,
    )
    output_paths = _export_paths_from_prefix(output_prefix)
    _write_exports(
        paths=output_paths,
        payload=payload,
    )

    print("Exported proposition susceptibility inputs:")
    print(f"  proposition rows: {output_paths.baseline_prop}")
    print(f"  round rows: {output_paths.baseline_round}")
    print(f"  by-policy proposition rows: {output_paths.by_policy_prop}")
    print(f"  by-policy round rows: {output_paths.by_policy_round}")
    print(f"  selection summary: {output_paths.selection_summary}")


if __name__ == "__main__":
    main()
