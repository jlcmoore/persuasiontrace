"""Export paper-facing assets to local files for manuscript appendices.

This script reads prompt templates directly from code used in experiments and
analysis, then writes reproducible snapshots under ``analysis/latex/generated``.
Re-run after upstream prompt edits to refresh all exported prompt artifacts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.latex.debategpt_bn_samples import (
    write_debategpt_bn_samples_table_tex,
)
from analysis.latex.latex_helpers import escape_latex_inline
from analysis.latex.proposition_samples import write_proposition_sample_table_tex
from analysis.simulator_llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeRound,
    render_round_prompt,
)
from annotation.prompt import format_dialogue_for_prompt
from annotation.runner import DEFAULT_MODEL as ANNOTATION_DEFAULT_MODEL
from annotation.runner import build_system_prompt as build_annotation_system_prompt
from experiment.condition import Condition, ContinuousMeasure, Roles
from experiment.endpoints import (
    PARTICIPANT_PROPOSITION_MODEL,
    PARTICIPANT_PROPOSITION_PROMPT,
)
from experiment.round import (
    LLM_HUMAN_LIKE_PROMPT_TEMPLATE,
    LLM_PERSUADER_NO_HEDGING,
    Round,
)
from rl.sim_target_env import (
    LLM_TARGET_TURN_JSON_PROMPT,
    LLM_TARGET_TURN_JSON_PROMPT_WITH_NODES,
)
from simulation.scripts.compute_joint_probabilities import (
    build_messages as build_joint_probability_messages,
)
from simulation.scripts.generate_belief_graphs import (
    prepare_messages as prepare_belief_graph_messages,
)
from simulation.target import (
    BayesianNetwork,
    BeliefEdge,
    BeliefRelevance,
    MessageAtom,
    RhetoricalModes,
    SimulatedTarget,
    TargetPersona,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERLEAF_INCLUDE_DIR = (
    REPO_ROOT.parent / "continuouspersuasion-overleaf" / "include" / "generated"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OVERLEAF_INCLUDE_DIR
    if DEFAULT_OVERLEAF_INCLUDE_DIR.exists()
    else (REPO_ROOT / "analysis" / "latex" / "generated")
)
DEFAULT_OVERLEAF_FIGURES_DIR = (
    REPO_ROOT.parent / "continuouspersuasion-overleaf" / "figures"
)
DEFAULT_FIGURE_OUTPUT_DIR = (
    DEFAULT_OVERLEAF_FIGURES_DIR
    if DEFAULT_OVERLEAF_FIGURES_DIR.exists()
    else (REPO_ROOT / "analysis" / "latex" / "generated" / "figures")
)
DEFAULT_FIGURE_MANIFEST_PATH = (
    REPO_ROOT / "analysis" / "latex" / "paper_figures_manifest.json"
)
DEFAULT_ATOMIZER_MODEL = "openai/gpt-5.4-mini"
DEFAULT_JUDGE_MODEL = "gpt-5.4-2026-03-17"
DEFAULT_BN_GRAPH_MODEL = "vertex_ai/gemini-3-flash-preview"
DEFAULT_JOINT_MODEL = "tsor13/spectrum-Llama-3.1-8B-v1"
SAMPLE_PROPOSITION = "[[PROPOSITION_PLACEHOLDER]]"
SAMPLE_BELIEF_1 = "[[BELIEF_1_PLACEHOLDER]]"
SAMPLE_BELIEF_2 = "[[BELIEF_2_PLACEHOLDER]]"


@dataclass
class PromptArtifact:
    """One exported prompt artifact.

    Attributes:
        slug: Stable machine-friendly identifier.
        title: Human-readable title.
        methods_reference: Where this prompt maps into manuscript methods.
        source_paths: Repo-relative paths that define this prompt.
        model: Primary model name used with this prompt.
        messages: Ordered chat messages in role/content format.
        notes: Optional implementation notes.
    """

    slug: str
    title: str
    methods_reference: str
    source_paths: list[str]
    model: str | None
    messages: list[dict[str, str]]
    notes: str = ""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prompt export.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export paper-facing prompt assets from code into local files for "
            "manuscript appendix usage."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for generated per-artifact TeX assets "
            "(default: ../continuouspersuasion-overleaf/include/generated when "
            "available, otherwise analysis/latex/generated)."
        ),
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    """Create a directory path if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _normalize_output_dir(output_dir: Path) -> Path:
    """Resolve output directory to an absolute path.

    Args:
        output_dir: Raw CLI output directory argument.

    Returns:
        Absolute normalized output directory path.
    """
    if output_dir.is_absolute():
        return output_dir
    return (REPO_ROOT / output_dir).resolve()


def _sanitize_listing_content(text: str) -> str:
    """Prepare text for a LaTeX listings block.

    Args:
        text: Raw text value.

    Returns:
        Content safe for lstlisting environment usage.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace(r"\end{lstlisting}", r"\end\{lstlisting\}")


def _display_message_role(raw_role: str) -> str:
    """Map internal message roles to manuscript-friendly role names.

    Args:
        raw_role: Raw message role name.

    Returns:
        Display role label.
    """
    normalized = raw_role.strip().lower()
    if normalized in {"target", "assistant"}:
        return "assistant"
    if normalized in {"persuader", "user"}:
        return "user"
    return normalized or "unknown"


def _render_artifact_tex(artifact: PromptArtifact) -> str:
    """Render one prompt artifact into a manuscript-ready LaTeX include.

    Args:
        artifact: Prompt artifact payload.

    Returns:
        LaTeX source string for a single prompt artifact.
    """
    lines: list[str] = [
        "% Auto-generated by analysis/latex/export_paper_assets.py",
        "% Prompt body only; captions/labels belong in non-generated files.",
        "",
    ]

    for index, message in enumerate(artifact.messages, start=1):
        role = _display_message_role(str(message.get("role") or "unknown"))
        content = str(message.get("content") or "")
        lines.append(
            r"\paragraph{Message "
            + str(index)
            + r" ("
            + escape_latex_inline(role)
            + r")}"
        )
        lines.append(r"\begin{lstlisting}")
        lines.append(_sanitize_listing_content(content))
        lines.append(r"\end{lstlisting}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_artifact_tex_files(
    output_dir: Path,
    artifacts: list[PromptArtifact],
) -> list[Path]:
    """Write one TeX include file per prompt artifact.

    Args:
        output_dir: Destination directory.
        artifacts: Ordered prompt artifacts.

    Returns:
        Absolute paths to generated TeX files.
    """
    written_paths: list[Path] = []
    for artifact in artifacts:
        artifact_path = output_dir / f"{artifact.slug}.tex"
        artifact_path.write_text(_render_artifact_tex(artifact), "utf-8")
        written_paths.append(artifact_path)

    # Remove legacy generated files no longer produced by this exporter.
    for legacy_name in (
        "paper_assets.tex",
        "simulator_verbalization_rhetoric_off.tex",
    ):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    return written_paths


def _resolve_manifest_path(path: Path) -> Path:
    """Resolve figure manifest path to an absolute path.

    Args:
        path: Raw path argument.

    Returns:
        Absolute manifest path.
    """
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_figure_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load figure manifest entries from JSON.

    Args:
        manifest_path: Path to JSON manifest.

    Returns:
        List of figure entry mappings.
    """
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text("utf-8"))
    figures = payload.get("figures", [])
    if not isinstance(figures, list):
        return []
    return [entry for entry in figures if isinstance(entry, dict)]


def _load_table_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load table manifest entries from JSON.

    Args:
        manifest_path: Path to JSON manifest.

    Returns:
        List of table entry mappings.
    """
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text("utf-8"))
    tables = payload.get("tables", [])
    if not isinstance(tables, list):
        return []
    return [entry for entry in tables if isinstance(entry, dict)]


def _export_figure_assets(manifest_path: Path) -> dict[str, Any]:
    """Export figure PDFs from manifest to the paper figures directory.

    Args:
        manifest_path: Path to figure manifest JSON.

    Returns:
        Summary dictionary containing copied, missing, and deferred figures.
    """
    entries = _load_figure_manifest(manifest_path)
    destination_root = DEFAULT_FIGURE_OUTPUT_DIR
    _ensure_dir(destination_root)

    copied: list[str] = []
    missing: list[str] = []
    deferred: list[str] = []
    required_missing: list[str] = []
    summary_lists = {
        "copied": copied,
        "missing": missing,
        "deferred": deferred,
        "required_missing": required_missing,
    }

    for entry in entries:
        _process_figure_entry(
            entry=entry,
            destination_root=destination_root,
            summary_lists=summary_lists,
        )

    if required_missing:
        missing_names = ", ".join(required_missing)
        raise FileNotFoundError(f"Missing required figure assets: {missing_names}")

    return {
        "destination_root": str(destination_root),
        "copied": copied,
        "missing": missing,
        "deferred": deferred,
    }


def _table_writer_map() -> dict[str, Any]:
    """Return writer callables keyed by table-writer identifier.

    Returns:
        Mapping from manifest writer id to `(output_dir, repo_root) -> Path` callables.
    """
    return {
        "proposition_samples": write_proposition_sample_table_tex,
        "debategpt_bn_samples": write_debategpt_bn_samples_table_tex,
    }


def _export_table_assets(
    *, manifest_path: Path, destination_root: Path
) -> dict[str, Any]:
    """Export table TeX assets from manifest to the generated include directory.

    Args:
        manifest_path: Path to table manifest JSON.
        destination_root: Include-directory destination root.

    Returns:
        Summary dictionary containing written, missing, and deferred tables.
    """
    entries = _load_table_manifest(manifest_path)
    _ensure_dir(destination_root)

    written: list[Path] = []
    missing: list[str] = []
    deferred: list[str] = []
    required_missing: list[str] = []
    summary_lists = {
        "written": written,
        "missing": missing,
        "deferred": deferred,
        "required_missing": required_missing,
    }
    writer_map = _table_writer_map()

    for entry in entries:
        _process_table_entry(
            entry=entry,
            destination_root=destination_root,
            writer_map=writer_map,
            summary_lists=summary_lists,
        )

    if required_missing:
        missing_names = ", ".join(required_missing)
        raise FileNotFoundError(f"Missing required table assets: {missing_names}")

    return {
        "destination_root": str(destination_root),
        "written": [str(path) for path in written],
        "written_paths": written,
        "missing": missing,
        "deferred": deferred,
    }


def _process_figure_entry(
    *,
    entry: dict[str, Any],
    destination_root: Path,
    summary_lists: dict[str, list[str]],
) -> None:
    """Copy one manifest entry if enabled and available.

    Args:
        entry: Figure manifest entry.
        destination_root: Destination directory root.
        summary_lists: Mutable list dictionary with copied/missing/deferred/
            required_missing keys.
    """
    copied = summary_lists["copied"]
    missing = summary_lists["missing"]
    deferred = summary_lists["deferred"]
    required_missing = summary_lists["required_missing"]

    figure_id = str(entry.get("id") or "unknown")
    enabled = bool(entry.get("enabled", True))
    required = bool(entry.get("required", False))
    source_pdf_raw = str(entry.get("source_pdf") or "").strip()
    destination_pdf = str(entry.get("destination_pdf") or "").strip()

    if not enabled:
        deferred.append(figure_id)
        return
    if not source_pdf_raw or not destination_pdf or not source_pdf_raw.endswith(".pdf"):
        missing.append(figure_id)
        if required:
            required_missing.append(figure_id)
        return

    source_path = Path(source_pdf_raw)
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    destination_path = destination_root / destination_pdf

    if not source_path.exists():
        missing.append(f"{figure_id} ({source_path})")
        if required:
            required_missing.append(figure_id)
        return

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    copied.append(str(destination_path))


def _process_table_entry(
    *,
    entry: dict[str, Any],
    destination_root: Path,
    writer_map: dict[str, Any],
    summary_lists: dict[str, list[Any]],
) -> None:
    """Render one manifest table entry if enabled and configured.

    Args:
        entry: Table manifest entry.
        destination_root: Destination include directory root.
        writer_map: Mapping from writer id to table writer callables.
        summary_lists: Mutable list dictionary with written/missing/deferred/
            required_missing keys.
    """
    table_id = str(entry.get("id") or "unknown")
    enabled = bool(entry.get("enabled", True))
    required = bool(entry.get("required", False))
    writer_key = str(entry.get("writer") or "").strip()
    destination_tex = str(entry.get("destination_tex") or "").strip()

    if not enabled:
        summary_lists["deferred"].append(table_id)
        return

    if not writer_key or not destination_tex or not destination_tex.endswith(".tex"):
        summary_lists["missing"].append(table_id)
        if required:
            summary_lists["required_missing"].append(table_id)
        return

    writer = writer_map.get(writer_key)
    if writer is None:
        summary_lists["missing"].append(f"{table_id} (unknown writer: {writer_key})")
        if required:
            summary_lists["required_missing"].append(table_id)
        return

    generated_path = writer(destination_root, REPO_ROOT)
    destination_path = destination_root / destination_tex
    if generated_path != destination_path:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_path, destination_path)
    summary_lists["written"].append(destination_path)


def _build_sample_distribution() -> list[dict[str, Any]]:
    """Build a small deterministic joint distribution for prompt rendering.

    Returns:
        List of state/probability rows over ``Belief_1``, ``Belief_2``, and
        ``Target``.
    """
    rows: list[tuple[dict[str, bool], float]] = []
    for belief_1, belief_2, target in itertools.product([True, False], repeat=3):
        weight = 1.0
        if belief_1 == target:
            weight *= 1.9
        if belief_2 == target:
            weight *= 0.55
        rows.append(
            (
                {
                    "Belief_1": belief_1,
                    "Belief_2": belief_2,
                    "Target": target,
                },
                weight,
            )
        )
    total_weight = sum(weight for _, weight in rows)
    return [
        {"state": state, "probability": float(weight / total_weight)}
        for state, weight in rows
    ]


def _build_sample_bn() -> BayesianNetwork:
    """Create a sample Bayesian network used to render simulator prompts.

    Returns:
        Sample BayesianNetwork object with deterministic distribution.
    """
    payload = {
        "target_proposition": SAMPLE_PROPOSITION,
        "belief_nodes": [
            SAMPLE_BELIEF_1,
            SAMPLE_BELIEF_2,
        ],
        "joint_distribution": _build_sample_distribution(),
    }
    return BayesianNetwork(**payload)


def _build_sample_conversation() -> list[dict[str, str]]:
    """Create a sample persuasion conversation snippet.

    Returns:
        Conversation history in ``role``/``content`` format.
    """
    return [
        {
            "role": "persuader",
            "content": "[[PERSUADER_TURN_1_PLACEHOLDER]]",
        },
        {
            "role": "target",
            "content": "[[TARGET_TURN_1_PLACEHOLDER]]",
        },
        {
            "role": "persuader",
            "content": "[[PERSUADER_TURN_2_PLACEHOLDER]]",
        },
    ]


def _build_sample_latest_atoms() -> list[MessageAtom]:
    """Create sample atomizer output for verbalizer prompt rendering.

    Returns:
        List of MessageAtom objects.
    """
    return [
        MessageAtom(
            text_span="[[ARGUMENT_ATOM_TEXT_SPAN_PLACEHOLDER]]",
            p_support=0.82,
            belief_targets=[
                BeliefRelevance(belief_id="Belief_1", relevance=0.9),
                BeliefRelevance(belief_id="Target", relevance=0.7),
            ],
            edge_targets=[
                BeliefEdge(source="Belief_1", target="Target", relevance=0.6),
            ],
            rhetorical_modes=RhetoricalModes(logos=0.85, ethos=0.2, pathos=0.25),
        )
    ]


def collect_prompt_artifacts() -> list[PromptArtifact]:
    """Collect all methods-relevant prompt artifacts.

    Returns:
        Ordered list of prompt artifacts for export.
    """
    artifacts: list[PromptArtifact] = []

    generic_human_condition = Condition(
        roles=Roles(human_persuader=True, human_target=True),
        factual_domain=False,
        continuous_measure=ContinuousMeasure.SERIAL_QUESTIONS,
        use_audio=False,
        show_transcript=False,
        control_dialogue=False,
        participant_proposition=False,
        turn_limit=10,
        minimum_turns=2,
    )
    generic_human_round = Round(
        condition=generic_human_condition,
        proposition=SAMPLE_PROPOSITION,
        target_initial_belief=0.40,
        target_final_belief=None,
        persuader_supports_proposition=True,
        messages=[],
    )
    artifacts.append(
        PromptArtifact(
            slug="generic_human_persuader_prompt",
            title="Generic Human Persuader On-Screen Prompt",
            methods_reference="Methods / Conditions",
            source_paths=[
                "src/experiment/condition.py",
                "src/experiment/round.py",
                "src/api/api.py",
            ],
            model=None,
            messages=[
                {
                    "role": "system",
                    "content": generic_human_round.prompt(
                        is_target=False,
                        include_instructions=True,
                        is_human=True,
                        during_round=True,
                    ),
                },
            ],
        )
    )
    artifacts.append(
        PromptArtifact(
            slug="generic_human_target_prompt",
            title="Generic Human Target On-Screen Prompt",
            methods_reference="Methods / Conditions",
            source_paths=[
                "src/experiment/condition.py",
                "src/experiment/round.py",
                "src/api/api.py",
            ],
            model=None,
            messages=[
                {
                    "role": "system",
                    "content": generic_human_round.prompt(
                        is_target=True,
                        include_instructions=True,
                        is_human=True,
                        during_round=True,
                    ),
                },
            ],
        )
    )
    artifacts.append(
        PromptArtifact(
            slug="llm_persuader_addendum",
            title="LLM Persuader System Addendum",
            methods_reference="Methods / Conditions",
            source_paths=["src/experiment/round.py"],
            model=None,
            messages=[
                {
                    "role": "system",
                    "content": LLM_PERSUADER_NO_HEDGING,
                },
            ],
        )
    )
    artifacts.append(
        PromptArtifact(
            slug="llm_output_format_addendum",
            title="LLM Output-Format Addendum",
            methods_reference="Methods / Conditions",
            source_paths=["src/experiment/round.py"],
            model=None,
            messages=[
                {
                    "role": "system",
                    "content": LLM_HUMAN_LIKE_PROMPT_TEMPLATE.format(
                        max_message_chars=generic_human_condition.max_message_chars,
                        max_audio_seconds=generic_human_condition.max_audio_seconds,
                    ),
                },
            ],
        )
    )

    participant_messages = [
        {"role": "system", "content": PARTICIPANT_PROPOSITION_PROMPT},
        {
            "role": "user",
            "content": "[[PARTICIPANT_DECISION_TEXT_PLACEHOLDER]]",
        },
    ]
    artifacts.append(
        PromptArtifact(
            slug="participant_proposition_rephrase",
            title="Participant Proposition Validation and Rephrase Prompt",
            methods_reference="Methods / Propositions",
            source_paths=["src/experiment/endpoints.py"],
            model=PARTICIPANT_PROPOSITION_MODEL,
            messages=participant_messages,
        )
    )

    annotation_turns = [
        {
            "speaker": "persuader",
            "text": "[[ANNOTATION_DIALOGUE_PERSUADER_TURN_1_PLACEHOLDER]]",
        },
        {
            "speaker": "target",
            "text": "[[ANNOTATION_DIALOGUE_TARGET_TURN_1_PLACEHOLDER]]",
        },
        {
            "speaker": "persuader",
            "text": "[[ANNOTATION_DIALOGUE_PERSUADER_TURN_2_PLACEHOLDER]]",
        },
    ]
    artifacts.append(
        PromptArtifact(
            slug="rhetoric_annotation",
            title="Rhetoric Annotation Prompt (Logos, Pathos, Ethos)",
            methods_reference="Methods / Persuasive Mechanisms",
            source_paths=["src/annotation/prompt.py", "src/annotation/runner.py"],
            model=ANNOTATION_DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": build_annotation_system_prompt(),
                },
                {
                    "role": "user",
                    "content": format_dialogue_for_prompt(annotation_turns, 2),
                },
            ],
        )
    )

    belief_graph_messages = prepare_belief_graph_messages(
        SAMPLE_PROPOSITION,
        min_beliefs=4,
        max_beliefs=4,
    )
    artifacts.append(
        PromptArtifact(
            slug="bn_belief_graph_generation",
            title="Bayesian Network Belief-Graph Generation Prompt",
            methods_reference="Methods / Target Simulator / BN Construction",
            source_paths=["src/simulation/scripts/generate_belief_graphs.py"],
            model=DEFAULT_BN_GRAPH_MODEL,
            messages=belief_graph_messages,
        )
    )

    joint_messages, _ = build_joint_probability_messages(
        SAMPLE_PROPOSITION,
        [
            SAMPLE_BELIEF_1,
            SAMPLE_BELIEF_2,
        ],
    )
    artifacts.append(
        PromptArtifact(
            slug="bn_joint_distribution_forced_completion",
            title="Bayesian Network Joint-Distribution Scoring Prompt",
            methods_reference="Methods / Target Simulator / BN Construction",
            source_paths=["src/simulation/scripts/compute_joint_probabilities.py"],
            model=DEFAULT_JOINT_MODEL,
            messages=joint_messages,
        )
    )

    sample_bn = _build_sample_bn()
    sample_conversation = _build_sample_conversation()
    sample_atoms = _build_sample_latest_atoms()

    simulated_target = SimulatedTarget(
        bn=sample_bn,
        llm_model=DEFAULT_ATOMIZER_MODEL,
        persona=TargetPersona.BALANCED,
        use_rhetorical_dimensions=True,
    )
    simulated_target.round_goal_supports_proposition = True

    artifacts.append(
        PromptArtifact(
            slug="simulator_atomization",
            title="Simulator Atomization Prompt",
            methods_reference="Methods / Full Bayesian Simulator / LLM Atomization",
            source_paths=["src/simulation/target.py"],
            model=DEFAULT_ATOMIZER_MODEL,
            messages=simulated_target.build_atomization_messages(
                sample_conversation,
                round_goal_supports_proposition=True,
            ),
        )
    )

    artifacts.append(
        PromptArtifact(
            slug="simulator_verbalization_rhetoric_on",
            title="Simulator Verbalization Prompt (Rhetoric Enabled)",
            methods_reference="Methods / Full Bayesian Simulator / LLM Verbalization",
            source_paths=["src/simulation/target.py"],
            model=DEFAULT_ATOMIZER_MODEL,
            messages=simulated_target.build_response_messages(
                current_belief=simulated_target.get_belief_state(0),
                conversation_history=sample_conversation,
                latest_atoms=sample_atoms,
            ),
        )
    )

    artifacts.append(
        PromptArtifact(
            slug="llm_target_turn_prompt",
            title="Unstructured LLM Target Prompt",
            methods_reference="Methods / Target Simulator / Baselines",
            source_paths=["src/rl/sim_target_env.py"],
            model=DEFAULT_ATOMIZER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": LLM_TARGET_TURN_JSON_PROMPT.format(
                        proposition=SAMPLE_PROPOSITION
                    ),
                }
            ],
        )
    )

    artifacts.append(
        PromptArtifact(
            slug="llm_target_turn_prompt_with_nodes",
            title="Structure-Conditioned LLM Target Prompt",
            methods_reference="Methods / Target Simulator / Baselines",
            source_paths=["src/rl/sim_target_env.py"],
            model=DEFAULT_ATOMIZER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": LLM_TARGET_TURN_JSON_PROMPT_WITH_NODES.format(
                        node_ids="Belief_1, Belief_2",
                        proposition=SAMPLE_PROPOSITION,
                        statements=(
                            f"- Belief_1: {SAMPLE_BELIEF_1}\n"
                            f"- Belief_2: {SAMPLE_BELIEF_2}"
                        ),
                    ),
                }
            ],
        )
    )

    judge_condition = Condition(
        roles=Roles(llm_persuader="gpt-5-2025-08-07", human_target=True),
        continuous_measure=ContinuousMeasure.SERIAL_QUESTIONS,
        factual_domain=False,
    )
    judge_round = JudgeRound(
        proposition=SAMPLE_PROPOSITION,
        updates=(0.08, -0.02, 0.05, 0.01),
        condition=judge_condition,
        target_initial_belief=0.42,
        serial_questions=(0.50, 0.48, 0.53, 0.54),
        messages=(
            (
                "persuader",
                "[[JUDGE_PERSUADER_TURN_1_PLACEHOLDER]]",
            ),
            (
                "target",
                "[[JUDGE_TARGET_TURN_1_PLACEHOLDER]]",
            ),
            (
                "persuader",
                "[[JUDGE_PERSUADER_TURN_2_PLACEHOLDER]]",
            ),
            (
                "target",
                "[[JUDGE_TARGET_TURN_2_PLACEHOLDER]]",
            ),
        ),
    )
    artifacts.append(
        PromptArtifact(
            slug="llm_judge_human_likeness",
            title="LLM-as-a-Judge Human-Likeness Prompt",
            methods_reference="Methods / Analyses / Human Likeness via LLM-as-a-Judge",
            source_paths=["analysis/simulator_llm_judge.py"],
            model=DEFAULT_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": render_round_prompt(
                        judge_round,
                        max_messages=20,
                        max_message_chars=220,
                    ),
                },
            ],
        )
    )

    return artifacts


def export_paper_assets(
    output_dir: Path,
) -> list[Path]:
    """Export all current paper-facing prompt artifacts to per-artifact TeX files.

    Args:
        output_dir: Destination directory for generated files.

    Returns:
        Absolute paths to generated TeX files.
    """
    normalized_output_dir = _normalize_output_dir(output_dir)
    _ensure_dir(normalized_output_dir)
    artifacts = collect_prompt_artifacts()
    written_prompt_paths = _write_artifact_tex_files(
        normalized_output_dir, artifacts=artifacts
    )
    table_manifest_path = _resolve_manifest_path(DEFAULT_FIGURE_MANIFEST_PATH)
    table_summary = _export_table_assets(
        manifest_path=table_manifest_path,
        destination_root=normalized_output_dir,
    )
    figure_manifest_path = _resolve_manifest_path(DEFAULT_FIGURE_MANIFEST_PATH)
    figure_summary = _export_figure_assets(figure_manifest_path)

    print(f"Table export destination: {table_summary['destination_root']}")
    print(f"Tables written: {len(table_summary['written'])}")
    if table_summary["missing"]:
        print(f"Tables missing: {len(table_summary['missing'])}")
    if table_summary["deferred"]:
        print(f"Tables deferred: {len(table_summary['deferred'])}")
    print(f"Figure export destination: {figure_summary['destination_root']}")
    print(f"Figures copied: {len(figure_summary['copied'])}")
    if figure_summary["missing"]:
        print(f"Figures missing: {len(figure_summary['missing'])}")
    if figure_summary["deferred"]:
        print(f"Figures deferred: {len(figure_summary['deferred'])}")
    return [*written_prompt_paths, *table_summary["written_paths"]]


def main() -> None:
    """Run prompt export and print a short summary."""
    args = parse_args()
    written_paths = export_paper_assets(args.output_dir)
    print(f"Exported {len(written_paths)} prompt artifacts.")
    print(f"Output directory: {_normalize_output_dir(args.output_dir)}")


if __name__ == "__main__":
    main()
