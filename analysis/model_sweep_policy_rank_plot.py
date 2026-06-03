"""Plot policy persuasiveness rankings by simulator panel from baseline episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .utils import resolve_repo_path

DEFAULT_EPISODES_JSONL = Path("results/rl_model_sweep_debategpt_bn/episodes.jsonl")
DEFAULT_OUTPUT_PDF = Path("analysis/figures/model_sweep_policy_rank_by_simulator.pdf")
DEFAULT_OUTPUT_CSV = Path("analysis/data/model_sweep_policy_rank_by_simulator.csv")
SIMULATOR_PANEL_ORDER = [
    "vanilla_llm_target",
    "structure_target",
    "full_simulated_target_avg",
]
SIMULATOR_LABELS = {
    "vanilla_llm_target": "Unstructured LLM Target",
    "structure_target": "Structured LLM Target",
    "full_simulated_target_avg": "BN Target",
}
BN_PERSONAS = ("logical", "emotional", "authoritarian")
POLICY_LABELS = {
    "naive": "naive",
    "openai/gpt-5.4": "gpt5.4",
    "openai/gpt-5.4-mini": "gpt5.4-mini",
    "openai/gpt-5-2025-08-07": "gpt5",
    "gpt-5-2025-08-07": "gpt5",
    "xai/grok-4.20-non-reasoning": "grok-4.20",
    "anthropic/claude-opus-4-7": "claude-opus-4.7",
    "gemini/gemini-3.1-pro-preview": "gemini-3.1-pro",
    "together_ai/Qwen/Qwen3.5-397B-A17B": "qwen3.5-397b",
}
MODEL_COLORS = {
    "naive": "#4C566A",
    "openai/gpt-5.4": "#0B84A5",
    "openai/gpt-5.4-mini": "#4F9DA6",
    "openai/gpt-5-2025-08-07": "#2E4057",
    "gpt-5-2025-08-07": "#2E4057",
    "xai/grok-4.20-non-reasoning": "#F6C85F",
    "anthropic/claude-opus-4-7": "#9FD356",
    "gemini/gemini-3.1-pro-preview": "#CA472F",
    "together_ai/Qwen/Qwen3.5-397B-A17B": "#8E6C8A",
}


@dataclass(frozen=True)
class PanelPolicyStat:
    """Summary metric for one policy within one simulator panel."""

    simulator_panel: str
    policy_model: str
    mean_terminal_delta: float
    ci95_low: float
    ci95_high: float
    n_episodes: int
    personas_averaged: int
    source_file: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot policy rankings by persuader-relative final delta with one "
            "horizontal simulator panel per subplot."
        )
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=DEFAULT_EPISODES_JSONL,
        help="Primary baseline episodes JSONL.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Path to output PDF figure.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output CSV summary.",
    )
    parser.add_argument(
        "--require-all-bn-personas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require logical/emotional/authoritarian to all be present when "
            "building BN averaged panel rows."
        ),
    )
    return parser.parse_args()


def _simulator_panel_for_episode(row: dict[str, Any]) -> str | None:
    """Map one episode row to a simulator panel key.

    Args:
        row: Episode row payload from baseline runner JSONL.

    Returns:
        Simulator panel key, or None when row should be excluded.
    """
    target_backend = str(row.get("target_backend") or "")
    if target_backend == "llm_target":
        if bool(row.get("llm_target_use_bayes_structure", False)):
            return "structure_target"
        return "vanilla_llm_target"
    if target_backend != "simulated_target":
        return None
    if bool(row.get("simulated_target_no_rhetoric", False)):
        return None
    return "full_simulated_target_avg"


def _safe_terminal_delta(row: dict[str, Any]) -> float | None:
    """Parse terminal delta when finite.

    Args:
        row: Episode row payload.

    Returns:
        Finite terminal delta, or None when missing or invalid.
    """
    raw_value = row.get("terminal_delta")
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _policy_label(policy_model: str) -> str:
    """Build concise display label for one policy model id.

    Args:
        policy_model: Full policy model id.

    Returns:
        Display label.
    """
    mapped = POLICY_LABELS.get(policy_model)
    if mapped is not None:
        return mapped
    if "/" in policy_model:
        return policy_model.rsplit("/", maxsplit=1)[-1]
    return policy_model


def _wrapped_policy_label(policy_model: str, *, width: int = 10) -> str:
    """Build a wrapped display label for x-axis model names.

    Args:
        policy_model: Full policy model id.
        width: Wrap width in characters.

    Returns:
        Wrapped label with newline separators.
    """
    base_label = _policy_label(policy_model).replace("-", "- ")
    return textwrap.fill(base_label, width=width, break_long_words=False)


def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
    """Estimate mean and normal-approximation 95 percent CI.

    Args:
        values: Numeric samples.

    Returns:
        Mean, lower CI bound, upper CI bound.
    """
    count = len(values)
    if count == 0:
        raise ValueError("Cannot compute CI for empty sample list.")
    values_arr = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values_arr))
    if count == 1:
        return mean_value, mean_value, mean_value
    std = float(np.std(values_arr, ddof=1))
    margin = 1.96 * std / math.sqrt(count)
    return mean_value, mean_value - margin, mean_value + margin


def _bn_weighted_mean_ci95(
    by_persona: dict[str, list[float]],
    *,
    require_all_bn_personas: bool,
) -> tuple[float, float, float, int, int] | None:
    """Compute equal-persona weighted mean and CI for BN panel values.

    Args:
        by_persona: Mapping persona -> episode deltas.
        require_all_bn_personas: Whether all personas must be present.

    Returns:
        Mean, lower CI, upper CI, total episodes, personas averaged. Returns
        ``None`` when no usable persona data exists.
    """
    if require_all_bn_personas and any(
        not by_persona.get(persona) for persona in BN_PERSONAS
    ):
        return None

    present_personas: list[str] = [
        persona for persona in BN_PERSONAS if by_persona.get(persona)
    ]
    if not present_personas:
        return None

    persona_weight = 1.0 / float(len(present_personas))
    weighted_mean = 0.0
    weighted_var_mean = 0.0
    total_episodes = 0
    for persona in present_personas:
        values = np.asarray(by_persona[persona], dtype=float)
        n_persona = int(values.size)
        total_episodes += n_persona
        persona_mean = float(np.mean(values))
        weighted_mean += persona_weight * persona_mean
        if n_persona > 1:
            persona_var = float(np.var(values, ddof=1))
            weighted_var_mean += (persona_weight**2) * (persona_var / n_persona)

    margin = 1.96 * math.sqrt(max(0.0, weighted_var_mean))
    return (
        weighted_mean,
        weighted_mean - margin,
        weighted_mean + margin,
        total_episodes,
        len(present_personas),
    )


def _load_episode_samples(
    *,
    episodes_jsonl: Path,
    allowed_policies: set[str] | None,
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, list[float]]],
    set[str],
]:
    """Load panel-aligned deltas from one episodes JSONL file.

    Args:
        episodes_jsonl: Episodes JSONL file path.
        allowed_policies: Optional policy allowlist. ``None`` keeps all policies.

    Returns:
        Non-BN panel values, BN persona values, and policies seen.
    """
    non_bn_deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    bn_persona_deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    policies_seen: set[str] = set()

    with episodes_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            panel = _simulator_panel_for_episode(row)
            if panel is None:
                continue
            terminal_delta = _safe_terminal_delta(row)
            if terminal_delta is None:
                continue
            policy_model = str(row.get("policy_model") or "")
            if not policy_model:
                continue
            if allowed_policies is not None and policy_model not in allowed_policies:
                continue

            policies_seen.add(policy_model)
            if panel != "full_simulated_target_avg":
                non_bn_deltas[panel][policy_model].append(terminal_delta)
                continue

            persona = str(row.get("persona") or "")
            if persona not in BN_PERSONAS:
                continue
            bn_persona_deltas[policy_model][persona].append(terminal_delta)

    return non_bn_deltas, bn_persona_deltas, policies_seen


def _build_panel_stats(
    *,
    non_bn_deltas: dict[str, dict[str, list[float]]],
    bn_persona_deltas: dict[str, dict[str, list[float]]],
    require_all_bn_personas: bool,
    source_file_for_policy: dict[str, str],
) -> dict[str, list[PanelPolicyStat]]:
    """Aggregate panel-level policy means and CIs.

    Args:
        non_bn_deltas: Non-BN panel values.
        bn_persona_deltas: BN panel persona values.
        require_all_bn_personas: Whether BN rows require all personas.
        source_file_for_policy: Mapping from policy model to source file path.

    Returns:
        Mapping from simulator panel to policy stats.
    """
    panel_stats: dict[str, list[PanelPolicyStat]] = defaultdict(list)

    for panel, by_policy in non_bn_deltas.items():
        for policy_model, values in by_policy.items():
            if not values:
                continue
            mean_value, ci_low, ci_high = _mean_ci95(values)
            panel_stats[panel].append(
                PanelPolicyStat(
                    simulator_panel=panel,
                    policy_model=policy_model,
                    mean_terminal_delta=mean_value,
                    ci95_low=ci_low,
                    ci95_high=ci_high,
                    n_episodes=len(values),
                    personas_averaged=1,
                    source_file=source_file_for_policy.get(policy_model, ""),
                )
            )

    for policy_model, by_persona in bn_persona_deltas.items():
        bn_metrics = _bn_weighted_mean_ci95(
            by_persona,
            require_all_bn_personas=require_all_bn_personas,
        )
        if bn_metrics is None:
            continue
        mean_value, ci_low, ci_high, total_episodes, total_personas = bn_metrics
        panel_stats["full_simulated_target_avg"].append(
            PanelPolicyStat(
                simulator_panel="full_simulated_target_avg",
                policy_model=policy_model,
                mean_terminal_delta=mean_value,
                ci95_low=ci_low,
                ci95_high=ci_high,
                n_episodes=total_episodes,
                personas_averaged=total_personas,
                source_file=source_file_for_policy.get(policy_model, ""),
            )
        )

    return panel_stats


def _write_summary_csv(
    *, output_csv: Path, panel_stats: dict[str, list[PanelPolicyStat]]
) -> None:
    """Write a tidy CSV of ranking stats.

    Args:
        output_csv: Destination CSV path.
        panel_stats: Aggregated panel stats.

    Returns:
        None.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "simulator_panel",
                "simulator_label",
                "rank_left_to_right",
                "policy_model",
                "policy_label",
                "mean_terminal_delta",
                "ci95_low",
                "ci95_high",
                "n_episodes",
                "personas_averaged",
                "source_file",
            ],
        )
        writer.writeheader()
        for panel in SIMULATOR_PANEL_ORDER:
            ranked = sorted(
                panel_stats.get(panel, []),
                key=lambda row: row.mean_terminal_delta,
                reverse=True,
            )
            for rank_idx, row in enumerate(ranked, start=1):
                writer.writerow(
                    {
                        "simulator_panel": panel,
                        "simulator_label": SIMULATOR_LABELS.get(panel, panel),
                        "rank_left_to_right": rank_idx,
                        "policy_model": row.policy_model,
                        "policy_label": _policy_label(row.policy_model),
                        "mean_terminal_delta": row.mean_terminal_delta,
                        "ci95_low": row.ci95_low,
                        "ci95_high": row.ci95_high,
                        "n_episodes": row.n_episodes,
                        "personas_averaged": row.personas_averaged,
                        "source_file": row.source_file,
                    }
                )


def _policy_color_map(panel_stats: dict[str, list[PanelPolicyStat]]) -> dict[str, str]:
    """Build deterministic color mapping for policy models.

    Args:
        panel_stats: Aggregated panel stats.

    Returns:
        Mapping from policy model id to color hex.
    """
    policies = sorted(
        {row.policy_model for rows in panel_stats.values() for row in rows},
        key=lambda model: _policy_label(model),
    )
    color_map: dict[str, str] = {}
    fallback_colors = plt.get_cmap("tab20").colors
    fallback_index = 0
    for policy_model in policies:
        known_color = MODEL_COLORS.get(policy_model)
        if known_color is not None:
            color_map[policy_model] = known_color
            continue
        color_map[policy_model] = fallback_colors[fallback_index % len(fallback_colors)]
        fallback_index += 1
    return color_map


def _plot_ranked_panels(
    *, output_pdf: Path, panel_stats: dict[str, list[PanelPolicyStat]]
) -> None:
    """Render ranked policy bars with 95 percent CIs by simulator panel.

    Args:
        output_pdf: Destination figure path.
        panel_stats: Aggregated panel stats.

    Returns:
        None.
    """
    available_panels = [
        panel for panel in SIMULATOR_PANEL_ORDER if panel_stats.get(panel)
    ]
    if not available_panels:
        raise ValueError("No simulator panels with data found in episodes input.")

    max_policies = max(len(panel_stats[panel]) for panel in available_panels)
    fig_width = max(11.0, 4.8 * len(available_panels))
    fig_height = max(1.6, 0.5 * (0.58 * max_policies + 1.8))
    figure, axes = plt.subplots(
        1,
        len(available_panels),
        figsize=(fig_width, fig_height),
        sharey=True,
    )
    if len(available_panels) == 1:
        axes = [axes]

    rank_positions = np.arange(1, max_policies + 1, dtype=float)
    rank_labels = [str(rank) for rank in range(1, max_policies + 1)]
    colors = _policy_color_map(panel_stats)
    for panel_index, (axis, panel) in enumerate(zip(axes, available_panels)):
        ranked = sorted(
            panel_stats.get(panel, []),
            key=lambda row: row.mean_terminal_delta,
            reverse=True,
        )
        y_values = np.arange(1, len(ranked) + 1, dtype=float)
        means = np.asarray(
            [row.mean_terminal_delta for row in ranked],
            dtype=float,
        )
        ci_lows = np.asarray([row.ci95_low for row in ranked], dtype=float)
        ci_highs = np.asarray([row.ci95_high for row in ranked], dtype=float)
        low_err = means - ci_lows
        high_err = ci_highs - means
        labels = [_wrapped_policy_label(row.policy_model) for row in ranked]
        bar_colors = [colors[row.policy_model] for row in ranked]

        axis.barh(
            y_values,
            means,
            xerr=np.vstack([low_err, high_err]),
            color=bar_colors,
            edgecolor="#2F2F2F",
            linewidth=0.7,
            alpha=0.95,
            error_kw={"elinewidth": 1.0, "ecolor": "#222222", "capsize": 2},
        )
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.8)
        axis.set_yticks(rank_positions)
        if panel_index == 0:
            axis.set_yticklabels(rank_labels)
            axis.tick_params(axis="y", labelsize=9, length=3, labelleft=True)
        else:
            axis.tick_params(axis="y", labelsize=9, length=3, labelleft=False)
        axis.set_title(SIMULATOR_LABELS.get(panel, panel), fontsize=11)
        axis.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
        axis.set_ylim(0.5, max_policies + 0.5)
        axis.invert_yaxis()

    legend_models = sorted(
        {row.policy_model for rows in panel_stats.values() for row in rows},
        key=_policy_label,
    )
    legend_handles = [
        Patch(facecolor=colors[model], edgecolor="#2F2F2F", label=_policy_label(model))
        for model in legend_models
    ]
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=6,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 1.03),
        )

    figure.supylabel("Rank", fontsize=10, x=0.01)
    figure.supxlabel(r"Persuasion delta ($\rightarrow$)", fontsize=10, y=0.035)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=0.4)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, format="pdf")
    plt.close(figure)


def main() -> None:
    """Run simulator-panel ranking aggregation and plotting pipeline."""
    args = parse_args()
    reference_file = Path(__file__).resolve()
    episodes_jsonl = resolve_repo_path(
        args.episodes_jsonl, reference_file=reference_file
    )
    output_pdf = resolve_repo_path(args.output_pdf, reference_file=reference_file)
    output_csv = resolve_repo_path(args.output_csv, reference_file=reference_file)

    non_bn_deltas, bn_persona_deltas, policies_seen = _load_episode_samples(
        episodes_jsonl=episodes_jsonl,
        allowed_policies=None,
    )
    source_file_for_policy: dict[str, str] = {
        policy_model: str(episodes_jsonl) for policy_model in policies_seen
    }

    panel_stats = _build_panel_stats(
        non_bn_deltas=non_bn_deltas,
        bn_persona_deltas=bn_persona_deltas,
        require_all_bn_personas=bool(args.require_all_bn_personas),
        source_file_for_policy=source_file_for_policy,
    )
    _write_summary_csv(output_csv=output_csv, panel_stats=panel_stats)
    _plot_ranked_panels(output_pdf=output_pdf, panel_stats=panel_stats)

    print(f"wrote_csv: {output_csv}")
    print(f"wrote_pdf: {output_pdf}")
    print(f"primary_episodes: {episodes_jsonl}")


if __name__ == "__main__":
    main()
