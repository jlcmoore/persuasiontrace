"""
Estimate target-only and structure-aware persuasion difficulty for simulator BNs.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from simulation.io import read_jsonl_records
from simulation.persuasion_difficulty import (
    MAX_DIFFICULTY_CAP,
    DifficultyEvalConfig,
    evaluate_record,
    summarize_rows,
)
from simulation.target_bins import TARGET_BELIEF_BIN_RANGES


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for simulator persuasion-difficulty analysis.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute target-only and structure-aware persuasion difficulty for "
            "Bayesian-network proposition sets."
        )
    )
    parser.add_argument(
        "--bn-jsonl",
        type=str,
        default=None,
        help="Path to fitted BN JSONL (records with id + bayesian_network).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "RL baseline config YAML. Uses data.proposition_files_by_source and "
            "supports recursive extends."
        ),
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated proposition sources to include when using --config.",
    )
    parser.add_argument(
        "--init-mode",
        choices=("all", "prior", "bin_centers", "bin_samples"),
        default="all",
        help="Initialization set to evaluate.",
    )
    parser.add_argument(
        "--init-bins",
        type=str,
        default="very_low,low,mid,high,very_high",
        help="Comma-separated target-belief bins used for bin modes.",
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=3,
        help="Number of random initializations per bin for init-mode=bin_samples.",
    )
    parser.add_argument(
        "--goal-delta",
        type=float,
        default=0.1,
        help="Absolute target-belief move size toward the opposite side.",
    )
    parser.add_argument(
        "--max-propositions",
        type=int,
        default=None,
        help="Optional cap on number of propositions per source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for bin sample initializations.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="analysis/data/simulator_persuasion_difficulty.csv",
        help="Per-proposition/per-initialization output CSV.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        default="analysis/data/simulator_persuasion_difficulty_summary.csv",
        help="Grouped summary output CSV.",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="analysis/figures/simulator_persuasion_difficulty",
        help=(
            "Output prefix for plots. Writes "
            "<prefix>_structure_scatter.png and <prefix>_structure_summary.png."
        ),
    )
    parser.add_argument(
        "--plot-mode",
        choices=("scatter", "both"),
        default="scatter",
        help="Plot output mode. Default writes only the scatter plot.",
    )
    args = parser.parse_args()
    if not args.bn_jsonl and not args.config:
        parser.error("Provide one of --bn-jsonl or --config.")
    return args


def _load_yaml_with_extends(
    config_path: Path,
    *,
    seen_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a YAML config with recursive `extends` support.

    Args:
        config_path: Path to the YAML config.
        seen_paths: Paths visited so far for cycle detection.

    Returns:
        Fully-resolved configuration payload.
    """
    resolved_path = config_path.resolve()
    seen = set() if seen_paths is None else set(seen_paths)
    if resolved_path in seen:
        raise ValueError(f"Cyclic config inheritance detected at: {resolved_path}")
    seen.add(resolved_path)

    payload = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=False) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config must deserialize to a dictionary.")

    extends_value = payload.pop("extends", None)
    if extends_value is None:
        return payload
    if not isinstance(extends_value, str):
        raise ValueError("`extends` must be a string path.")

    base_path = Path(extends_value)
    if not base_path.is_absolute():
        base_path = resolved_path.parent / base_path
    base_payload = _load_yaml_with_extends(base_path, seen_paths=seen)
    merged = OmegaConf.merge(base_payload, payload)
    merged_payload = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(merged_payload, dict):
        raise ValueError("Merged config must be a dictionary.")
    return merged_payload


def parse_sources(value: str | None) -> list[str] | None:
    """Parse a comma-separated source list.

    Args:
        value: Raw CLI string.

    Returns:
        Parsed source list, or ``None`` when not provided.
    """
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def parse_bins(value: str) -> list[str]:
    """Parse and validate initialization bins.

    Args:
        value: Comma-separated bin names.

    Returns:
        Parsed and validated bin list.
    """
    bins = [item.strip() for item in value.split(",") if item.strip()]
    if not bins:
        raise ValueError("At least one init bin is required.")
    unknown = [item for item in bins if item not in TARGET_BELIEF_BIN_RANGES]
    if unknown:
        raise ValueError(f"Unknown init bins: {unknown}")
    return bins


def resolve_input_files(
    *,
    bn_jsonl: str | None,
    config_path: str | None,
    selected_sources: list[str] | None,
) -> list[tuple[str, Path]]:
    """Resolve proposition-source file paths from either JSONL or config.

    Args:
        bn_jsonl: Optional direct BN JSONL path.
        config_path: Optional RL config path.
        selected_sources: Optional subset of sources for config mode.

    Returns:
        List of (source, path) tuples.
    """
    if bn_jsonl:
        path = Path(bn_jsonl)
        return [(_infer_source_label_from_path(path), path)]

    assert config_path is not None
    payload = _load_yaml_with_extends(Path(config_path))
    data_section = payload.get("data", {})
    if not isinstance(data_section, dict):
        raise ValueError("Config missing `data` section.")
    files_by_source = data_section.get("proposition_files_by_source", {})
    if not isinstance(files_by_source, dict):
        raise ValueError("Config data.proposition_files_by_source must be a mapping.")

    out: list[tuple[str, Path]] = []
    for source, path_raw in files_by_source.items():
        if selected_sources and source not in selected_sources:
            continue
        if not isinstance(path_raw, str):
            continue
        out.append((str(source), Path(path_raw)))

    if selected_sources and not out:
        raise ValueError(
            f"No proposition file entries matched requested sources={selected_sources}."
        )
    if not out:
        raise ValueError("No proposition files resolved from config.")
    return out


def _infer_source_label_from_path(path: Path) -> str:
    """Infer a friendly source label from a BN JSONL path.

    Args:
        path: Input proposition file path.

    Returns:
        Inferred source label.
    """
    stem = path.stem
    prefix = "fitted_bayesian_networks_"
    if stem.startswith(prefix):
        inferred = stem[len(prefix) :]
        if inferred:
            return inferred
    return stem or "direct"


def load_bn_records(
    source_to_path: list[tuple[str, Path]],
    max_propositions: int | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Load proposition records with Bayesian networks.

    Args:
        source_to_path: Source/path pairs to load.
        max_propositions: Optional per-source cap.

    Returns:
        List of (source, record) tuples.
    """
    loaded: list[tuple[str, dict[str, Any]]] = []
    for source, file_path in source_to_path:
        records = read_jsonl_records(file_path=file_path)
        valid: list[dict[str, Any]] = []
        for record in records:
            proposition_id = record.get("id")
            bn_payload = record.get("bayesian_network")
            if isinstance(proposition_id, str) and isinstance(bn_payload, dict):
                valid.append(record)
        if max_propositions is not None:
            valid = valid[:max_propositions]
        loaded.extend((source, record) for record in valid)
    return loaded


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write dictionaries to CSV.

    Args:
        path: Output path.
        rows: Rows to write.
        columns: CSV column ordering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def init_label_sort_key(label: str) -> tuple[int, int, str]:
    """Return a stable sort key for initialization labels.

    Args:
        label: Initialization label string.

    Returns:
        Sort key tuple for presentation order.
    """
    if label == "prior":
        return (0, 0, label)
    if label.startswith("bin_center:"):
        bin_name = label.split(":", maxsplit=1)[1]
        bin_order = list(TARGET_BELIEF_BIN_RANGES).index(bin_name)
        return (1, bin_order, label)
    if label.startswith("bin_sample:"):
        parts = label.split(":")
        if len(parts) == 3 and parts[1] in TARGET_BELIEF_BIN_RANGES:
            bin_order = list(TARGET_BELIEF_BIN_RANGES).index(parts[1])
            return (2, bin_order, label)
    return (3, 0, label)


def source_color_map(sources: list[str]) -> dict[str, str]:
    """Build a deterministic color map for sources.

    Args:
        sources: Ordered source labels.

    Returns:
        Mapping from source label to color hex string.
    """
    base_palette = [
        "#1B9E77",
        "#D95F02",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#E6AB02",
        "#A6761D",
        "#666666",
    ]
    return {
        source: base_palette[index % len(base_palette)]
        for index, source in enumerate(sorted(set(sources)))
    }


def plot_structure_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot structure-aware difficulty versus initialized target belief.

    Args:
        rows: Per-initialization metric rows.
        output_path: Destination PNG path.
    """
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = [str(row["source"]) for row in rows]
    color_by_source = source_color_map(sources)

    fig, axis = plt.subplots(figsize=(8, 5))
    for source in sorted(set(sources)):
        source_rows = [row for row in rows if str(row["source"]) == source]
        uncapped_rows = [
            row for row in source_rows if not bool(row["structure_aware_capped"])
        ]
        capped_rows = [
            row for row in source_rows if bool(row["structure_aware_capped"])
        ]
        x_values = [float(row["init_target_belief"]) for row in uncapped_rows]
        y_values = [float(row["structure_aware_difficulty"]) for row in uncapped_rows]
        axis.scatter(
            x_values,
            y_values,
            alpha=0.65,
            s=26,
            color=color_by_source[source],
            label=source,
        )
        if capped_rows:
            x_capped = [float(row["init_target_belief"]) for row in capped_rows]
            y_capped = [MAX_DIFFICULTY_CAP for _ in capped_rows]
            axis.scatter(
                x_capped,
                y_capped,
                alpha=0.9,
                s=42,
                marker="x",
                color=color_by_source[source],
                label=f"{source} (capped)",
            )

    axis.set_xlabel("Initialized Target Belief")
    axis.set_ylabel("Structure-Aware Difficulty")
    axis.set_yscale("log")
    axis.set_title("Structure-Aware Persuasion Difficulty vs Initial Belief")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_structure_summary(
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot mean structure-aware difficulty by initialization label.

    Args:
        summary_rows: Grouped summary rows.
        output_path: Destination PNG path.
    """
    if not summary_rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_labels = sorted(
        {str(row["init_label"]) for row in summary_rows},
        key=init_label_sort_key,
    )
    label_to_x = {label: index for index, label in enumerate(ordered_labels)}
    sources = sorted({str(row["source"]) for row in summary_rows})
    color_by_source = source_color_map(sources)

    fig, axis = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(sources))
    for index, source in enumerate(sources):
        source_rows = [row for row in summary_rows if str(row["source"]) == source]
        row_by_label = {str(row["init_label"]): row for row in source_rows}
        x_values = [
            label_to_x[label] + (index - (len(sources) - 1) / 2.0) * width
            for label in ordered_labels
        ]
        y_values = [
            (
                float(row_by_label[label]["mean_structure_aware_difficulty"])
                if label in row_by_label
                else 0.0
            )
            for label in ordered_labels
        ]
        axis.bar(
            x_values,
            y_values,
            width=width * 0.95,
            color=color_by_source[source],
            alpha=0.85,
            label=source,
        )

    axis.set_xticks(list(label_to_x.values()))
    axis.set_xticklabels(ordered_labels, rotation=20, ha="right")
    axis.set_ylabel("Mean Structure-Aware Difficulty")
    axis.set_yscale("log")
    axis.set_title("Mean Structure-Aware Difficulty by Initialization")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_metric_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build detailed and summary metric rows from CLI args.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (detail_rows, summary_rows).
    """
    rng = random.Random(args.seed)
    init_bins = parse_bins(args.init_bins)
    sources = parse_sources(args.sources)
    source_to_path = resolve_input_files(
        bn_jsonl=args.bn_jsonl,
        config_path=args.config,
        selected_sources=sources,
    )
    loaded_records = load_bn_records(
        source_to_path=source_to_path,
        max_propositions=args.max_propositions,
    )
    eval_config = DifficultyEvalConfig(
        init_mode=args.init_mode,
        bins=init_bins,
        samples_per_bin=args.samples_per_bin,
        goal_delta=float(args.goal_delta),
    )
    rows: list[dict[str, Any]] = []
    for source, record in loaded_records:
        rows.extend(
            evaluate_record(
                source=source,
                record=record,
                eval_config=eval_config,
                rng=rng,
            )
        )
    return rows, summarize_rows(rows)


def write_outputs(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    """Write CSV and plot outputs for computed metrics.

    Args:
        args: Parsed CLI arguments.
        rows: Per-initialization metric rows.
        summary_rows: Grouped summary rows.
    """
    output_columns = [
        "source",
        "proposition_id",
        "init_label",
        "init_target_belief",
        "goal_target_belief",
        "required_abs_delta",
        "target_only_difficulty_logit_delta",
        "target_only_difficulty_local_per_unit",
        "structure_aware_difficulty",
        "structure_aware_best_directional_slope",
        "structure_aware_best_node",
        "structure_aware_capped",
        "structure_aware_infeasible_direction",
    ]
    write_csv(Path(args.output_csv), rows, output_columns)

    summary_columns = [
        "source",
        "init_label",
        "n_rows",
        "mean_required_abs_delta",
        "mean_target_only_difficulty_logit_delta",
        "mean_structure_aware_difficulty",
        "mean_structure_minus_target_only",
    ]
    write_csv(Path(args.output_summary_csv), summary_rows, summary_columns)

    if args.plot_prefix:
        prefix_path = Path(args.plot_prefix)
        scatter_paths = [
            Path(f"{prefix_path}_structure_scatter.png"),
            Path(f"{prefix_path}_structure_scatter.pdf"),
        ]
        for scatter_path in scatter_paths:
            plot_structure_scatter(rows, scatter_path)
            print(f"Wrote structure-aware scatter plot to {scatter_path}")
        if args.plot_mode == "both":
            summary_paths = [
                Path(f"{prefix_path}_structure_summary.png"),
                Path(f"{prefix_path}_structure_summary.pdf"),
            ]
            for summary_path in summary_paths:
                plot_structure_summary(summary_rows, summary_path)
                print(f"Wrote structure-aware summary plot to {summary_path}")


def main() -> None:
    """Run simulator persuasion-difficulty analysis from CLI args."""
    args = parse_args()
    rows, summary_rows = build_metric_rows(args)
    write_outputs(args=args, rows=rows, summary_rows=summary_rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.output_summary_csv}")


if __name__ == "__main__":
    main()
