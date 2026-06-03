"""Summarize persuasion delta across user-defined condition buckets.

This script is intended for paper-facing Results figures where each bar is a
named condition bucket (for example: control, single-turn, multi-turn).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import FILTER_SPECS, condition_matches_filters

from .data_loading import load_dataframe
from .stats import bootstrap_mean_ci

DEFAULT_OUTPUT_CSV = Path("analysis/data/persuasiveness_bucket_summary.csv")
DEFAULT_OUTPUT_PDF = Path("analysis/figures/persuasiveness_bucket_summary.pdf")
DEFAULT_BOOTSTRAP_REPLICATES = 5000
BUCKET_LABEL_MAP = {
    "control": "Control",
    "standard_text": "Standard\nProp.",
    "personal_text": "Personal\nProp.",
    "audio": "Audio",
}


@dataclass(frozen=True)
class BucketSpec:
    """Parsed bucket specification.

    Attributes:
        label: Bucket display name.
        filters: Condition filter mapping.
    """

    label: str
    filters: dict[str, object]


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute persuasion-delta summaries for named condition buckets "
            "and render one comparison figure."
        )
    )
    add_min_date_arg(parser)
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help=(
            "Bucket definition in the form "
            "'label:key=value,key=value'. "
            "Keys are Condition filter names from experiment.condition_filters."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for bucket summaries.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path for the summary figure.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Bootstrap replicates for mean 95%% CIs.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop buckets with zero matching rounds.",
    )
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    """Resolve a path relative to repository root.

    Args:
        path: Path from CLI args.

    Returns:
        Absolute output path.
    """
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).resolve()


def _filter_parser_map() -> dict[str, object]:
    """Build mapping from filter key to parser callable.

    Returns:
        Mapping from normalized filter key to parser function.
    """
    parser_map: dict[str, object] = {}
    for spec in FILTER_SPECS:
        key = str(spec["name"])
        parser_map[key] = spec["parser"]
    return parser_map


def _parse_bucket_definition(
    raw_spec: str,
    *,
    parser_map: dict[str, object],
) -> BucketSpec:
    """Parse one --bucket definition string.

    Args:
        raw_spec: Raw CLI bucket string.
        parser_map: Condition filter parser map.

    Returns:
        Parsed bucket spec.

    Raises:
        ValueError: If syntax or filter keys are invalid.
    """
    label_raw, separator, filters_raw = raw_spec.partition(":")
    label = label_raw.strip()
    if not separator or not label:
        raise ValueError(
            f"Invalid --bucket '{raw_spec}'. Expected 'label:key=value,...'."
        )
    filters: dict[str, object] = {}
    if not filters_raw.strip():
        return BucketSpec(label=label, filters=filters)

    for pair in filters_raw.split(","):
        key, parsed_value = _parse_bucket_filter_pair(
            pair,
            raw_spec=raw_spec,
            parser_map=parser_map,
        )
        filters[key] = parsed_value
    return BucketSpec(label=label, filters=filters)


def _parse_bucket_filter_pair(
    pair: str,
    *,
    raw_spec: str,
    parser_map: dict[str, object],
) -> tuple[str, object]:
    """
    Parse one key=value pair from a bucket definition.

    Args:
        pair: Raw key=value text.
        raw_spec: Full bucket spec for error messages.
        parser_map: Condition filter parser map.

    Returns:
        Tuple of normalized key and parsed value.

    Raises:
        ValueError: If the pair is invalid or fails parser validation.
    """
    key_raw, eq, value_raw = pair.partition("=")
    key = key_raw.strip().replace("-", "_")
    value = value_raw.strip()
    if not eq or not key:
        raise ValueError(f"Invalid filter pair '{pair}' in --bucket '{raw_spec}'.")
    parser = parser_map.get(key)
    if parser is None:
        known = ", ".join(sorted(parser_map))
        raise ValueError(
            f"Unknown filter key '{key}' in --bucket '{raw_spec}'. "
            f"Known keys: {known}"
        )
    try:
        parsed_value = parser(value)
    except argparse.ArgumentTypeError as error:
        raise ValueError(
            f"Invalid value '{value}' for key '{key}' in --bucket '{raw_spec}'."
        ) from error
    return key, parsed_value


def _parse_buckets(raw_buckets: list[str]) -> list[BucketSpec]:
    """Parse all CLI bucket definitions.

    Args:
        raw_buckets: Raw --bucket values from CLI.

    Returns:
        Parsed bucket specs.

    Raises:
        ValueError: If no buckets provided or parsing fails.
    """
    if not raw_buckets:
        raise ValueError("Provide at least one --bucket definition.")
    parser_map = _filter_parser_map()
    parsed: list[BucketSpec] = []
    for raw in raw_buckets:
        parsed.append(_parse_bucket_definition(raw, parser_map=parser_map))
    return parsed


def _bucket_mask(df: pd.DataFrame, filters: dict[str, object]) -> pd.Series:
    """Compute boolean mask for rows matching condition filters.

    Args:
        df: Dataframe from analysis.load_dataframe.
        filters: Condition filter mapping.

    Returns:
        Boolean mask indexed like ``df``.
    """
    if not filters:
        return pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    return df["condition_obj"].apply(
        lambda condition: condition_matches_filters(condition, filters)
    )


def _summarize_bucket(
    *,
    df: pd.DataFrame,
    bucket: BucketSpec,
    n_bootstrap: int,
) -> dict[str, object]:
    """Summarize one bucket's persuasion-delta statistics.

    Args:
        df: All filtered rounds dataframe.
        bucket: Parsed bucket definition.
        n_bootstrap: Bootstrap replicates for CI.

    Returns:
        Summary row dictionary.
    """
    mask = _bucket_mask(df, bucket.filters)
    sub = df.loc[mask]
    deltas = sub["delta_dir"].to_numpy(dtype=float)
    if deltas.size == 0:
        return {
            "bucket": bucket.label,
            "n_rounds": 0,
            "mean_delta": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "filters": ",".join(
                f"{key}={value}" for key, value in bucket.filters.items()
            ),
        }
    mean_delta, ci_lo, ci_hi = bootstrap_mean_ci(deltas, n_boot=n_bootstrap)
    return {
        "bucket": bucket.label,
        "n_rounds": int(deltas.size),
        "mean_delta": float(mean_delta),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "filters": ",".join(f"{key}={value}" for key, value in bucket.filters.items()),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write bucket summary rows to CSV.

    Args:
        path: Output CSV path.
        rows: Bucket summary rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["bucket", "n_rounds", "mean_delta", "ci95_lo", "ci95_hi", "filters"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _display_bucket_label(raw_bucket: str) -> str:
    """Convert internal bucket keys to paper-facing labels.

    Args:
        raw_bucket: Internal bucket label from CLI specs.

    Returns:
        Display label for plotting.
    """
    return BUCKET_LABEL_MAP.get(raw_bucket.strip(), raw_bucket.strip())


def _plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Render bar chart with 95% CIs for bucket means.

    Args:
        path: Output PDF path.
        rows: Bucket summary rows.
    """
    plot_rows = [
        row
        for row in rows
        if int(row["n_rounds"]) > 0 and np.isfinite(float(row["mean_delta"]))
    ]
    if not plot_rows:
        return
    plot_rows = sorted(
        plot_rows,
        key=lambda row: (
            -float(row["mean_delta"]),
            str(row["bucket"]),
        ),
    )

    labels = [_display_bucket_label(str(row["bucket"])) for row in plot_rows]
    means = np.asarray([float(row["mean_delta"]) for row in plot_rows], dtype=float)
    ci_lo = np.asarray([float(row["ci95_lo"]) for row in plot_rows], dtype=float)
    ci_hi = np.asarray([float(row["ci95_hi"]) for row in plot_rows], dtype=float)
    errors = np.vstack([means - ci_lo, ci_hi - means])

    x_pos = np.arange(len(plot_rows), dtype=float)
    fig, axis = plt.subplots(figsize=(4.0, 3.0))
    bars = axis.bar(x_pos, means, color="#4c78a8", alpha=0.9)
    axis.errorbar(
        x_pos,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=4,
    )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xticks(x_pos)
    axis.set_xticklabels(labels, rotation=0, ha="center")
    # axis.tick_params(axis="x", labelsize=6.0, pad=1.2)
    axis.set_ylabel("Mean Persuasion Delta (→)")
    axis.grid(axis="y", linestyle=":", alpha=0.25)
    # y_low, y_high = axis.get_ylim()
    # y_offset = 0.03 * (y_high - y_low)
    # for bar_patch, row in zip(bars, plot_rows):
    #     bar_height = float(bar_patch.get_height())
    #     x_center = float(bar_patch.get_x() + bar_patch.get_width() / 2.0)
    #     if bar_height >= 0.0:
    #         y_text = bar_height + y_offset
    #         vertical_alignment = "bottom"
    #     else:
    #         y_text = bar_height - y_offset
    #         vertical_alignment = "top"
    #     axis.text(
    #         x_center,
    #         y_text,
    #         f"N={int(row['n_rounds'])}",
    #         ha="center",
    #         va=vertical_alignment,
    #         fontsize=5.8,
    #         color="#2b2b2b",
    #     )
    # fig.subplots_adjust(left=0.34, bottom=0.30, right=0.99, top=0.99)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """Run bucket-level persuasiveness summary pipeline."""
    args = parse_args()
    buckets = _parse_buckets(list(args.bucket))
    output_csv = _resolve_repo_path(args.output_csv)
    output_pdf = _resolve_repo_path(args.output_pdf)

    df = load_dataframe(min_date=args.min_date, filters=None)
    if df.empty:
        raise ValueError("No rounds found for the provided min-date filter.")

    rows = [
        _summarize_bucket(df=df, bucket=bucket, n_bootstrap=int(args.bootstrap))
        for bucket in buckets
    ]
    if args.drop_empty:
        rows = [row for row in rows if int(row["n_rounds"]) > 0]
    if not rows:
        raise ValueError("No bucket rows available after filtering.")

    _write_csv(output_csv, rows)
    _plot_summary(output_pdf, rows)
    print(f"Wrote bucket summary CSV: {output_csv}")
    print(f"Wrote bucket summary figure: {output_pdf}")


if __name__ == "__main__":
    main()
