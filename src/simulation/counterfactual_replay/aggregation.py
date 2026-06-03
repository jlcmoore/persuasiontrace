"""Aggregation helpers for counterfactual replay round-error outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .math import finite_mean, replay_human_like_score

_REQUIRED_ROUND_ERROR_COLUMNS = (
    "corpus",
    "final_target_abs_error",
    "serial_trajectory_mae",
    "final_node_mae",
    "node_delta_mae",
)


def write_csv_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    """Write CSV rows to disk with explicit field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_round_error_rows(path: Path) -> list[dict[str, Any]]:
    """Load replay round-error CSV rows and coerce metric columns to floats."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for column in _REQUIRED_ROUND_ERROR_COLUMNS:
            if column not in fieldnames:
                raise ValueError(
                    f"Round-error CSV is missing required column '{column}': {path}"
                )
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            row["final_target_abs_error"] = _parse_float_or_nan(
                raw.get("final_target_abs_error")
            )
            row["serial_trajectory_mae"] = _parse_float_or_nan(
                raw.get("serial_trajectory_mae")
            )
            row["final_node_mae"] = _parse_float_or_nan(raw.get("final_node_mae"))
            row["node_delta_mae"] = _parse_float_or_nan(raw.get("node_delta_mae"))
            rows.append(row)
    return rows


def build_summary_rows(round_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate round-level replay errors to corpus-level summary rows."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in round_rows:
        corpus = str(row["corpus"])
        grouped.setdefault(corpus, []).append(row)
    summary_rows = [
        _summary_row_for_corpus(corpus=corpus, rows=rows)
        for corpus, rows in sorted(grouped.items(), key=lambda item: item[0])
    ]
    pooled_groups: dict[str, list[dict[str, Any]]] = {}
    for row in round_rows:
        pooled_key = _pooled_persona_corpus_key(str(row["corpus"]))
        if pooled_key is None:
            continue
        pooled_groups.setdefault(pooled_key, []).append(row)
    summary_rows.extend(
        _summary_row_for_corpus(corpus=corpus, rows=rows)
        for corpus, rows in sorted(pooled_groups.items(), key=lambda item: item[0])
    )
    return summary_rows


def summary_table_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert summary rows into the compact terminal table format."""
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        rows.append(
            {
                "corpus": row["corpus"],
                "n": int(row["n_rounds"]),
                "target_err": round(float(row["mean_final_target_abs_error"]), 4),
                "node_err": round(float(row["mean_final_node_mae"]), 4),
                "score": round(float(row["replay_human_like_score"]), 4),
            }
        )
    return rows


def _summary_row_for_corpus(
    *,
    corpus: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one corpus worth of replay round rows."""
    mean_final_target_error = finite_mean(
        [float(row["final_target_abs_error"]) for row in rows]
    )
    mean_serial_mae = finite_mean([float(row["serial_trajectory_mae"]) for row in rows])
    mean_final_node_mae = finite_mean([float(row["final_node_mae"]) for row in rows])
    mean_node_delta_mae = finite_mean([float(row["node_delta_mae"]) for row in rows])

    score = replay_human_like_score(
        mean_final_target_error=mean_final_target_error,
        mean_final_node_mae=mean_final_node_mae,
        mean_node_delta_mae=mean_node_delta_mae,
    )
    return {
        "corpus": corpus,
        "n_rounds": int(len(rows)),
        "mean_final_target_abs_error": mean_final_target_error,
        "mean_serial_trajectory_mae": mean_serial_mae,
        "mean_final_node_mae": mean_final_node_mae,
        "mean_node_delta_mae": mean_node_delta_mae,
        "replay_human_like_score": score,
    }


def _pooled_persona_corpus_key(corpus: str) -> str | None:
    """Return pooled corpus key for persona-split full simulated target rows."""
    marker = "__persona="
    prefix = "full_simulated_target__sim="
    if not corpus.startswith(prefix) or marker not in corpus:
        return None
    base = corpus.split(marker, maxsplit=1)[0]
    return f"{base}__persona=pooled"


def _parse_float_or_nan(raw_value: str | None) -> float:
    """Parse a float cell; return NaN for empty cells."""
    if raw_value is None:
        return math.nan
    value = raw_value.strip()
    if not value:
        return math.nan
    return float(value)
