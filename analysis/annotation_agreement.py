"""
Compute inter-annotator agreement between two annotation models.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from analysis.tables import print_table
from annotation.records import (
    ANNOTATION_CODES,
    add_annotation_args,
    add_max_rows_arg,
    condition_from_source_path,
    iter_annotation_jsonl_records,
    parse_annotation_record,
    resolve_annotation_paths_with_default,
)
from experiment.condition_filters import (
    add_condition_filter_args,
    condition_matches_filters,
    filters_from_args,
)
from experiment.round_lookup import normalize_source_path

SUMMARY_COLUMNS = [
    "model_a",
    "model_b",
    "code",
    "pairs",
    "agreements",
    "disagreements",
    "agreement_rate",
    "kappa",
]
SCORE_COLUMNS = [
    "model_a",
    "model_b",
    "code",
    "score",
    "model_a_count",
    "model_b_count",
    "agree_count",
    "disagree_count",
    "agree_rate",
    "share_either",
]


@dataclass(frozen=True)
class AnnotationRow:
    """Container for one parsed annotation row."""

    model: str
    key: tuple[object, ...]
    scores: dict[str, int]
    source_path: str


@dataclass(frozen=True)
class ScoreCounts:
    """Counts for a specific annotation score."""

    count_a: int
    count_b: int
    agree: int


def parse_args() -> argparse.Namespace:
    """Parse CLI args for inter-annotator agreement."""
    parser = argparse.ArgumentParser(
        description="Summarize inter-annotator agreement for annotations."
    )
    add_annotation_args(parser)
    add_condition_filter_args(parser)
    parser.add_argument(
        "--model-a",
        type=str,
        default=None,
        help="Model name for annotator A (defaults to auto-detection).",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Model name for annotator B (defaults to auto-detection).",
    )
    add_max_rows_arg(parser)
    return parser.parse_args()


def target_key_from_record(target: dict[str, object]) -> tuple[object, ...]:
    """Build a stable key for matching two annotations."""
    return (
        normalize_source_path(str(target.get("source_path", ""))),
        target.get("line_index"),
        target.get("round_index"),
        target.get("role"),
        target.get("message_index"),
        target.get("unit_type"),
        target.get("sentence_index"),
    )


def read_annotation_rows(
    paths: Iterable[Path],
    *,
    max_rows: int | None,
) -> list[AnnotationRow]:
    """Read annotation rows from JSONL files, attaching model names."""
    rows: list[AnnotationRow] = []
    current_path: Path | None = None
    model_name = "unknown"
    for path, record in iter_annotation_jsonl_records(list(paths), max_rows=max_rows):
        if current_path != path:
            current_path = path
            model_name = "unknown"
        model_name = update_model_name(record, model_name)
        row = build_annotation_row(record, model_name)
        if row is None:
            continue
        rows.append(row)
    return rows


def update_model_name(record: dict[str, object], current_name: str) -> str:
    """Return the updated model name for a record."""
    record_type = record.get("type")
    if record_type != "meta":
        return current_name
    model_val = record.get("model")
    if isinstance(model_val, str) and model_val.strip():
        return model_val.strip()
    return current_name


def coerce_scores(raw_scores: dict[str, float]) -> dict[str, int]:
    """Coerce raw scores into bounded integer scores."""
    scores: dict[str, int] = {}
    for code, value in raw_scores.items():
        score = int(round(value))
        if score < 0 or score > 5:
            continue
        scores[code] = score
    return scores


def build_annotation_row(
    record: dict[str, object], model_name: str
) -> AnnotationRow | None:
    """Build an AnnotationRow from a parsed record."""
    if record.get("type") != "annotation":
        return None
    parsed = parse_annotation_record(record)
    if parsed is None:
        return None
    raw_scores, target = parsed
    scores = coerce_scores(raw_scores)
    if not scores:
        return None
    source_path = normalize_source_path(str(target.get("source_path", "")))
    return AnnotationRow(
        model=model_name,
        key=target_key_from_record(target),
        scores=scores,
        source_path=source_path,
    )


def select_models(
    model_names: list[str], model_a: str | None, model_b: str | None
) -> tuple[str, str]:
    """Choose the two models to compare."""
    if model_a and model_b:
        return model_a, model_b
    unique_models = sorted(set(model_names))
    if len(unique_models) != 2:
        raise ValueError(
            "Expected exactly two models; pass --model-a and --model-b to choose."
        )
    return unique_models[0], unique_models[1]


def build_pair_scores(
    rows: list[AnnotationRow],
    *,
    model_a: str,
    model_b: str,
    condition_filters: dict[str, object] | None,
) -> dict[str, list[tuple[int, int]]]:
    """Collect paired scores per code for two models."""
    by_model: dict[str, dict[tuple[object, ...], AnnotationRow]] = defaultdict(dict)
    for row in rows:
        if condition_filters:
            condition_obj = condition_from_source_path(row.source_path)
            if condition_obj is not None and not condition_matches_filters(
                condition_obj, condition_filters
            ):
                continue
        by_model[row.model][row.key] = row

    paired_scores: dict[str, list[tuple[int, int]]] = {
        code: [] for code in ANNOTATION_CODES
    }
    keys = set(by_model.get(model_a, {})).intersection(by_model.get(model_b, {}))
    for key in keys:
        row_a = by_model[model_a][key]
        row_b = by_model[model_b][key]
        for code in ANNOTATION_CODES:
            score_a = row_a.scores.get(code)
            score_b = row_b.scores.get(code)
            if score_a is None or score_b is None:
                continue
            paired_scores[code].append((score_a, score_b))
    return paired_scores


def agreement_summary(
    paired_scores: dict[str, list[tuple[int, int]]],
) -> list[dict[str, object]]:
    """Compute agreement summary rows for each code."""
    rows: list[dict[str, object]] = []
    for code in ANNOTATION_CODES:
        pairs = paired_scores.get(code, [])
        total = len(pairs)
        agreements = sum(1 for score_a, score_b in pairs if score_a == score_b)
        disagreements = total - agreements
        agreement_rate = agreements / total if total else 0.0
        kappa = cohens_kappa(pairs)
        rows.append(
            {
                "code": code,
                "pairs": total,
                "agreements": agreements,
                "disagreements": disagreements,
                "agreement_rate": agreement_rate,
                "kappa": kappa,
            }
        )
    return rows


def cohens_kappa(pairs: list[tuple[int, int]]) -> float:
    """Compute Cohen's kappa for paired integer scores."""
    total = len(pairs)
    if total == 0:
        return 0.0
    agree_count = sum(1 for score_a, score_b in pairs if score_a == score_b)
    observed_agreement = agree_count / total
    counts_a: Counter[int] = Counter()
    counts_b: Counter[int] = Counter()
    for score_a, score_b in pairs:
        counts_a[score_a] += 1
        counts_b[score_b] += 1
    expected_agreement = 0.0
    for score in range(0, 6):
        expected_agreement += (counts_a.get(score, 0) / total) * (
            counts_b.get(score, 0) / total
        )
    if expected_agreement >= 1.0:
        return 0.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def agreement_by_score(
    paired_scores: dict[str, list[tuple[int, int]]],
) -> list[dict[str, object]]:
    """Compute agreement counts by score (symmetric across annotators)."""
    rows: list[dict[str, object]] = []
    for code in ANNOTATION_CODES:
        pairs = paired_scores.get(code, [])
        rows.extend(build_score_rows(code, pairs))
    return rows


def build_score_rows(
    code: str, pairs: list[tuple[int, int]]
) -> list[dict[str, object]]:
    """Build per-score agreement rows for a single code."""
    per_score_a, per_score_b, per_score_both = score_counters(pairs)
    total_pairs = len(pairs)
    rows: list[dict[str, object]] = []
    for score in range(0, 6):
        counts = ScoreCounts(
            count_a=per_score_a.get(score, 0),
            count_b=per_score_b.get(score, 0),
            agree=per_score_both.get(score, 0),
        )
        rows.append(
            build_score_row(
                code=code,
                score=score,
                counts=counts,
                total_pairs=total_pairs,
            )
        )
    return rows


def score_counters(
    pairs: list[tuple[int, int]],
) -> tuple[Counter[int], Counter[int], Counter[int]]:
    """Return per-score counters for both annotators."""
    per_score_a: Counter[int] = Counter()
    per_score_b: Counter[int] = Counter()
    per_score_both: Counter[int] = Counter()
    for score_a, score_b in pairs:
        per_score_a[score_a] += 1
        per_score_b[score_b] += 1
        if score_a == score_b:
            per_score_both[score_a] += 1
    return per_score_a, per_score_b, per_score_both


def build_score_row(
    *,
    code: str,
    score: int,
    counts: ScoreCounts,
    total_pairs: int,
) -> dict[str, object]:
    """Return a formatted per-score agreement row."""
    either = counts.count_a + counts.count_b - counts.agree
    disagree = either - counts.agree
    agree_rate = counts.agree / either if either else 0.0
    share_either = either / total_pairs if total_pairs else 0.0
    return {
        "code": code,
        "score": score,
        "model_a_count": counts.count_a,
        "model_b_count": counts.count_b,
        "agree_count": counts.agree,
        "disagree_count": disagree,
        "agree_rate": agree_rate,
        "share_either": share_either,
    }


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_agreement_tables(
    model_a: str,
    model_b: str,
    summary_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
) -> None:
    """Print agreement summary tables."""
    print(f"Models: {model_a} vs {model_b}")
    print("\nOverall agreement by code")
    print_table(
        summary_rows,
        columns=[
            "code",
            "pairs",
            "agreements",
            "disagreements",
            "agreement_rate",
            "kappa",
        ],
        formatters={
            "agreement_rate": lambda value: f"{value:.3f}",
            "kappa": lambda value: f"{value:.3f}",
        },
        aligns={
            "pairs": "right",
            "agreements": "right",
            "disagreements": "right",
            "agreement_rate": "right",
            "kappa": "right",
        },
    )
    print("\nAgreement by score (symmetric)")
    print_table(
        score_rows,
        columns=[
            "code",
            "score",
            "model_a_count",
            "model_b_count",
            "agree_count",
            "disagree_count",
            "agree_rate",
            "share_either",
        ],
        formatters={
            "agree_rate": lambda value: f"{value:.3f}",
            "share_either": lambda value: f"{value:.3f}",
        },
        aligns={
            "score": "right",
            "model_a_count": "right",
            "model_b_count": "right",
            "agree_count": "right",
            "disagree_count": "right",
            "agree_rate": "right",
            "share_either": "right",
        },
    )


def attach_model_labels(
    rows: list[dict[str, object]], model_a: str, model_b: str
) -> list[dict[str, object]]:
    """Attach model labels to each row."""
    return [{"model_a": model_a, "model_b": model_b, **row} for row in rows]


def write_agreement_outputs(
    *,
    model_a: str,
    model_b: str,
    summary_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Write agreement outputs to CSV."""
    summary_csv = Path("analysis/data/annotation_agreement_summary.csv")
    score_csv = Path("analysis/data/annotation_agreement_by_score.csv")
    summary_rows_with_models = attach_model_labels(summary_rows, model_a, model_b)
    score_rows_with_models = attach_model_labels(score_rows, model_a, model_b)
    write_csv(summary_csv, summary_rows_with_models, SUMMARY_COLUMNS)
    write_csv(score_csv, score_rows_with_models, SCORE_COLUMNS)
    return summary_csv, score_csv


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    annotation_paths = resolve_annotation_paths_with_default(args.annotations)
    rows = read_annotation_rows(annotation_paths, max_rows=args.max_rows)
    model_names = [row.model for row in rows]
    model_a, model_b = select_models(model_names, args.model_a, args.model_b)
    condition_filters = filters_from_args(args)

    paired_scores = build_pair_scores(
        rows, model_a=model_a, model_b=model_b, condition_filters=condition_filters
    )
    summary_rows = agreement_summary(paired_scores)
    score_rows = agreement_by_score(paired_scores)

    summary_csv, score_csv = write_agreement_outputs(
        model_a=model_a,
        model_b=model_b,
        summary_rows=summary_rows,
        score_rows=score_rows,
    )

    print_agreement_tables(model_a, model_b, summary_rows, score_rows)
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {score_csv}")


if __name__ == "__main__":
    main()
