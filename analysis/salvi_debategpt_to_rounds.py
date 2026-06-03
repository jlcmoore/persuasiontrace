"""Convert Salvi DebateGPT rows into synthetic round results for annotation.

This converter keeps the existing annotation pipeline unchanged by writing
rows into the repository's `results/` JSONL structure. It also writes a sidecar
index CSV with Likert-scale outcomes and treatment metadata for downstream
ordinal regression.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

DEFAULT_INPUT_CSV = Path("/tmp/debategpt.csv")
DEFAULT_OUTPUT_ROOT = Path("results")
DEFAULT_CONDITION_DIR = "human_target=True&llm_persuader=salvi-debategpt&fd=False"
DEFAULT_INDEX_CSV = Path("analysis/data/salvi_debategpt_round_index.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert Salvi DebateGPT CSV rows into synthetic results rounds for "
            "annotation and write a sidecar row index CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input DebateGPT CSV path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root results directory.",
    )
    parser.add_argument(
        "--condition-dir",
        default=DEFAULT_CONDITION_DIR,
        help="Condition directory name under results/.",
    )
    parser.add_argument(
        "--output-date",
        default=datetime.date.today().isoformat(),
        help="Output file date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DEFAULT_INDEX_CSV,
        help="Output sidecar index CSV path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of source rows to convert.",
    )
    return parser.parse_args()


def _normalize_side(raw_side: str) -> str:
    """Normalize a side label to `PRO` or `CON`.

    Args:
        raw_side: Side string from source data.

    Returns:
        Canonical side string.

    Raises:
        ValueError: If side is not recognized.
    """
    cleaned = str(raw_side).strip().upper()
    if cleaned in {"PRO", "CON"}:
        return cleaned
    raise ValueError(f"Unsupported side value: {raw_side!r}")


def _likert_to_unit_interval(score: int) -> float:
    """Map a 1..5 Likert score to [0.0, 1.0].

    Args:
        score: Integer in [1, 5].

    Returns:
        Rescaled value in [0.0, 1.0].

    Raises:
        ValueError: If score is outside [1, 5].
    """
    if score < 1 or score > 5:
        raise ValueError(f"Likert score must be in [1, 5], got {score}.")
    return float(score - 1) / 4.0


def _opponent_supports_proposition(participant_side: str) -> bool:
    """Infer whether the opponent supports the proposition.

    Args:
        participant_side: Canonical participant side (`PRO` or `CON`).

    Returns:
        True when opponent side is PRO, else False.
    """
    return participant_side == "CON"


def _opponent_side(participant_side: str) -> str:
    """Return the opposite side label.

    Args:
        participant_side: Canonical participant side (`PRO` or `CON`).

    Returns:
        Opponent side as `PRO` or `CON`.
    """
    if participant_side == "PRO":
        return "CON"
    return "PRO"


def _clean_text(value: object) -> str:
    """Return stripped text for a source field.

    Args:
        value: Raw CSV value.

    Returns:
        Trimmed string.
    """
    return str(value or "").strip()


def _build_messages(source_row: dict[str, str]) -> list[dict[str, str]]:
    """Build a fixed 3x2 multi-turn message sequence.

    Args:
        source_row: One source CSV row.

    Returns:
        Alternating target/persuader messages.
    """
    return [
        {"role": "target", "content": _clean_text(source_row["argument"])},
        {"role": "persuader", "content": _clean_text(source_row["argumentOpponent"])},
        {"role": "target", "content": _clean_text(source_row["rebuttal"])},
        {"role": "persuader", "content": _clean_text(source_row["rebuttalOpponent"])},
        {"role": "target", "content": _clean_text(source_row["conclusion"])},
        {"role": "persuader", "content": _clean_text(source_row["conclusionOpponent"])},
    ]


def _read_source_rows(path: Path, max_rows: int | None) -> list[dict[str, str]]:
    """Read source CSV rows.

    Args:
        path: Input CSV path.
        max_rows: Optional cap on row count.

    Returns:
        Loaded source rows.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_rows is not None:
        return rows[: max(0, int(max_rows))]
    return rows


def _build_round_payload(
    source_row: dict[str, str], source_index: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one synthetic round and one sidecar metadata row.

    Args:
        source_row: Source CSV row.
        source_index: Zero-based source row index.

    Returns:
        Tuple of (round payload, sidecar metadata row).
    """
    participant_side = _normalize_side(str(source_row["side"]))
    pre_likert = int(str(source_row["agreementPreTreatment"]).strip())
    post_likert = int(str(source_row["agreementPostTreatment"]).strip())
    initial_belief = _likert_to_unit_interval(pre_likert)
    final_belief = _likert_to_unit_interval(post_likert)
    opponent_supports = _opponent_supports_proposition(participant_side)
    treatment = _clean_text(source_row["treatmentType"])
    topic = _clean_text(source_row["topic"])

    round_payload: dict[str, object] = {
        "condition": {
            "roles": {"human_target": True, "llm_persuader": "salvi-debategpt"},
            "factual_domain": False,
        },
        "proposition": topic,
        "target_initial_belief": initial_belief,
        "target_final_belief": final_belief,
        "persuader_supports_proposition": opponent_supports,
        "target_id": f"salvi_target_{source_index}",
        "persuader_id": f"salvi_opp_{source_index}",
        "messages": _build_messages(source_row),
    }

    metadata_row: dict[str, object] = {
        "source_row_index": int(source_index),
        "debate_id": _clean_text(source_row["debateID"]),
        "treatment_type": treatment,
        "personalized": bool("personalized" in treatment.lower()),
        "opponent_is_ai": bool(treatment.startswith("Human-AI")),
        "participant_side": participant_side,
        "opponent_side": _opponent_side(participant_side),
        "perceived_opponent": _clean_text(source_row["perceivedOpponent"]).lower(),
        "topic": topic,
        "agreement_pre_likert": int(pre_likert),
        "agreement_post_likert": int(post_likert),
        "side_agreement_pre_likert": int(
            str(source_row["sideAgreementPreTreatment"]).strip()
        ),
        "side_agreement_post_likert": int(
            str(source_row["sideAgreementPostTreatment"]).strip()
        ),
        "target_initial_belief": float(initial_belief),
        "target_final_belief": float(final_belief),
        "persuader_supports_proposition": bool(opponent_supports),
    }
    return round_payload, metadata_row


def _write_results_jsonl(
    rounds: list[dict[str, object]],
    output_root: Path,
    condition_dir: str,
    output_date: str,
) -> Path:
    """Write synthetic rounds to one results JSONL file.

    Args:
        rounds: Round payloads.
        output_root: Results root directory.
        condition_dir: Encoded condition directory.
        output_date: Output file date (YYYY-MM-DD).

    Returns:
        Absolute output file path.
    """
    output_dir = output_root / condition_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_date}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(rounds, ensure_ascii=True) + "\n")
    return output_path.resolve()


def _write_index_csv(
    *,
    metadata_rows: list[dict[str, object]],
    output_path: Path,
    source_path: Path,
) -> None:
    """Write sidecar index rows with round-location join keys.

    Args:
        metadata_rows: Row metadata without round-location columns.
        output_path: CSV output path.
        source_path: Absolute path to the generated results JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "line_index",
        "round_index",
        "source_row_index",
        "debate_id",
        "treatment_type",
        "personalized",
        "opponent_is_ai",
        "participant_side",
        "opponent_side",
        "perceived_opponent",
        "topic",
        "agreement_pre_likert",
        "agreement_post_likert",
        "side_agreement_pre_likert",
        "side_agreement_post_likert",
        "target_initial_belief",
        "target_final_belief",
        "persuader_supports_proposition",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_index, base_row in enumerate(metadata_rows):
            row = dict(base_row)
            row["source_path"] = str(source_path)
            row["line_index"] = 0
            row["round_index"] = int(round_index)
            writer.writerow(row)


def main() -> None:
    """Run CSV-to-results conversion and sidecar index export."""
    args = parse_args()
    source_rows = _read_source_rows(args.input_csv, args.max_rows)
    rounds: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for source_index, source_row in enumerate(source_rows):
        round_payload, metadata_row = _build_round_payload(source_row, source_index)
        rounds.append(round_payload)
        metadata_rows.append(metadata_row)

    output_path = _write_results_jsonl(
        rounds=rounds,
        output_root=args.output_root,
        condition_dir=str(args.condition_dir),
        output_date=str(args.output_date),
    )
    _write_index_csv(
        metadata_rows=metadata_rows,
        output_path=args.index_csv,
        source_path=output_path,
    )

    print(f"Wrote {len(rounds)} rounds to {output_path}")
    print(f"Wrote index CSV to {args.index_csv}")


if __name__ == "__main__":
    main()
