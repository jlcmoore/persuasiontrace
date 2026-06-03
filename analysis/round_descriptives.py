"""
Summarize round-level descriptives by condition.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

from analysis.annotation_regression import condition_factors, extract_message_text
from experiment import load_round_results
from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import (
    add_condition_filter_args,
    condition_matches_filters,
    filters_from_args,
)

from .tables import print_table


def parse_args() -> argparse.Namespace:
    """Parse CLI args for round-level descriptives.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Summarize round-level descriptives by condition."
    )
    add_min_date_arg(parser)
    add_condition_filter_args(parser)
    parser.add_argument(
        "--output-csv",
        default="analysis/data/round_descriptives.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def count_role_messages(round_obj: object, role: str) -> int:
    """Count messages for a role within a round.

    Args:
        round_obj: Round-like object with messages.
        role: Message role to count.

    Returns:
        Count of messages matching the role.
    """
    messages = getattr(round_obj, "messages", [])
    role_name = role.lower()
    return sum(
        1 for message in messages if str(message.get("role", "")).lower() == role_name
    )


def total_role_message_chars(round_obj: object, role: str) -> int:
    """Return total characters for a role's messages in a round.

    Args:
        round_obj: Round-like object with messages.
        role: Message role to summarize.

    Returns:
        Total character count across messages for the role.
    """
    messages = getattr(round_obj, "messages", [])
    role_name = role.lower()
    total = 0
    for message in messages:
        if str(message.get("role", "")).lower() != role_name:
            continue
        text = extract_message_text(message)
        total += len(text)
    return total


def average_role_message_chars(round_obj: object, role: str) -> float:
    """Return average message length (chars) for a role in a round.

    Args:
        round_obj: Round-like object with messages.
        role: Message role to summarize.

    Returns:
        Mean message length in characters for the role.
    """
    count = count_role_messages(round_obj, role)
    if count == 0:
        return 0.0
    return float(total_role_message_chars(round_obj, role)) / count


def summarize(values: list[float]) -> tuple[float, float]:
    """Return mean and stderr for a list of floats.

    Args:
        values: Numeric values to summarize.

    Returns:
        (mean, stderr) for the provided values.
    """
    count = len(values)
    if count == 0:
        return float("nan"), float("nan")
    mean_val = sum(values) / count
    if count < 2:
        return mean_val, float("nan")
    variance = sum((val - mean_val) ** 2 for val in values) / (count - 1)
    stderr = math.sqrt(variance) / math.sqrt(count)
    return mean_val, stderr


def resolve_group_value(records: list[dict[str, object]], key: str) -> object:
    """Return a stable group value for a key, or 'mixed' if inconsistent.

    Args:
        records: Per-round records for a group.
        key: Field to resolve.

    Returns:
        The shared value if consistent, otherwise "mixed".
    """
    values = {record.get(key) for record in records}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def build_round_records(
    *, min_date: str | None, condition_filters: dict[str, object]
) -> list[dict[str, object]]:
    """Collect per-round records for descriptives.

    Args:
        min_date: Optional min date filter for rounds.
        condition_filters: Condition filter dictionary.

    Returns:
        List of per-round records with metrics.
    """
    cond_to_rounds = load_round_results(min_date)
    records: list[dict[str, object]] = []
    for condition, list_of_round_lists in cond_to_rounds.items():
        if condition_filters and not condition_matches_filters(
            condition, condition_filters
        ):
            continue
        condition_label = str(condition.as_non_id_role())
        factors = condition_factors(condition)
        for round_obj in itertools.chain.from_iterable(list_of_round_lists):
            if (
                round_obj.target_initial_belief is None
                or round_obj.target_final_belief is None
            ):
                continue
            persuader_count = count_role_messages(round_obj, "persuader")
            target_count = count_role_messages(round_obj, "target")
            total_messages = len(getattr(round_obj, "messages", []))
            records.append(
                {
                    "condition": condition_label,
                    "persuader_type": factors.get("persuader_type"),
                    "participant_proposition": factors.get("participant_proposition"),
                    "delta_changed": round_obj.target_belief_change(),
                    "persuader_msg_count": persuader_count,
                    "target_msg_count": target_count,
                    "total_messages": total_messages,
                    "persuader_avg_chars": average_role_message_chars(
                        round_obj, "persuader"
                    ),
                    "target_avg_chars": average_role_message_chars(round_obj, "target"),
                }
            )
    return records


def summarize_group(
    *,
    group_type: str,
    group_label: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize a group of round records into a single row.

    Args:
        group_type: Grouping type label (e.g., "overall", "condition").
        group_label: Group identifier.
        records: Per-round records to summarize.

    Returns:
        Summary row as a dictionary.
    """
    delta_vals = [float(record["delta_changed"]) for record in records]
    persuader_count = [float(record["persuader_msg_count"]) for record in records]
    target_count = [float(record["target_msg_count"]) for record in records]
    total_messages = [float(record["total_messages"]) for record in records]
    persuader_chars = [float(record["persuader_avg_chars"]) for record in records]
    target_chars = [float(record["target_avg_chars"]) for record in records]

    delta_mean, delta_stderr = summarize(delta_vals)
    persuader_count_mean, persuader_count_stderr = summarize(persuader_count)
    target_count_mean, target_count_stderr = summarize(target_count)
    total_messages_mean, total_messages_stderr = summarize(total_messages)
    persuader_chars_mean, persuader_chars_stderr = summarize(persuader_chars)
    target_chars_mean, target_chars_stderr = summarize(target_chars)

    return {
        "group_type": group_type,
        "group_label": group_label,
        "persuader_type": resolve_group_value(records, "persuader_type"),
        "participant_proposition": resolve_group_value(
            records, "participant_proposition"
        ),
        "n_rounds": len(records),
        "mean_delta_changed": delta_mean,
        "stderr_delta_changed": delta_stderr,
        "mean_persuader_msg_count": persuader_count_mean,
        "stderr_persuader_msg_count": persuader_count_stderr,
        "mean_target_msg_count": target_count_mean,
        "stderr_target_msg_count": target_count_stderr,
        "mean_total_messages": total_messages_mean,
        "stderr_total_messages": total_messages_stderr,
        "mean_persuader_avg_chars": persuader_chars_mean,
        "stderr_persuader_avg_chars": persuader_chars_stderr,
        "mean_target_avg_chars": target_chars_mean,
        "stderr_target_avg_chars": target_chars_stderr,
    }


def write_csv(
    output_path: Path, rows: list[dict[str, object]], columns: list[str]
) -> None:
    """Write summary rows to CSV.

    Args:
        output_path: Output CSV path.
        rows: Summary rows to write.
        columns: Column order to write.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the round-level descriptive summary."""
    args = parse_args()
    condition_filters = filters_from_args(args)
    records = build_round_records(
        min_date=args.min_date, condition_filters=condition_filters
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["condition"])].append(record)

    rows: list[dict[str, object]] = []
    if records:
        rows.append(
            summarize_group(
                group_type="overall",
                group_label="overall",
                records=records,
            )
        )
    for label in sorted(grouped):
        rows.append(
            summarize_group(
                group_type="condition",
                group_label=label,
                records=grouped[label],
            )
        )

    columns = [
        "group_type",
        "group_label",
        "persuader_type",
        "participant_proposition",
        "n_rounds",
        "mean_delta_changed",
        "stderr_delta_changed",
        "mean_persuader_msg_count",
        "stderr_persuader_msg_count",
        "mean_target_msg_count",
        "stderr_target_msg_count",
        "mean_total_messages",
        "stderr_total_messages",
        "mean_persuader_avg_chars",
        "stderr_persuader_avg_chars",
        "mean_target_avg_chars",
        "stderr_target_avg_chars",
    ]
    output_path = Path(args.output_csv)
    write_csv(output_path, rows, columns)

    if not rows:
        print("No rounds found for the requested filters.")
        return

    summary_rows = [
        {
            "group_label": row["group_label"],
            "n_rounds": row["n_rounds"],
            "mean_delta_changed": row["mean_delta_changed"],
            "stderr_delta_changed": row["stderr_delta_changed"],
            "mean_persuader_msg_count": row["mean_persuader_msg_count"],
            "mean_target_msg_count": row["mean_target_msg_count"],
            "mean_persuader_avg_chars": row["mean_persuader_avg_chars"],
            "mean_target_avg_chars": row["mean_target_avg_chars"],
        }
        for row in rows
        if row["group_type"] == "condition"
    ]
    print_table(
        summary_rows,
        columns=[
            "group_label",
            "n_rounds",
            "mean_delta_changed",
            "stderr_delta_changed",
            "mean_persuader_msg_count",
            "mean_target_msg_count",
            "mean_persuader_avg_chars",
            "mean_target_avg_chars",
        ],
        formatters={
            "mean_delta_changed": lambda value: f"{value:.3f}",
            "stderr_delta_changed": lambda value: f"{value:.3f}",
            "mean_persuader_msg_count": lambda value: f"{value:.2f}",
            "mean_target_msg_count": lambda value: f"{value:.2f}",
            "mean_persuader_avg_chars": lambda value: f"{value:.1f}",
            "mean_target_avg_chars": lambda value: f"{value:.1f}",
        },
        widths={"group_label": 52},
    )


if __name__ == "__main__":
    main()
