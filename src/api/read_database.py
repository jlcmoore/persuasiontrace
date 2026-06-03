"""
src/api/read_database.py

Author: Jared Moore
Date: July, 2025

Utilities to read the sql database
"""

import argparse
import json
import logging
import sys
import textwrap
from collections import Counter

import pandas as pd
from sqlalchemy import inspect
from sqlmodel import Session, col, create_engine, select

from experiment.condition import Condition, Roles
from experiment.round import Round, output_conditions_and_rounds
from experiment.utils import set_logger

from .sql_model import (
    CONNECT_ARGS,
    SQLITE_FILE_NAME,
    SQLITE_URL_FMT,
    FlaggedResponse,
    Proposition,
    RoundORM,
    SentMessageBase,
    ensure_schema_compatibility,
)
from .sql_queries import (
    exclude_round_messages,
    get_bonuses,
    get_feedback,
    get_flagged_messages,
    get_wait_statistics,
    has_dialogue_messages,
    print_finished_rounds,
    rounds_by_condition,
)

logger = logging.getLogger(__name__)
CANONICAL_GPT5_MODEL = "gpt-5-2025-08-07"
GPT5_ALIAS_MODEL = "gpt-5"


def _canonicalize_condition_for_export(condition: Condition) -> Condition:
    """
    Return a copy of a condition with canonicalized persuader model alias.

    Args:
        condition: Input condition.

    Returns:
        Condition with the GPT-5 persuader alias normalized.
    """
    roles = condition.roles
    canonical_llm_persuader = (
        CANONICAL_GPT5_MODEL
        if roles.llm_persuader == GPT5_ALIAS_MODEL
        else roles.llm_persuader
    )
    canonical_roles = Roles(
        human_persuader=roles.human_persuader,
        human_target=roles.human_target,
        llm_persuader=canonical_llm_persuader,
        llm_target=roles.llm_target,
        simulated_target=roles.simulated_target,
        simulated_target_persona=roles.simulated_target_persona,
    )
    if canonical_roles == roles:
        return condition
    return condition.model_copy(update={"roles": canonical_roles})


def _canonicalize_round_for_export(round_obj: Round) -> Round:
    """
    Return a round copy with canonicalized persuader alias in condition roles.

    Args:
        round_obj: Input round.

    Returns:
        Round whose ``condition.roles.llm_persuader`` uses canonical GPT-5 alias.
    """
    canonical_condition = _canonicalize_condition_for_export(round_obj.condition)
    if canonical_condition == round_obj.condition:
        return round_obj
    return round_obj.model_copy(update={"condition": canonical_condition})


def read_database():
    """
    Read the database main function.
    """
    parser = argparse.ArgumentParser(prog="Database Queries")
    parser.add_argument(
        "--database",
        type=str,
        default=SQLITE_FILE_NAME,
        help="Path to the database",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="WARNING",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="subcommand help", required=True
    )

    save_rounds_parser = subparsers.add_parser(
        "save-rounds", help="Save the current rounds"
    )
    save_rounds_parser.set_defaults(func=handle_save_rounds)
    save_rounds_parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Whether not to output the results to a file",
    )
    save_rounds_parser.add_argument(
        "--include-short-rounds",
        default=False,
        action="store_true",
        help="Whether to include the rounds that are too short",
    )

    gen_bonuses_parser = subparsers.add_parser(
        "get-bonuses", help="Outputs bonuses as csv"
    )
    gen_bonuses_parser.set_defaults(func=handle_get_bonuses)
    gen_bonuses_parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Only include rounds completed in the last N hours (default: all time)",
    )

    feedback_parser = subparsers.add_parser("get-feedback", help="Print all feedback")
    feedback_parser.set_defaults(func=handle_get_feedback)

    print_finished = subparsers.add_parser(
        "print-completed", help="Print completed rounds"
    )
    print_finished.add_argument("--external-id", type=str, required=True)
    print_finished.set_defaults(func=print_finished_rounds)

    wait_times_parser = subparsers.add_parser(
        "wait-times", help="Compute wait statistics"
    )
    wait_times_parser.set_defaults(func=get_wait_statistics)

    flagged_message_parser = subparsers.add_parser(
        "flagged-messages", help="Output all flagged messages"
    )
    flagged_message_parser.set_defaults(func=handle_flagged_messages)
    flagged_message_parser.add_argument(
        "--reason",
        type=str,
        default=None,
        choices=[fr.name for fr in FlaggedResponse],
        help="The reason for the flagged message.",
    )

    ppt_props_parser = subparsers.add_parser(
        "print-ppt-props",
        help="Print participant original inputs and rephrased propositions",
    )
    ppt_props_parser.set_defaults(func=handle_participant_propositions)
    ppt_props_parser.add_argument(
        "--include-short-rounds",
        default=False,
        action="store_true",
        help="Whether to include rounds that are too short or too quick",
    )
    ppt_props_parser.add_argument(
        "--include-unfinished",
        default=False,
        action="store_true",
        help="Whether to include rounds that are unfinished",
    )
    ppt_props_parser.add_argument(
        "--props-only",
        default=False,
        action="store_true",
        help="Whether to print only propositions, one per line",
    )

    args = parser.parse_args()

    set_logger(args.log, local_logger=logger)
    filename = SQLITE_URL_FMT.format(filename=args.database)
    engine = create_engine(filename, echo=False, connect_args=CONNECT_ARGS)
    # Upgrade older DBs by adding missing columns with defaults (NULL)
    ensure_schema_compatibility(engine)
    inspector = inspect(engine)
    with Session(engine) as session:
        expected_tables = {
            "roundorm",
            "externaluser",
            "participant",
            "sentmessage",
            "proposition",
        }
        existing_tables = set(inspector.get_table_names())
        if expected_tables != existing_tables:
            raise ValueError("Necessary tables do not exist.")

        all_args = vars(args).copy()
        all_args.pop("command", None)
        args.func(session, **all_args)


def handle_flagged_messages(session, reason: str | None, **_):
    """
    Handler to get flagged messages and print them.
    """
    messages = get_flagged_messages(
        session, FlaggedResponse[reason] if reason else None
    )
    for m in messages:
        json.dump(
            SentMessageBase(**m.model_dump()).model_dump(),
            sys.stdout,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        sys.stdout.write("\n")


def handle_get_bonuses(session, hours: float | None = None, **_):
    """
    Handler to call get_bonuses and outuput
    """
    bonuses = get_bonuses(session, hours)
    output_bonuses_to_csv(bonuses)


def output_bonuses_to_csv(bonuses: Counter[int]):
    """
    Outputs the bonuses as a CSV
    """
    bonuses_df = pd.DataFrame(
        list(bonuses.items()), columns=["Participant ID", "Bonus Count"]
    )
    # Ensure bonus amounts are numeric and formatted to two decimal places
    bonuses_df["Bonus Count"] = pd.to_numeric(
        bonuses_df["Bonus Count"], errors="coerce"
    ).round(2)

    print(bonuses_df)
    bonuses_df.to_csv("bonuses.csv", index=False, float_format="%.2f")
    print("Bonuses saved to bonuses.csv.")


def handle_get_feedback(session, **_):
    """
    Handler to call `get_feedback` with and ouput
    """
    feedback = get_feedback(session)
    if not feedback:
        print("No feedback found.")
        return

    for idx, entry in enumerate(feedback, start=1):
        text = entry or ""
        wrapped = textwrap.fill(text, subsequent_indent="   ")
        print(f"{idx}. {wrapped}")
        print("")


def handle_save_rounds(
    session, dry_run: bool = True, include_short_rounds: bool = True, **_
):
    """
    Handler to save rounds with and ouput
    """
    all_round_summary = _compute_round_export_summary(
        session, include_short_rounds=include_short_rounds
    )
    raw_condition_to_rounds = rounds_by_condition(
        session, include_short_rounds=include_short_rounds
    )
    condition_to_rounds: dict[Condition, list[Round]] = {}
    for condition, rounds in raw_condition_to_rounds.items():
        canonical_condition = _canonicalize_condition_for_export(condition)
        canonical_rounds = [
            _canonicalize_round_for_export(round_obj) for round_obj in rounds
        ]
        if canonical_condition not in condition_to_rounds:
            condition_to_rounds[canonical_condition] = []
        condition_to_rounds[canonical_condition].extend(canonical_rounds)

    output_conditions_and_rounds(
        condition_to_rounds,
        dry_run=dry_run,
        round_summary=all_round_summary,
    )


def _compute_round_export_summary(session, include_short_rounds: bool):
    """
    Compute round export summary by non-id condition.

    Args:
        session: SQL session to query rounds and messages.
        include_short_rounds: Whether to include short/quick rounds.

    Returns:
        Dict mapping non-id conditions to summary counts.
    """
    rounds = session.exec(select(RoundORM).order_by(RoundORM.updated_at)).all()
    summary: dict[object, dict[str, int]] = {}
    for rd_orm in rounds:
        condition = rd_orm.condition()
        if condition is None:
            continue
        non_id_condition = _canonicalize_condition_for_export(
            condition.as_non_id_role()
        )
        if non_id_condition not in summary:
            summary[non_id_condition] = {
                "total_rounds": 0,
                "saved_rounds": 0,
                "unique_persuader_ids": set(),
                "unique_target_ids": set(),
                "saved_persuader_ids": set(),
                "saved_target_ids": set(),
                "excluded_unfinished": 0,
                "excluded_short_quick": 0,
                "excluded_timed_out_no_messages": 0,
            }
        counts = summary[non_id_condition]
        counts["total_rounds"] += 1
        if rd_orm.persuader_id is not None:
            counts["unique_persuader_ids"].add(rd_orm.persuader_id)
        if rd_orm.target_id is not None:
            counts["unique_target_ids"].add(rd_orm.target_id)

        round_obj = rd_orm.as_round()
        if rd_orm.timed_out and not has_dialogue_messages(round_obj):
            counts["excluded_timed_out_no_messages"] += 1
            continue

        finished = (
            rd_orm.target_initial_belief is not None
            and rd_orm.target_final_belief is not None
        )
        saved = False

        if not finished:
            counts["excluded_unfinished"] += 1
        else:
            if include_short_rounds:
                saved = True
            else:
                too_short, too_quick = exclude_round_messages(session, rd_orm)
                if too_short or too_quick:
                    counts["excluded_short_quick"] += 1
                else:
                    saved = True

        if saved:
            counts["saved_rounds"] += 1
            if rd_orm.persuader_id is not None:
                counts["saved_persuader_ids"].add(rd_orm.persuader_id)
            if rd_orm.target_id is not None:
                counts["saved_target_ids"].add(rd_orm.target_id)

    summarized: dict[object, dict[str, int]] = {}
    for condition, counts in summary.items():
        summarized[condition] = {
            "total_rounds": counts["total_rounds"],
            "saved_rounds": counts["saved_rounds"],
            "excluded_unfinished": counts["excluded_unfinished"],
            "excluded_short_quick": counts["excluded_short_quick"],
            "excluded_timed_out_no_messages": counts["excluded_timed_out_no_messages"],
            "unique_persuaders": len(counts["unique_persuader_ids"]),
            "unique_targets": len(counts["unique_target_ids"]),
            "unique_persuaders_saved": len(counts["saved_persuader_ids"]),
            "unique_targets_saved": len(counts["saved_target_ids"]),
        }
    return summarized


def handle_participant_propositions(
    session,
    include_short_rounds: bool = False,
    include_unfinished: bool = False,
    props_only: bool = False,
    **_,
):
    """
    Print participant original inputs and rephrased propositions.

    Args:
        session: SQLAlchemy session for database access.
        include_short_rounds: Whether to include short/quick rounds.
        include_unfinished: Whether to include unfinished rounds.
        props_only: Whether to print only propositions, one per line.

    Returns:
        None.
    """
    label_width = 12
    indent = " " * (label_width + 1)
    statement = (
        select(RoundORM, Proposition)
        .join(Proposition, RoundORM.proposition == Proposition.id)
        .where(col(Proposition.participant_id).is_not(None))
        .order_by(Proposition.participant_id, Proposition.id, RoundORM.id)
    )
    rows = session.exec(statement).all()
    if not rows:
        print("No participant propositions found.")
        return

    current_participant = None
    seen_props: set[str] = set()
    for round_row, prop in rows:
        if prop.id in seen_props:
            continue
        if round_row.timed_out and not has_dialogue_messages(round_row.as_round()):
            continue
        finished = (
            round_row.target_initial_belief is not None
            and round_row.target_final_belief is not None
        )
        if not include_unfinished and not finished:
            continue
        if not include_short_rounds:
            too_short, too_quick = exclude_round_messages(session, round_row)
            if too_short or too_quick:
                continue
        seen_props.add(prop.id)
        if props_only:
            sys.stdout.write(f"{prop.id}\n")
            continue
        if prop.participant_id != current_participant:
            current_participant = prop.participant_id
            print(f"\nParticipant {current_participant}")
            print("-" * 20)
        original = prop.original_text or ""
        print(
            textwrap.fill(
                f"{'Original:':<{label_width}} {original}",
                subsequent_indent=indent,
            )
        )
        print(
            textwrap.fill(
                f"{'Proposition:':<{label_width}} {prop.id}",
                subsequent_indent=indent,
            )
        )
        print("")
