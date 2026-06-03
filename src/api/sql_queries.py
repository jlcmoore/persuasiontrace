"""
src/api/sql_queries.py

Author: Jared Moore
Date: July, 2025

Utilities to query the SQL database.
"""

import logging
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from typing import Counter as TypeCounter
from typing import Iterable, Tuple

import pandas as pd
from sqlalchemy import func
from sqlmodel import Session, asc, desc, or_, select

from experiment.condition import PERSUADER_BONUS, TARGET_BONUS, Condition
from experiment.round import Round

from .sql_model import (
    ExternalUser,
    FlaggedResponse,
    Participant,
    Proposition,
    RoundORM,
    SentMessage,
)
from .utils import ServerSettings, min_positive_timedelta_diff

logger = logging.getLogger(__name__)


def _proposition_has_belief_nodes(proposition: Proposition) -> bool:
    """
    Check whether a proposition has non-empty Bayesian-network belief nodes.

    Args:
        proposition: Proposition candidate.

    Returns:
        True when `bayesian_network.belief_nodes` contains at least one
        non-empty node string.
    """
    bayes_payload = proposition.bayesian_network
    if not isinstance(bayes_payload, dict):
        return False
    belief_nodes = bayes_payload.get("belief_nodes")
    if not isinstance(belief_nodes, list):
        return False
    for node_text in belief_nodes:
        if str(node_text).strip():
            return True
    return False


def _choose_proposition_by_assignment_frequency(
    possible: list[Proposition], session: Session, control_dialogue: bool
) -> Proposition:
    if not possible:
        raise ValueError("Cannot choose from an empty proposition list")

    if len(possible) == 1:
        return possible[0]

    proposition_ids = [proposition.id for proposition in possible]
    chosen_column = (
        RoundORM.proposition_during_round if control_dialogue else RoundORM.proposition
    )
    count_rows = session.exec(
        select(
            chosen_column,
            func.count(),  # pylint: disable=not-callable
        )
        .where(chosen_column.in_(proposition_ids))  # pylint: disable=no-member
        .group_by(chosen_column)
    ).all()
    assigned_counts = {prop_id: int(count) for prop_id, count in count_rows}

    # Inverse-frequency weighting: propositions that have been assigned fewer
    # times are sampled more often, while overused options remain possible.
    weights = [
        1.0 / (1.0 + assigned_counts.get(proposition.id, 0)) ** 2
        for proposition in possible
    ]
    return random.choices(possible, weights=weights, k=1)[0]


def get_all_rounds(
    first: Participant, second: Participant | None, session: Session
) -> list[RoundORM]:
    """Returns all of the rounds `first` and `second` have played in."""
    rounds = get_participant_rounds(first, session)
    all_rounds = rounds
    if second:
        other_rounds = get_participant_rounds(second, session)
        all_rounds.extend(other_rounds)
    return all_rounds


def has_dialogue_messages(round_obj: Round) -> bool:
    """Return whether a round has at least one non-empty dialogue message.

    Args:
        round_obj: Round to inspect.

    Returns:
        True when any message contains non-whitespace text.
    """
    for message in round_obj.messages:
        content = str(message.get("content", ""))
        if content.strip():
            return True
    return False


def exclude_round_messages(session: Session, rd: RoundORM) -> (bool, bool):
    """
    Counts the time each message took to send and its characters in Round.
    Returns a (bool, bool) if the messages where too_short or too_quick
    (fewer than 10 characters, shorter than 5 seconds). Only applies
    to humans.
    """
    messages = session.exec(
        select(SentMessage)
        .where(SentMessage.round_id == rd.id)
        .where(SentMessage.flagged.is_not(True))  # pylint: disable=no-member
        .order_by(asc(SentMessage.created_at), desc(SentMessage.is_target))
        # earliest to latest, then whether it is the target or not
    ).all()

    target_lengths = []
    persuader_lengths = []
    target_times = []
    persuader_times = []

    # The start of the round plus 30 seconds (how long they have to wait on the
    # instructions page for)
    # TODO: make this a constant that is passed to the Vue pages
    last_sent = rd.created_at + timedelta(seconds=30)
    last_was_target = None

    for message in messages:
        if not message.message_content:
            continue

        characters = len(message.message_content)
        # If somehow we mess up the order
        elapsed_time = min_positive_timedelta_diff(message.created_at, last_sent)
        last_sent = message.created_at
        count_timestamp = not last_was_target or last_was_target != message.is_target
        if not count_timestamp:
            logger.warning("Message out of order. Ignoring time stamp diffs.")
        last_was_target = message.is_target

        # Only count the messages and times for human targets and persuaders
        if message.is_target:
            target_lengths.append(characters)
            if count_timestamp:
                target_times.append(elapsed_time)
        else:
            persuader_lengths.append(characters)
            if count_timestamp:
                persuader_times.append(elapsed_time)

    if rd.persuader_id is not None and rd.target_id is None:
        # We need to swap the times here.
        # Persuader messages and LLM response get added at the same time.
        persuader_times, target_times = target_times, persuader_times

    too_short = False
    too_quick = False

    if persuader_lengths and rd.persuader_id is not None:
        assert persuader_times
        avg_persuader_length = sum(persuader_lengths) / len(persuader_lengths)
        avg_persuader_time = sum(persuader_times, timedelta(0)) / len(persuader_times)
        too_short |= avg_persuader_length < 10
        too_quick |= avg_persuader_time < timedelta(seconds=5)

    if target_lengths and rd.target_id is not None:
        assert target_times
        avg_target_length = sum(target_lengths) / len(target_lengths)
        avg_target_time = sum(target_times, timedelta(0)) / len(target_times)

        too_short |= avg_target_length < 10
        too_quick |= avg_target_time < timedelta(seconds=5)

    if too_short:
        short_msg = "Messages too short "
        if rd.target_id:
            short_msg += f"target lengths: {target_lengths}, "
        if rd.persuader_id:
            short_msg += f"persuader lengths: {persuader_lengths}"
        logger.warning(short_msg)

    if too_quick:
        quick_msg = "Messages sent too quickly "
        if rd.target_id:
            quick_msg += f"target times: {target_times}, "
        if rd.persuader_id:
            quick_msg += f"persuader times: {persuader_times}"
        logger.warning(quick_msg)

    return too_short, too_quick


def rounds_by_condition(
    session: Session,
    include_short_rounds: bool = False,
    group_by_persuader: bool = True,
) -> dict[Condition, list[Tuple[Round, dict[str, Any]]]]:
    """
    Returns all of the current rounds as a dict of Conditions to (Rounds, metadata)
    Excludes unfinished rounds (where either of the target and persuader have not chosen)
    Excludes rounds in which one of the participants sent fewer than 10 characters per message
        or spent fewer than 5s on their turns (sent messages) on average.

        group_by_persuader (bool): If True groups games by persuader id, not target id
    """
    # Order the rounds first by persuader and then earliest to latest
    # Ignore incompleted rounds.
    rounds = session.exec(
        select(RoundORM)
        .where(RoundORM.target_final_belief.is_not(None))  # pylint: disable=no-member
        .where(RoundORM.target_initial_belief.is_not(None))  # pylint: disable=no-member
        .order_by(RoundORM.persuader_id)
        .order_by(RoundORM.updated_at)
    ).all()

    condition_to_rounds: dict[Condition, list[Round]] = {}
    for rd_orm in rounds:
        rd = rd_orm.as_round()

        if rd.timed_out and not has_dialogue_messages(rd):
            logger.info(
                "Skipping timed-out round %s with no dialogue messages", rd_orm.id
            )
            continue

        too_short, too_quick = exclude_round_messages(session, rd_orm)
        if not rd.finished():
            logger.warning(f"Round for rd {rd} is not over")
        elif include_short_rounds or (not too_quick and not too_short):

            # Set it so that just half of the condition has an id.
            # TODO: attend to this later for other possible grouping mechanisms
            condition = rd.condition.as_non_id_role(no_target_id=not group_by_persuader)

            if condition not in condition_to_rounds:
                condition_to_rounds[condition] = []

            condition_to_rounds[condition].append(rd)
    return condition_to_rounds


def get_wait_statistics(session: Session, **_):
    """
    1) For each human participant and each completed round they play,
       computes how long they waited in the lobby before that round started.
       - For the Nth round (N>1):  round.created_at - (N-1).round.updated_at
       - For the 1st round:       round.created_at - participant.created_at
         (only if participant.created_at is set; else we skip it)

    2) For each completed round, computes the reply delays between the two sides,
       collecting two lists of delays (in seconds):
         a) including the very first inter-message delay,
         b) excluding the very first inter-message delay in each round.

    Prints summary (count, mean, median) for both metrics.
    """
    # 1) Fetch all completed rounds, sorted by creation time
    rounds = session.exec(select(RoundORM).order_by(RoundORM.created_at)).all()

    # Group per participant
    per_participant: dict[int, list[RoundORM]] = defaultdict(list)
    for rd in rounds:
        for pid in (rd.persuader_id, rd.target_id):
            if pid is not None:
                per_participant[pid].append(rd)

    lobby_waits = []  # in seconds

    # Compute lobby waits
    for pid, player_rounds in per_participant.items():
        player = session.get(Participant, pid)
        # sort rounds by created_at
        player_rounds.sort(key=lambda r: r.created_at)
        for i, rd in enumerate(player_rounds):
            if rd.created_at is None:
                continue
            if i == 0:
                prev_ts = player.created_at
            else:
                # Subsequent rounds: measure from the time they last spoke in the prior round
                prev_rd = player_rounds[i - 1]
                # Determine whether they were target or persuader in the previous round
                is_target_prev = prev_rd.target_id == pid
                last_msg = get_last_sent_message(session, is_target_prev, prev_rd.id)
                if last_msg is None or last_msg.created_at is None:
                    # If they never sent a message, fall back to the previous round's updated_at
                    prev_ts = prev_rd.updated_at
                else:
                    prev_ts = last_msg.created_at
            if prev_ts and rd.created_at >= prev_ts:
                wait_t = (rd.created_at - prev_ts).total_seconds()
                lobby_waits.append(wait_t)

    # 2) Compute reply delays
    reply_incl = []
    reply_excl = []

    for rd in rounds:
        msgs = session.exec(
            select(SentMessage)
            .where(SentMessage.round_id == rd.id)
            .where(SentMessage.flagged.is_(False))  # pylint: disable=no-member
            .order_by(SentMessage.created_at)
        ).all()

        last_ts = {True: None, False: None}  # True=target, False=persuader
        saw_first = False

        for msg in msgs:
            other = not msg.is_target
            prev = last_ts[other]
            if prev is not None and msg.created_at:
                delta_s = (msg.created_at - prev).total_seconds()
                reply_incl.append(delta_s)
                if saw_first:
                    reply_excl.append(delta_s)
                else:
                    saw_first = True
            last_ts[msg.is_target] = msg.created_at

    # Helper to print stats
    def summarize(name, data):
        if not data:
            print(f"{name}: no data")
            return
        print(
            f"{name}: count={len(data)}, "
            f"max={max(data):.1f}s, "
            f"mean={statistics.mean(data):.1f}s, "
            f"median={statistics.median(data):.1f}s"
        )

    print("\n=== Lobby Wait Times (seconds) ===")
    print("\t(This includes incomplete rounds.)")
    summarize("All inferred lobby waits", lobby_waits)

    print("\n=== Reply Delays (seconds) ===")
    summarize("Including first reply", reply_incl)
    summarize("Excluding first reply", reply_excl)


def _scalar_from_exec(session: Session, stmt) -> int:
    """
    Execute a COUNT(*)-style statement and robustly return the scalar integer
    across SQLAlchemy/SQLModel result variants.
    """
    res = session.exec(stmt)
    row = res.first()
    if row is None:
        return 0
    if isinstance(row, (tuple, list)):
        return int(row[0])
    return int(row)


def print_finished_rounds(session: Session, external_id: str, **_):
    """
    For the passed external participant prints the number of rounds they finished.
    """
    base_statement = (
        select(func.count())  # pylint: disable=not-callable
        .select_from(ExternalUser)
        .join(Participant, ExternalUser.id == Participant.id)
        .where(ExternalUser.external_id == external_id)
    )

    as_target = (
        base_statement.join(RoundORM, RoundORM.target_id == Participant.id)
        .where(RoundORM.target_final_belief.is_not(None))  # pylint: disable=no-member
        .where(RoundORM.target_initial_belief.is_not(None))  # pylint: disable=no-member
    )

    as_persuader = (
        base_statement.join(RoundORM, RoundORM.persuader_id == Participant.id)
        .where(RoundORM.target_final_belief.is_not(None))  # pylint: disable=no-member
        .where(RoundORM.target_initial_belief.is_not(None))  # pylint: disable=no-member
    )

    # Robustly fetch the single COUNT() value for each query across SQLAlchemy versions.
    count = _scalar_from_exec(session, as_target) + _scalar_from_exec(
        session, as_persuader
    )

    print(f"Completed rounds: {count}")


def get_feedback(session: Session) -> list[str]:
    """
    Returns the anonymized feedback given by each participant, if it exists.
    """

    base_statement = select(Participant.feedback).where(
        Participant.feedback.is_not(None)  # pylint: disable=no-member
    )
    return session.exec(base_statement).all()


def get_flagged_messages(
    session: Session, reason: FlaggedResponse | None = None
) -> list[str]:
    """
    Returns the flagged messages matching `reason` or all of the if None
    """

    base_statement = select(SentMessage).where(
        SentMessage.flagged.is_(True)  # pylint: disable=no-member
    )
    if reason:
        base_statement = base_statement.where(
            SentMessage.flagged_response.is_(reason)  # pylint: disable=no-member
        )
    return session.exec(base_statement).all()


def get_bonuses(
    session: Session,
    hours: float | None = None,
    target_bonus: int = TARGET_BONUS,
    persuader_bonus: int = PERSUADER_BONUS,
) -> Counter[str]:
    """
    For each external participant tabulates the number of rounds they have won
    for use in dispersing bonus payments.

    Only counts participants whose `Participant.work_approved is None`.
    A persuader gets a bonus if they convinced the target in the direction
    of their stance. A target gets a bonus if in a factual domain they ended
    up on the “correct” side.
    """
    bonuses: Counter[str] = Counter()

    # 1) select all completed rounds for participants not yet approved
    stmt = (
        select(RoundORM)
        .where(RoundORM.target_initial_belief.is_not(None))  # pylint: disable=no-member
        .where(RoundORM.target_final_belief.is_not(None))  # pylint: disable=no-member
        .join(
            Participant,
            or_(
                RoundORM.persuader_id == Participant.id,
                RoundORM.target_id == Participant.id,
            ),
        )
        .where(Participant.work_approved.is_(None))  # pylint: disable=no-member
    )

    if hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = stmt.where(RoundORM.updated_at >= cutoff)

    rounds = session.exec(stmt).all()

    for rd in rounds:
        # persuader bonus
        if rd.persuader_id is not None and rd.target_persuaded():
            ext = session.get(ExternalUser, rd.persuader_id)
            if ext:
                bonuses[ext.external_id] += rd.target_belief_change() * (
                    persuader_bonus
                )

        # target bonus (factual-only)
        if rd.target_id is not None:
            try:
                if rd.target_answered_correctly():
                    ext = session.get(ExternalUser, rd.target_id)
                    if ext:
                        bonuses[ext.external_id] += target_bonus
            except ValueError:
                # non-factual round, or not finished—skip
                pass

    return bonuses


def get_last_sent_message(
    session: Session, is_target: bool | None, round_id: int
) -> SentMessage | None:
    """
    Returns the last message sent by target or persuader in `round_id` using the `session` or None.
    If is_target is None just returns the last
    """
    statement = (
        select(SentMessage)
        .where(SentMessage.round_id == round_id)
        .where(SentMessage.flagged.is_(False))  # pylint: disable=no-member
    )
    if is_target is not None:
        statement = statement.where(SentMessage.is_target == is_target)

    statement = statement.order_by(
        SentMessage.created_at.desc()  # pylint: disable=no-member
    )

    return session.exec(statement).first()


def get_user_messages(
    session: Session, round_id: int, is_target: bool
) -> list[dict[str, Any]]:
    """
    Return user-facing message history without loading audio payloads.

    Parameters:
        session: Active database session.
        round_id: Round identifier to fetch messages for.
        is_target: Whether the requesting user is the target.

    Returns:
        A list of dicts with message text and sender labels.
    """
    statement = (
        select(
            SentMessage.message_content,
            SentMessage.is_target,
            SentMessage.created_at,
        )
        .where(SentMessage.round_id == round_id)
        .where(SentMessage.flagged.is_(False))  # pylint: disable=no-member
        .where(SentMessage.message_content.is_not(None))  # pylint: disable=no-member
        .where(SentMessage.message_content != "")
        .order_by(asc(SentMessage.created_at), asc(SentMessage.is_target))
    )
    rows = session.exec(statement).all()

    messages: list[dict[str, Any]] = []
    for row in rows:
        message_content, message_is_target, created_at = row
        messages.append(
            {
                "text": message_content,
                "is_target": message_is_target,
                "created_at": created_at,
            }
        )

    for prev, curr in zip(messages, messages[1:]):
        if prev["is_target"] == curr["is_target"]:
            message = (
                f"Round {round_id!r}: messages at "
                f"{prev['created_at']!r}/{curr['created_at']!r} "
                f"did not alternate (both is_target={curr['is_target']})"
            )
            logger.error(message)

    user_messages: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        ours = msg["is_target"] == is_target
        # Ignore the message if it is the last one and not ours
        # and no other message has been received.
        if i == len(messages) - 1 and not ours:
            continue
        user_messages.append(
            {
                "text": msg["text"],
                "sender": "You" if ours else "Agent",
            }
        )
    return user_messages


def populate_tables(
    session: Session,
    propositions_filenames: list[str],
    conditions: Iterable[Condition] = None,
):
    """
    Populates the propositions table from one or more JSONL files.

    - propositions_filename may be a single path or a list of paths.

    Errors if the passed conditions do not match any of the passed propositions.
    """

    # track exact combinations to enforce stricter matching
    required_combos = set()
    seen_combos = set()

    if conditions:
        for c in conditions:
            # Normalize to match how propositions are normalized
            cd = False if c.control_dialogue is None else bool(c.control_dialogue)
            fd = c.factual_domain
            pic = c.proposition_is_correct  # None stays None

            required_combos.add((cd, fd, pic))

            if (
                cd
            ):  # All control dialogues require "first order" propositions that are not control
                required_combos.add((False, fd, pic))

    for filename in propositions_filenames:
        df = pd.read_json(filename, lines=True)
        for _, row in df.iterrows():
            row_dict = row.to_dict()

            # Check if already exists
            existing = session.get(Proposition, row_dict["id"])

            # Normalize nullable fields
            if pd.isnull(row_dict.get("proposition_is_correct")):
                row_dict["proposition_is_correct"] = None
            if pd.isnull(row_dict.get("control_dialogue")):
                row_dict["control_dialogue"] = False
            if pd.isnull(row_dict.get("original_text")):
                row_dict["original_text"] = None
            if pd.isnull(row_dict.get("proposition_source")):
                row_dict["proposition_source"] = None

            # Ensure we only pass expected columns to the model
            # AND omit None values for JSON columns to ensure they are stored as SQL NULL
            # rather than the JSON literal string 'null'
            prop_data = {
                k: v
                for k, v in row_dict.items()
                if k in Proposition.model_fields and v is not None
            }

            # Record combination for stricter matching
            seen_combos.add(
                (
                    row_dict.get("control_dialogue"),
                    row_dict.get("factual_domain"),
                    row_dict.get("proposition_is_correct"),
                )
            )

            if not existing:
                proposition = Proposition(**prop_data)
                session.add(proposition)

    if conditions:
        # verify each exact condition combination appears in propositions
        missing_combos = required_combos - seen_combos
        if missing_combos:
            raise ValueError(
                f"No proposition loaded matching exact condition combination(s): {missing_combos}"
            )

    session.commit()


def get_participant_rounds(
    participant: Participant, session: Session
) -> list[RoundORM] | None:
    """
    For the given participant, returns, using `session`, a list of rounds which the participant
    appears in or None
    """
    statement = select(RoundORM).where(
        or_(
            RoundORM.persuader_id == participant.id,
            RoundORM.target_id == participant.id,
        )
    )
    return session.exec(statement).all()


def get_paired_participant(
    participant: Participant,
    session: Session,
    settings: ServerSettings,
) -> Participant | None:
    """
    Returns a participant for this `participant` to play with, using `session`.
    Only returns participants who have been in the waiting room for less than
    `settings.waiting_room_timeout`.
    """
    if not participant.entered_waiting_room:
        return None

    n_minutes_ago = datetime.now(timezone.utc) - settings.waiting_room_timeout

    chosen_participant = None

    # Make the relevant sql query
    statement = (
        select(Participant)
        .where(Participant.id != participant.id)
        .where(
            Participant.entered_waiting_room.is_not(None)  # pylint: disable=no-member
        )
        .where(Participant.entered_waiting_room >= n_minutes_ago)
        .with_for_update(skip_locked=True)
    )

    if participant.role:
        if participant.role == "persuader":
            statement = statement.where(
                or_(
                    Participant.role.is_(None),  # pylint: disable=no-member
                    Participant.role != "persuader",
                )
            )
        else:  # participant.role == 'target'
            statement = statement.where(
                or_(
                    Participant.role.is_(None),  # pylint: disable=no-member
                    Participant.role != "target",
                )
            )

    # Don't let participants play with each other again
    targets = (
        select(RoundORM.persuader_id)
        .where(RoundORM.persuader_id.is_not(None))  # pylint: disable=no-member
        .where(RoundORM.target_id == participant.id)
    )
    persuaders = (
        select(RoundORM.target_id)
        .where(RoundORM.target_id.is_not(None))  # pylint: disable=no-member
        .where(RoundORM.persuader_id == participant.id)
    )
    statement = statement.where(
        Participant.id.not_in(targets)  # pylint: disable=no-member
    ).where(
        Participant.id.not_in(persuaders)  # pylint: disable=no-member
    )

    # The other participant must be in the same condition as this participant, if assigned
    if participant.condition is not None:
        statement = statement.where(
            or_(
                (Participant.condition == participant.condition),
                (Participant.condition.is_(None)),  # pylint: disable=no-member
            )
        )

    participants = session.exec(statement).all()

    if participants:

        chosen_participant = participants[0]
    return chosen_participant


def round_types_count(
    session: Session, include_short_rounds: bool = False
) -> TypeCounter[Condition]:
    """
    Returns a `Counter` over all of the seen `Condition`s.

    - include_short_rounds: Whether to include the rounds that are too short, etc.
    """
    # NB: we only count completed games which could result in too many games of a certain type
    # but that isn't so bad
    condition_to_rounds = rounds_by_condition(
        session, include_short_rounds=include_short_rounds
    )

    condition_counter = Counter()

    for c, rds in condition_to_rounds.items():
        condition_counter[c.as_non_id_role()] += len(rds)

    return condition_counter


#### API helpers


def get_round_types_remaining(
    session: Session, condition_num_rounds: TypeCounter[Condition]
) -> (set[Condition], set[Condition]):
    """
    In the given `session`, tabulates the remaining round types given the rounds played
    (e.g. from `condition_num_rounds`).
    Returns
    - set[Condition]: the Conditions yet to be filled for paired rounds
    - set[Condition]: the Conditions yet to be filled for non-paired rounds
    """
    current_count = round_types_count(session)
    remaining_round_counts = condition_num_rounds - current_count
    non_paired_conditions = set()
    paired_conditions = set()

    for condition in remaining_round_counts.keys():
        if condition.roles.is_paired_human():
            paired_conditions.add(condition)
        else:
            non_paired_conditions.add(condition)

    return paired_conditions, non_paired_conditions


def choose_condition(
    participant: Participant,
    chosen_participant: Participant,
    paired_rounds_remaining_set: set[Condition],
    non_paired_rounds_remaining_set: set[Condition],
    overassign_non_paired_conditions: bool,
    condition_num_rounds: TypeCounter[Condition],
    waiting_room_timeout: timedelta,
) -> Condition:
    """
    Chooses the condition to slot this `participant` into, based on the current counts of
    completed rounds. Returns the name of the condition type.
    Returns None if no condition chosen (the participant should wait)
    """
    if not participant.entered_waiting_room:
        raise ValueError("Participant must be in waiting room")

    chosen_condition = None

    participant_waiting = participant.waiting_time()

    if len(non_paired_rounds_remaining_set) == 0 and not paired_rounds_remaining_set:
        raise ValueError("No more rounds to assign")

    if participant.condition:
        # If we can, have this player play this condition
        chosen_condition = Condition(**participant.condition)

        # Tell them to wait b/c no participant
        if chosen_condition.roles.is_paired_human() and not chosen_participant:
            chosen_condition = None

    else:
        # For a non paired round there must be non paired rounds left
        # or we will overstuff non paired rounds
        has_non_paired_roles = bool(
            non_paired_rounds_remaining_set
            or any(
                not condition.roles.is_paired_human()
                for condition in condition_num_rounds.keys()
            )
        )

        assign_non_paired_round = (
            has_non_paired_roles
            and (
                overassign_non_paired_conditions
                or bool(non_paired_rounds_remaining_set)
            )
            # Then there are
            # 1) no more paired rounds
            # 2) the participant has been waiting too long or
            # 3) we randomly select non paired anyway.
            # (Note that the small probability will be called compounded as the
            # participant waits and repeatedly calls this function.
            # We query about every 2 seconds for 60 seconds so assume 30 queries
            # p(non paired if waiting for full time) = 1 - .95^ 30 = .78
            # but they probably don't wait the full time
            and (
                not paired_rounds_remaining_set
                or participant_waiting > waiting_room_timeout
                or random.random() < 0.01
            )
        )

        assign_paired_round = (
            chosen_participant is not None and not assign_non_paired_round
        )

        assert not (assign_non_paired_round and assign_paired_round)

        if assign_paired_round:
            if chosen_participant.condition:
                chosen_condition = Condition(**chosen_participant.condition)
            else:
                chosen_condition = random.choice(list(paired_rounds_remaining_set))

        elif assign_non_paired_round:

            # Even if we have filled all the slots, we simply add more participants
            # Make sure not to assign to a paired human role from the main condition
            non_paired_roles = set(
                filter(
                    lambda c: not c.roles.is_paired_human(), condition_num_rounds.keys()
                )
            )

            if len(non_paired_rounds_remaining_set) > 0:
                # We have not yet filled the necessary slots
                non_paired_roles = non_paired_rounds_remaining_set

            possible_roles = list(
                filter(
                    lambda condition: (not participant.role)
                    or (condition.roles.human_target and participant.role == "target")
                    or (
                        condition.roles.human_persuader
                        and participant.role == "persuader"
                    ),
                    non_paired_roles,
                )
            )

            # The below really should not happen
            if len(possible_roles) < 1:
                raise ValueError("No roles left to assign")

            chosen_condition = random.choice(possible_roles)

    # Change the participants to relfect the condition
    if chosen_condition:
        if participant.condition:
            assert Condition(**participant.condition) == chosen_condition
        else:
            participant.condition = chosen_condition.model_dump()

        if chosen_participant:
            if chosen_participant.condition:
                assert Condition(**chosen_participant.condition) == chosen_condition
            else:
                chosen_participant.condition = chosen_condition.model_dump()

    return chosen_condition


def choose_proposition_for_participants(
    p1: Participant,
    p2: Participant | None,
    session: Session,
    control_dialogue: bool = False,
) -> Proposition:
    """
    Chooses a proposition that has not been seen by either participant and which
    matches their condition.

    control_dialogue: Whether to look for a control dialogue proposition

    """
    all_rounds = get_all_rounds(p1, p2, session)
    assert p1.condition
    if p2:
        assert p1.condition == p2.condition

    condition = Condition(**p1.condition)

    if control_dialogue:
        p_ids = [rd.proposition_control for rd in all_rounds]
    else:
        p_ids = [rd.proposition for rd in all_rounds]

    statement = (
        select(Proposition)
        .where(~Proposition.id.in_(p_ids))  # pylint: disable=no-member
        .where(Proposition.factual_domain == condition.factual_domain)
        .where(Proposition.proposition_is_correct == condition.proposition_is_correct)
        .where(
            Proposition.control_dialogue.is_(  # pylint: disable=no-member
                control_dialogue
            )
        )
    )

    requires_bayes_network = bool(
        condition.roles.simulated_target
        or condition.llm_target_use_bayes_structure
        or (condition.enable_node_belief_survey and not control_dialogue)
    )
    if requires_bayes_network:
        statement = statement.where(
            Proposition.bayesian_network.isnot(None)  # pylint: disable=no-member
        )
    elif (condition.proposition_source is None or control_dialogue) and (
        control_dialogue or not condition.control_dialogue
    ):
        statement = statement.where(
            Proposition.bayesian_network.is_(None)  # pylint: disable=no-member
        )
    if condition.proposition_source is not None and not control_dialogue:
        statement = statement.where(
            Proposition.proposition_source == condition.proposition_source.value
        )
    if condition.participant_proposition and not control_dialogue:
        target_participant = None
        if p1.role == "target":
            target_participant = p1
        elif p2 and p2.role == "target":
            target_participant = p2
        if not target_participant:
            raise ValueError(
                "Target participant is required for participant_proposition"
            )
        statement = statement.where(Proposition.participant_id == target_participant.id)
    elif not control_dialogue:
        statement = statement.where(
            Proposition.participant_id.is_(None)  # pylint: disable=no-member
        )

    possible = session.exec(statement).all()
    if condition.enable_node_belief_survey and not control_dialogue:
        possible = [
            proposition
            for proposition in possible
            if _proposition_has_belief_nodes(proposition)
        ]

    if not possible:
        if condition.enable_node_belief_survey and not control_dialogue:
            raise ValueError(
                "Node-belief survey is enabled, but no matching propositions contain "
                "bayesian_network.belief_nodes."
            )
        raise ValueError("No propositions with desired attributes found")

    return _choose_proposition_by_assignment_frequency(
        possible=possible, session=session, control_dialogue=control_dialogue
    )
