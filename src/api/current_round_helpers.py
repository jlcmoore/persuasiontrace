"""
src/api/current_round_helpers.py

Author: Jared Moore
Date: July, 2025

Helpers for assigning participants to rounds and conditions.
"""

import logging

from experiment.condition import Condition

from .sql_model import Participant

logger = logging.getLogger(__name__)


def choose_participant_conditions(
    first: Participant,
    second: Participant | None,
    chosen_condition: Condition,
):
    """
    Chooses roles for the participants if not already set. Modifies the objects.
    A helper for `choose_round`.
    """
    if not first.conditions_assigned():
        # First assign the roles
        if second:
            # Only allow paired-human conditions when two players are present
            assert (
                chosen_condition.roles.is_paired_human()
            ), "Received two participants but condition is not paired human"

            # if second already had a role, first inverts it; else default first->persuader
            first.role = "target" if second.role == "persuader" else "persuader"
            if not second.role:
                assert first.role == "persuader"
                second.role = "target"

        else:
            # No second participant: condition must be non-paired
            assert (
                not chosen_condition.roles.is_paired_human()
            ), "Single participant but condition is paired human"
            first.role = (
                "target" if chosen_condition.roles.human_target else "persuader"
            )

    assert first.conditions_assigned()

    if second and not second.conditions_assigned():
        choose_participant_conditions(
            first=second,
            second=first,
            chosen_condition=chosen_condition,
        )
        assert second.conditions_assigned()
        assert first.role != second.role


def assign_participants(
    first: Participant, second: Participant | None
) -> (int | None, int | None):
    """
    Assigns the passed participants to roles for a round. For use in a round.
    Returns the persuader and target ids
    """
    persuader_id = None
    target_id = None

    assert first.role
    assert not second or second.role

    if first.role == "persuader":
        persuader_id = first.id
    else:
        assert first.role == "target"
        target_id = first.id

    if second:
        if second.role == "persuader":
            persuader_id = second.id
        else:
            assert second.role == "target"
            target_id = second.id

    if first and second:
        assert target_id is not None and persuader_id is not None
    else:
        assert target_id is not None or persuader_id is not None

    return persuader_id, target_id
