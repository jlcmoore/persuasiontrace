"""
src/tests/api/test_current_round_helpers.py

Author: Jared Moore
Date: July, 2025

Tests for current round helpers
"""

import pytest

from api.current_round_helpers import assign_participants, choose_participant_conditions
from api.sql_model import Participant
from experiment.condition import PAIRED_HUMAN_ROLE, Condition, Roles

# Helpers to build minimal Conditions
NON_PAIRED_HUMAN_TARGET = Condition(
    roles=Roles(llm_persuader="gpt-4o", human_target=True),
    factual_domain=False,
)
NON_PAIRED_HUMAN_PERSUADER = Condition(
    roles=Roles(human_persuader=True, llm_target="gpt-4o"),
    factual_domain=False,
)


def make_participant(pid: int, role=None):
    """Quick constructor for Participant with just id and optional role."""
    return Participant(id=pid, role=role)


def test_paired_with_nonpaired_condition_reject():
    p1 = make_participant(1)
    p2 = make_participant(2)
    # non-paired condition but two participants -> should assert
    with pytest.raises(AssertionError):
        choose_participant_conditions(p1, p2, NON_PAIRED_HUMAN_TARGET)


def test_non_paired_with_paired_condition_reject():
    p1 = make_participant(1)
    # paired condition but only one participant -> should assert
    with pytest.raises(AssertionError):
        choose_participant_conditions(
            p1, None, Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False)
        )


def test_non_paired_assign_target():
    p1 = make_participant(1)
    # no second, condition says human_target
    choose_participant_conditions(p1, None, NON_PAIRED_HUMAN_TARGET)
    assert p1.role == "target"
    # assign_participants should reflect that
    persuader_id, target_id = assign_participants(p1, None)
    assert persuader_id is None
    assert target_id == 1


def test_non_paired_assign_persuader():
    p1 = make_participant(5)
    choose_participant_conditions(p1, None, NON_PAIRED_HUMAN_PERSUADER)
    assert p1.role == "persuader"
    pid, tid = assign_participants(p1, None)
    assert pid == 5
    assert tid is None


def test_non_paired_reject_paired_condition():
    p1 = make_participant(1)
    with pytest.raises(AssertionError):
        # PAIRED_HUMAN_ROLE is a paired condition; second=None should assert
        choose_participant_conditions(
            p1,
            None,
            Condition(
                roles=PAIRED_HUMAN_ROLE,
                factual_domain=False,
            ),
        )


def test_paired_both_unassigned():
    p1 = make_participant(10)
    p2 = make_participant(20)
    # paired human–human
    cond = Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False)
    choose_participant_conditions(p1, p2, cond)
    # by default first=>persuader, second=>target
    assert p1.role == "persuader"
    assert p2.role == "target"
    pid, tid = assign_participants(p1, p2)
    assert pid == 10 and tid == 20


def test_paired_existing_second():
    # if second already has a role, first inverts it
    p1 = make_participant(3)
    p2 = make_participant(4, role="target")
    cond = Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False)
    choose_participant_conditions(p1, p2, cond)
    # second was target => first becomes persuader
    assert p2.role == "target"
    assert p1.role == "persuader"
    # assign_participants
    pid, tid = assign_participants(p1, p2)
    assert pid == 3 and tid == 4


def test_paired_existing_first():
    # if first already has a role, second gets the opposite
    p1 = make_participant(7, role="target")
    p2 = make_participant(8)
    cond = Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False)
    choose_participant_conditions(p1, p2, cond)
    assert p1.role == "target"
    assert p2.role == "persuader"
    pid, tid = assign_participants(p1, p2)
    assert pid == 8 and tid == 7


def test_assign_participants_missing_role():
    p1 = make_participant(1)  # no role
    with pytest.raises(AssertionError):
        assign_participants(p1, None)


def test_assign_participants_both_missing_one_role():
    # first has role, second missing => still works
    p1 = make_participant(11, role="persuader")
    p2 = make_participant(22)
    with pytest.raises(AssertionError):
        # since second.role is None, the function won't get to assert valid pair,
        # but the very first assert( first.role ) passes, then assert not second or second.role
        # So it bails when trying to assign target/persuader? Actually the second assert
        # assert not second or second.role will fail, so we get AssertionError.
        assign_participants(p1, p2)
