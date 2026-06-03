"""
src/tests/api/test_sql_queries.py

Author: Jared Moore
Date: July, 2025

Tests for the sql database queries.
"""

import json
import random
import time
from collections import Counter
from datetime import datetime, timezone

import pytest
from sqlmodel import select

from api.sql_model import ExternalUser, Participant, Proposition, RoundORM, SentMessage
from api.sql_queries import (
    choose_condition,
    choose_proposition_for_participants,
    get_bonuses,
    get_last_sent_message,
    get_paired_participant,
    get_participant_rounds,
    get_round_types_remaining,
    populate_tables,
    round_types_count,
    rounds_by_condition,
)
from api.utils import DEFAULT_WAITING_ROOM_TIMEOUT
from experiment.condition import (
    PAIRED_HUMAN_ROLE,
    Condition,
    PropositionSource,
    Roles,
)
from experiment.utils import EXAMPLE_PROPOSITIONS_FILE, get_data_file_path

from .context import TEST_SETTINGS, engine_fixture, session_fixture

# # Clean slate before each test
# @pytest.fixture(autouse=True)
# def _cleanup(session, engine):
#     SQLModel.metadata.drop_all(engine)
#     session.commit()
#     yield
#     session.rollback()


def test_get_participant_rounds(session):
    user = ExternalUser(external_id="u1")
    session.add(user)
    session.commit()
    session.refresh(user)
    par = Participant(id=user.id, role=None, condition=None)
    session.add(par)
    session.commit()

    # no rounds yet
    assert get_participant_rounds(par, session) == []

    # need a proposition for the FK
    prop = Proposition(id="p1", factual_domain=True, proposition_is_correct=True)
    session.add(prop)
    session.commit()

    # as persuader
    rd1 = RoundORM(proposition=prop.id, persuader_id=par.id, llm_target="gpt-4o")
    session.add(rd1)
    session.commit()
    session.refresh(rd1)
    assert rd1 in get_participant_rounds(par, session)

    # as target
    rd2 = RoundORM(proposition=prop.id, target_id=par.id, llm_persuader="gpt-4o")
    session.add(rd2)
    session.commit()
    session.refresh(rd2)
    rds = get_participant_rounds(par, session)
    assert rd1 in rds and rd2 in rds


def test_populate_tables_and_no_dupes(session):
    file = get_data_file_path(f"{EXAMPLE_PROPOSITIONS_FILE}.jsonl")
    populate_tables(session, propositions_filenames=[file])
    first = session.exec(select(Proposition)).all()
    assert len(first) > 0

    # run again, no duplicates
    populate_tables(session, propositions_filenames=[file])
    second = session.exec(select(Proposition)).all()
    assert len(second) == len(first)


def test_populate_tables_handles_nan_original_text(session, tmp_path):
    file = tmp_path / "props_nan_text.jsonl"
    row = {
        "id": "prop_nan_text",
        "factual_domain": False,
        "proposition_is_correct": None,
        "control_dialogue": False,
        "original_text": float("nan"),
    }
    with open(file, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    populate_tables(session, propositions_filenames=[str(file)])
    proposition = session.get(Proposition, "prop_nan_text")
    assert proposition is not None
    assert proposition.original_text is None


def test_get_last_sent_message(session):
    prop = Proposition(id="plast", factual_domain=True, proposition_is_correct=True)
    session.add(prop)
    user = ExternalUser(external_id="u2")
    session.add(user)
    session.commit()
    session.refresh(prop)
    session.refresh(user)

    rd = RoundORM(proposition=prop.id, persuader_id=user.id, target_id=user.id)
    session.add(rd)
    session.commit()
    session.refresh(rd)

    # two target messages
    tm1 = SentMessage(
        audio="",
        transcript=None,
        message_content="one",
        thought_content=None,
        reasoning_trace=None,
        flagged=False,
        flagged_response=None,
        is_target=True,
        round_id=rd.id,
    )
    session.add(tm1)
    session.commit()
    time.sleep(1)
    tm2 = SentMessage(
        audio="",
        transcript=None,
        message_content="two",
        thought_content=None,
        reasoning_trace=None,
        flagged=False,
        flagged_response=None,
        is_target=True,
        round_id=rd.id,
    )
    session.add(tm2)
    session.commit()

    last = get_last_sent_message(session, is_target=True, round_id=rd.id)
    assert last.id == tm2.id
    assert last.message_content == "two"


def test_round_types_count_and_remaining(session):
    # no rounds
    assert round_types_count(session) == Counter()

    # insert two rounds under distinct conditions
    prop = Proposition(id="pcnt", factual_domain=True, proposition_is_correct=False)
    session.add(prop)
    session.commit()

    # paired human persuaded to agree
    rd1 = RoundORM(
        proposition=prop.id,
        persuader_id=1,
        target_id=2,
        target_initial_belief=0.3,
        target_final_belief=0.8,
        persuader_supports_proposition=True,
    )
    # llm persuader, human target, convinced negatively
    rd2 = RoundORM(
        proposition=prop.id,
        llm_persuader="gpt-4o-2024-08-06",
        target_id=3,
        target_initial_belief=0.7,
        target_final_belief=0.4,
        persuader_supports_proposition=False,
    )
    session.add_all([rd1, rd2])
    session.commit()

    cnts = round_types_count(session)
    cond1 = Condition(
        roles=Roles(human_persuader=True, human_target=True),
        factual_domain=True,
        proposition_is_correct=False,
    )
    cond2 = Condition(
        roles=Roles(llm_persuader="gpt-4o-2024-08-06", human_target=True),
        factual_domain=True,
        proposition_is_correct=False,
    )
    assert cnts == Counter({cond1: 1, cond2: 1})

    # remaining to run: want 2 of cond1, 1 of cond2
    paired_rem, _ = get_round_types_remaining(
        session,
        condition_num_rounds=Counter({cond1: 2, cond2: 1}),
    )
    assert cond1 in paired_rem
    assert cond2 not in paired_rem


def test_rounds_by_condition_excludes_and_includes_short_quick(session):
    prop = Proposition(id="pshort", factual_domain=False)
    session.add(prop)
    session.commit()

    # a “completed” round
    rd = RoundORM(
        proposition=prop.id,
        persuader_id=10,
        target_id=20,
        target_initial_belief=0.2,
        target_final_belief=0.8,
        persuader_supports_proposition=True,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    # insert one very short message each side
    for is_target in (False, True):
        m = SentMessage(
            audio="",
            transcript={
                "role": "target" if is_target else "persuader",
                "content": "short",
            },
            message_content="short",  # length=5 <10 -> too_short
            thought_content=None,
            reasoning_trace=None,
            flagged=False,
            flagged_response=None,
            is_target=is_target,
            round_id=rd.id,
        )
        session.add(m)
    session.commit()

    # by default, exclude short rounds
    grouping = rounds_by_condition(session, include_short_rounds=False)
    assert not grouping

    # but include when asked
    grouping2 = rounds_by_condition(session, include_short_rounds=True)
    assert len(grouping2) == 1
    cond = next(iter(grouping2))
    assert cond.roles.is_paired_human()
    assert len(grouping2[cond]) == 1


def test_get_bonuses(session):
    # persuader bonus
    u = ExternalUser(external_id="B1")
    session.add(u)
    session.commit()
    p = Participant(id=u.id, role="persuader", work_approved=None)
    session.add(p)
    session.commit()

    prop = Proposition(id="pb", factual_domain=True, proposition_is_correct=True)
    session.add(prop)
    session.commit()

    rd = RoundORM(
        proposition=prop.id,
        persuader_id=p.id,
        llm_target="test",
        target_initial_belief=0.1,
        target_final_belief=0.9,
        persuader_supports_proposition=True,
    )
    session.add(rd)
    session.commit()

    bonuses = get_bonuses(session)
    assert bonuses["B1"] == 4

    # target bonus
    u2 = ExternalUser(external_id="B2")
    session.add(u2)
    session.commit()
    p2 = Participant(id=u2.id, role="target", work_approved=None)
    session.add(p2)
    session.commit()

    rd2 = RoundORM(
        proposition=prop.id,
        llm_persuader="gpt-4o-2024-08-06",
        target_id=p2.id,
        target_initial_belief=0.2,
        target_final_belief=0.8,
        persuader_supports_proposition=True,
    )
    session.add(rd2)
    session.commit()

    bonuses2 = get_bonuses(session)
    assert bonuses2["B2"] == 1


def test_get_paired_participant_basic(session):
    u1 = ExternalUser(external_id="A")
    u2 = ExternalUser(external_id="B")
    session.add_all([u1, u2])
    session.commit()
    session.refresh(u1)
    session.refresh(u2)

    p1 = Participant(id=u1.id, role=None)
    p2 = Participant(id=u2.id, role=None)
    session.add_all([p1, p2])
    session.commit()

    # not yet in waiting room
    assert get_paired_participant(p1, session, TEST_SETTINGS) is None

    # both enter waiting
    now = datetime.now(timezone.utc)
    p1.entered_waiting_room = now
    p2.entered_waiting_room = now
    session.add_all([p1, p2])
    session.commit()

    pair = get_paired_participant(p1, session, TEST_SETTINGS)
    assert pair == p2
    prop = Proposition(id="cr", factual_domain=False)
    session.add(prop)
    session.commit()
    rd = RoundORM(proposition=prop.id, persuader_id=p1.id, target_id=p2.id)
    session.add(rd)
    session.commit()

    # without rematch, a second call returns None
    assert get_paired_participant(p1, session, TEST_SETTINGS) is None


def test_paired_participant_condition_and_rematch(session):
    # two users/participants
    u1 = ExternalUser(external_id="C1")
    u2 = ExternalUser(external_id="C2")
    session.add_all([u1, u2])
    session.commit()
    p1 = Participant(id=u1.id, role=None)
    p2 = Participant(id=u2.id, role=None)
    session.add_all([p1, p2])
    session.commit()

    # both enter waiting
    ts = datetime.now(timezone.utc)
    p1.entered_waiting_room = ts
    p2.entered_waiting_room = ts
    session.add_all([p1, p2])
    session.commit()

    # assign different conditions -> no pairing
    c1 = Condition(
        roles=PAIRED_HUMAN_ROLE,
        factual_domain=True,
        proposition_is_correct=True,
    )
    c2 = Condition(
        roles=PAIRED_HUMAN_ROLE,
        factual_domain=False,
        proposition_is_correct=None,
    )
    p1.condition = c1.model_dump()
    p2.condition = c2.model_dump()
    session.add_all([p1, p2])
    session.commit()
    assert get_paired_participant(p1, session, TEST_SETTINGS) is None

    # same condition -> pairs
    p2.condition = c1.model_dump()
    session.add(p2)
    session.commit()
    assert get_paired_participant(p1, session, TEST_SETTINGS) == p2

    # record a round between them -> no rematch
    prop = Proposition(id="cr", factual_domain=False)
    session.add(prop)
    session.commit()
    rd = RoundORM(proposition=prop.id, persuader_id=p1.id, target_id=p2.id)
    session.add(rd)
    session.commit()
    assert get_paired_participant(p1, session, TEST_SETTINGS) is None


def test_choose_condition_basic_paths(session):
    # prepare a participant
    p = Participant(id=500, role=None, condition=None)
    session.add(p)
    session.commit()

    # 1) not in waiting room -> error
    with pytest.raises(ValueError):
        choose_condition(
            participant=p,
            chosen_participant=None,
            paired_rounds_remaining_set={
                Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False)
            },
            non_paired_rounds_remaining_set={
                Condition(roles=Roles(llm_target="gpt-4o", human_persuader=True))
            },
            overassign_non_paired_conditions=True,
            condition_num_rounds=Counter(),
            waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
        )

    # 2) both sets empty -> error
    p.entered_waiting_room = datetime.now(timezone.utc)
    session.add(p)
    session.commit()
    with pytest.raises(ValueError):
        choose_condition(
            participant=p,
            chosen_participant=None,
            paired_rounds_remaining_set=set(),
            non_paired_rounds_remaining_set=set(),
            overassign_non_paired_conditions=True,
            condition_num_rounds=Counter(),
            waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
        )

    # 3) small wait, paired_only -> returns None
    cond_pair = Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False)
    cond_non = Condition(
        roles=Roles(llm_target="gpt-4o", human_persuader=True), factual_domain=False
    )
    res = choose_condition(
        participant=p,
        chosen_participant=None,
        paired_rounds_remaining_set={cond_pair},
        non_paired_rounds_remaining_set={cond_non},
        overassign_non_paired_conditions=True,
        condition_num_rounds=Counter({cond_non: 1}),
        waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
    )
    assert res is None

    # 4) waited long enough -> picks non-paired
    p.entered_waiting_room = (
        datetime.now(timezone.utc) - 2 * DEFAULT_WAITING_ROOM_TIMEOUT
    )
    session.add(p)
    session.commit()

    res2 = choose_condition(
        participant=p,
        chosen_participant=None,
        paired_rounds_remaining_set={cond_pair},
        non_paired_rounds_remaining_set={cond_non},
        overassign_non_paired_conditions=True,
        condition_num_rounds=Counter({cond_non: 1}),
        waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
    )
    assert res2 == cond_non

    # 5) participant.role filtering -> only roles matching their fixed role
    p.role = "target"
    p.condition = None
    session.add(p)
    session.commit()
    bad = Condition(
        roles=Roles(llm_persuader="gpt-4o", llm_target="claude"), factual_domain=False
    )
    good = Condition(
        roles=Roles(llm_persuader="gpt-4o", human_target=True), factual_domain=False
    )
    # waited long
    p.entered_waiting_room = (
        datetime.now(timezone.utc) - 2 * DEFAULT_WAITING_ROOM_TIMEOUT
    )
    session.add(p)
    session.commit()

    # only 'good' remains
    with pytest.raises(ValueError):
        # if only bad is in set, error
        choose_condition(
            participant=p,
            chosen_participant=None,
            paired_rounds_remaining_set=set(),
            non_paired_rounds_remaining_set={bad},
            overassign_non_paired_conditions=False,
            condition_num_rounds=Counter({bad: 0}),
            waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
        )

    # when good is offered, picks it
    chosen = choose_condition(
        participant=p,
        chosen_participant=None,
        paired_rounds_remaining_set=set(),
        non_paired_rounds_remaining_set={good},
        overassign_non_paired_conditions=True,
        condition_num_rounds=Counter({good: 1}),
        waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
    )
    assert chosen == good


def _make_participant(session, pid: int, condition: dict) -> Participant:
    """
    Create a Participant with a given id and assigned condition.
    """
    p = Participant(id=pid, role=None, condition=condition)
    session.add(p)
    session.commit()
    session.refresh(p)
    session.expunge(p)
    return p


@pytest.fixture(autouse=True)
def fixed_random_seed():
    # ensure deterministic choice when only one proposition is available
    random.seed(0)


def test_single_participant_gets_only_matching_proposition(session):
    # create two propositions in the DB matching factual_domain=False
    prop1 = Proposition(id="pA", factual_domain=False, proposition_is_correct=None)
    prop2 = Proposition(id="pB", factual_domain=False, proposition_is_correct=None)
    session.add_all([prop1, prop2])
    session.commit()

    # participant condition: non-paired, any persuader role (doesn't matter here)
    cond = Condition(
        roles=Roles(human_persuader=True, llm_target="gpt-4o"), factual_domain=False
    )
    p1 = _make_participant(session, pid=1, condition=cond.model_dump())

    # first call should return either pA or pB
    chosen = choose_proposition_for_participants(p1, None, session)
    assert chosen.id in {"pA", "pB"}

    # record a round so that p1 has seen chosen.id
    rd = RoundORM(
        proposition=chosen.id,
        persuader_id=p1.id,
    )
    session.add(rd)
    session.commit()

    # the other proposition should be returned now
    other = "pB" if chosen.id == "pA" else "pA"
    chosen2 = choose_proposition_for_participants(p1, None, session)
    assert chosen2.id == other


def test_node_belief_survey_requires_belief_nodes(session):
    """When node survey is enabled, propositions must include belief nodes."""
    prop = Proposition(
        id="bn_missing_nodes",
        factual_domain=False,
        proposition_is_correct=None,
        bayesian_network={"edges": []},
    )
    session.add(prop)
    session.commit()

    cond = Condition(
        roles=Roles(llm_persuader="gpt-4o", human_target=True),
        factual_domain=False,
        enable_node_belief_survey=True,
    )
    p1 = _make_participant(session, pid=31, condition=cond.model_dump())

    with pytest.raises(ValueError, match="belief_nodes"):
        choose_proposition_for_participants(p1, None, session)


def test_proposition_source_can_select_bn_without_node_survey(session):
    """Pinned proposition_source may sample BN-backed propositions without survey."""
    prop_source_bn = Proposition(
        id="bn_from_source",
        factual_domain=False,
        proposition_is_correct=None,
        proposition_source=PropositionSource.DEBATEGPT.value,
        bayesian_network={"edges": []},
    )
    prop_non_source_plain = Proposition(
        id="plain_other_source",
        factual_domain=False,
        proposition_is_correct=None,
        proposition_source=PropositionSource.PPT.value,
        bayesian_network=None,
    )
    session.add_all([prop_source_bn, prop_non_source_plain])
    session.commit()

    cond = Condition(
        roles=Roles(llm_persuader="gpt-4o", human_target=True),
        factual_domain=False,
        proposition_source=PropositionSource.DEBATEGPT,
        enable_node_belief_survey=False,
        llm_target_use_bayes_structure=False,
    )
    p1 = _make_participant(session, pid=33, condition=cond.model_dump())

    chosen = choose_proposition_for_participants(p1, None, session)
    assert chosen.id == "bn_from_source"


def test_control_dialogue_base_can_use_bn_without_node_survey(session):
    """Control rounds may use BN-backed base propositions when needed."""
    base_bn = Proposition(
        id="control_base_bn",
        factual_domain=False,
        proposition_is_correct=None,
        bayesian_network={"edges": []},
    )
    control_prop = Proposition(
        id="control_line",
        factual_domain=False,
        proposition_is_correct=None,
        control_dialogue=True,
    )
    session.add_all([base_bn, control_prop])
    session.commit()

    cond = Condition(
        roles=Roles(llm_persuader="gpt-4o", human_target=True),
        factual_domain=False,
        control_dialogue=True,
        enable_node_belief_survey=False,
        llm_target_use_bayes_structure=False,
    )
    p1 = _make_participant(session, pid=34, condition=cond.model_dump())

    chosen_base = choose_proposition_for_participants(p1, None, session)
    assert chosen_base.id == "control_base_bn"

    chosen_control = choose_proposition_for_participants(
        p1, None, session, control_dialogue=True
    )
    assert chosen_control.id == "control_line"


def test_node_belief_survey_filters_to_valid_belief_nodes(session):
    """Node-survey selection skips BN propositions with empty/missing nodes."""
    prop_bad = Proposition(
        id="bn_bad",
        factual_domain=False,
        proposition_is_correct=None,
        bayesian_network={"belief_nodes": ["   "], "edges": []},
    )
    prop_good = Proposition(
        id="bn_good",
        factual_domain=False,
        proposition_is_correct=None,
        bayesian_network={"belief_nodes": ["A causes B"], "edges": []},
    )
    session.add_all([prop_bad, prop_good])
    session.commit()

    cond = Condition(
        roles=Roles(llm_persuader="gpt-4o", human_target=True),
        factual_domain=False,
        enable_node_belief_survey=True,
    )
    p1 = _make_participant(session, pid=32, condition=cond.model_dump())

    chosen = choose_proposition_for_participants(p1, None, session)
    assert chosen.id == "bn_good"


def test_two_participants_same_condition(session):
    # create two propositions
    prop1 = Proposition(id="X", factual_domain=True, proposition_is_correct=True)
    prop2 = Proposition(id="Y", factual_domain=True, proposition_is_correct=True)
    session.add_all([prop1, prop2])
    session.commit()

    # both share the same condition
    cond = Condition(
        roles=Roles(human_persuader=True, human_target=True),
        factual_domain=True,
        proposition_is_correct=True,
    )
    p1 = _make_participant(session, pid=10, condition=cond.model_dump())
    p2 = _make_participant(session, pid=20, condition=cond.model_dump())

    # first pick: either X or Y
    first = choose_proposition_for_participants(p1, p2, session)
    assert first.factual_domain is True
    assert first.proposition_is_correct is True
    assert first.id in {"X", "Y"}

    # create a Round for p2 as well (to simulate either side) using the same prop
    rd = RoundORM(
        proposition=first.id,
        persuader_id=p1.id,
        target_id=p2.id,
        target_initial_belief=0.0,
        target_final_belief=1.0,
        persuader_supports_proposition=True,
    )
    session.add(rd)
    session.commit()

    # now only the other proposition remains
    second = choose_proposition_for_participants(p1, p2, session)
    assert second.id in {"X", "Y"} and second.id != first.id


def test_two_participants_mismatched_condition_raises(session):
    prop = Proposition(id="Z", factual_domain=False, proposition_is_correct=None)
    session.add(prop)
    session.commit()

    cond1 = Condition(
        roles=Roles(human_persuader=True, human_target=True), factual_domain=False
    )
    cond2 = Condition(
        roles=Roles(human_persuader=True, human_target=True),
        factual_domain=True,
        proposition_is_correct=True,
    )
    p1 = _make_participant(session, pid=5, condition=cond1.model_dump())
    p2 = _make_participant(session, pid=6, condition=cond2.model_dump())

    with pytest.raises(AssertionError):
        _ = choose_proposition_for_participants(p1, p2, session)


def test_no_matching_propositions_raises(session):
    # only one proposition exists
    prop = Proposition(id="ONLY", factual_domain=True, proposition_is_correct=False)
    session.add(prop)
    session.commit()

    cond = Condition(
        roles=Roles(human_persuader=True, human_target=True),
        factual_domain=True,
        proposition_is_correct=False,
        persuader_supports_proposition=False,
    )
    p1 = _make_participant(session, pid=7, condition=cond.model_dump())
    # p1 sees that proposition in a prior round
    rd = RoundORM(
        proposition=prop.id,
        persuader_id=p1.id,
        target_id=None,
        target_initial_belief=0.0,
        target_final_belief=1.0,
        persuader_supports_proposition=False,
    )
    session.add(rd)
    session.commit()

    with pytest.raises(ValueError):
        _ = choose_proposition_for_participants(p1, None, session)


def test_choose_proposition_weights_toward_less_assigned(session, monkeypatch):
    """
    Proposition sampling should weight less-assigned propositions higher.
    """
    prop_low = Proposition(
        id="weighted_low",
        factual_domain=False,
        proposition_is_correct=None,
        control_dialogue=False,
    )
    prop_high = Proposition(
        id="weighted_high",
        factual_domain=False,
        proposition_is_correct=None,
        control_dialogue=False,
    )
    session.add_all([prop_low, prop_high])

    participant = _make_participant(
        session,
        pid=801,
        condition=Condition(
            roles=Roles(human_persuader=True, llm_target="gpt-4o"),
            factual_domain=False,
        ).model_dump(),
    )

    history_participant = Participant(id=802, role=None, condition=None)
    session.add(history_participant)
    session.commit()

    for _ in range(5):
        session.add(
            RoundORM(
                proposition=prop_high.id,
                persuader_id=history_participant.id,
                llm_target="gpt-4o",
            )
        )
    session.commit()

    captured_weights = {}

    def _fake_choices(population, *, weights, k):
        captured_weights.update(
            {proposition.id: weight for proposition, weight in zip(population, weights)}
        )
        return [population[0]]

    monkeypatch.setattr("api.sql_queries.random.choices", _fake_choices)

    _ = choose_proposition_for_participants(participant, None, session)
    assert captured_weights["weighted_low"] > captured_weights["weighted_high"]


def test_control_dialogue_weighting_uses_control_assignment_counts(
    session, monkeypatch
):
    """
    Control-dialogue sampling should weight by proposition_during_round history.
    """
    base_prop = Proposition(
        id="base_non_control",
        factual_domain=False,
        proposition_is_correct=None,
        control_dialogue=False,
    )
    control_low = Proposition(
        id="control_weighted_low",
        factual_domain=False,
        proposition_is_correct=None,
        control_dialogue=True,
    )
    control_high = Proposition(
        id="control_weighted_high",
        factual_domain=False,
        proposition_is_correct=None,
        control_dialogue=True,
    )
    session.add_all([base_prop, control_low, control_high])

    participant = _make_participant(
        session,
        pid=811,
        condition=Condition(
            roles=Roles(human_persuader=True, llm_target="gpt-4o"),
            factual_domain=False,
            control_dialogue=True,
        ).model_dump(),
    )

    history_participant = Participant(id=812, role=None, condition=None)
    session.add(history_participant)
    session.commit()

    for _ in range(6):
        session.add(
            RoundORM(
                proposition=base_prop.id,
                proposition_during_round=control_high.id,
                control_dialogue=True,
                persuader_id=history_participant.id,
                llm_target="gpt-4o",
            )
        )
    session.commit()

    captured_weights = {}

    def _fake_choices(population, *, weights, k):
        captured_weights.update(
            {proposition.id: weight for proposition, weight in zip(population, weights)}
        )
        return [population[0]]

    monkeypatch.setattr("api.sql_queries.random.choices", _fake_choices)

    _ = choose_proposition_for_participants(
        participant, None, session, control_dialogue=True
    )
    assert (
        captured_weights["control_weighted_low"]
        > captured_weights["control_weighted_high"]
    )
