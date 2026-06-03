"""
# src/tests/api/test_sql_model.py

Author: Jared Moore
Date: July, 2025

Tests for sql model basics.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, select

from api.sql_model import ExternalUser, Participant, Proposition, RoundORM, SentMessage
from experiment.condition import Condition, PropositionSource
from experiment.round import Round as InMemoryRound

from .context import engine_fixture, session_fixture


def test_participant_helpers_wait_and_role(session: Session):
    # Create a participant without entered_waiting_room
    p = Participant(id=100, role="target")
    session.add(p)
    session.commit()
    session.refresh(p)

    # conditions_assigned() should be True if role is set
    assert p.conditions_assigned()

    # is_target() should reflect its role
    assert p.is_target() is True

    # waiting_time() with no entry should be None
    assert p.waiting_time() is None

    # Now set entered_waiting_room in the past
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    p.entered_waiting_room = past.replace(tzinfo=None)  # model stores naive
    session.add(p)
    session.commit()
    session.refresh(p)

    delta = p.waiting_time()
    assert isinstance(delta, timedelta)
    assert delta.total_seconds() >= 5

    # Change role to persuader
    p2 = Participant(id=101, role="persuader")
    session.add(p2)
    session.commit()
    assert p2.is_target() is False


def test_proposition_factual_and_relationship(session: Session):
    # By default factual_domain is True
    prop = Proposition(id="p1", proposition_is_correct=True)
    session.add(prop)
    session.commit()
    session.refresh(prop)
    assert prop.factual_domain is True

    # proposition_is_correct=None should raise
    with pytest.raises(ValueError):
        Proposition(id="p2")  # missing truth value on factual domain

    # Now create a human target and an LLM persuader
    u = ExternalUser(external_id="ext-xyz")
    session.add(u)
    session.commit()
    session.refresh(u)

    # Create a RoundORM tied to this proposition
    rd = RoundORM(
        llm_persuader="gpt-4o-2024-08-06",
        target_id=u.id,
        proposition=prop.id,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    # The round should point back to the proposition object
    assert rd.proposition_obj.id == prop.id  # pylint: disable=no-member
    # And (if SQLModel wiring worked) the prop should see the round in its .rounds
    # Sometimes SQLModel auto-infers the reverse; if not, at least the forward should work:
    rounds = session.exec(select(RoundORM).where(RoundORM.proposition == prop.id)).all()
    assert rd in rounds


def test_roundorm_condition_builder(session: Session):
    # Mix of human and LLM
    u = ExternalUser(external_id="AAA")

    prop = Proposition(id="ptest", proposition_is_correct=False)
    session.add(prop)

    session.add(u)
    session.commit()
    session.refresh(u)

    # human persuader (id) and LLM target
    rd1 = RoundORM(
        persuader_id=u.id,
        llm_target="meta-llama/Llama-2-70b-chat-hf",
        proposition=prop.id,
    )
    session.add(rd1)
    session.commit()
    session.refresh(rd1)

    cond1 = rd1.condition()
    assert isinstance(cond1, Condition)
    # persuader is human
    assert cond1.roles.human_persuader is True
    assert cond1.roles.llm_persuader is None
    # target is LLM
    assert cond1.roles.llm_target == "meta-llama/Llama-2-70b-chat-hf"
    assert cond1.roles.human_target is False

    # LLM persuader and human target
    rd2 = RoundORM(
        llm_persuader="claude-3-5-sonnet-20240620",
        target_id=u.id,
        proposition=prop.id,
        persuader_supports_proposition=True,
    )
    session.add(rd2)
    session.commit()
    session.refresh(rd2)
    cond2 = rd2.condition()
    assert cond2.roles.llm_persuader == "claude-3-5-sonnet-20240620"
    assert cond2.roles.human_target is True
    assert rd2.persuader_supports_proposition


def test_roundorm_condition_builder_node_survey_source(session: Session):
    """RoundORM should reconstruct node-survey conditions with proposition source."""
    prop = Proposition(
        id="ptest_bn",
        proposition_is_correct=False,
        proposition_source="debategpt",
        bayesian_network={"belief_nodes": ["A implies B"], "edges": []},
    )
    session.add(prop)

    user = ExternalUser(external_id="SURVEY-SRC")
    session.add(user)
    session.commit()
    session.refresh(user)

    rd = RoundORM(
        llm_persuader="naive",
        target_id=user.id,
        proposition=prop.id,
        enable_node_belief_survey=True,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    cond = rd.condition()
    assert isinstance(cond, Condition)
    assert cond.enable_node_belief_survey is True
    assert cond.proposition_source == PropositionSource.DEBATEGPT


def test_roundorm_as_round_conversion(session: Session):
    # Prepare proposition record
    prop = Proposition(id="ptest", proposition_is_correct=False)
    session.add(prop)
    # Prepare human persuader
    user = ExternalUser(external_id="BBB")
    session.add(user)
    session.commit()
    session.refresh(prop)
    session.refresh(user)

    # Make a round with a human persuader and human target
    rd = RoundORM(
        persuader_id=user.id,
        target_id=user.id,
        proposition=prop.id,
        persuader_supports_proposition=True,
    )
    # set beliefs & stance so that the Round validator won't complain in-memory
    rd.target_initial_belief = 0.2
    rd.target_final_belief = 0.8
    session.add(rd)
    session.commit()
    session.refresh(rd)

    # Add two non-flagged messages (persuader -> target)
    msg1 = SentMessage(
        audio="",
        transcript={"role": "persuader", "content": "foo"},
        message_content="foo",
        thought_content="think1",
        reasoning_trace="trace1",
        flagged=False,
        flagged_response=None,
        is_target=False,
        round_id=rd.id,
    )
    # message 2
    msg2 = SentMessage(
        audio="",
        transcript={"role": "target", "content": "bar"},
        message_content="bar",
        thought_content=None,
        reasoning_trace=None,
        flagged=False,
        flagged_response=None,
        is_target=True,
        round_id=rd.id,
    )
    session.add_all([msg1, msg2])
    session.commit()
    session.refresh(msg1)
    session.refresh(msg2)

    # Convert to in-memory Round
    mem = rd.as_round()
    assert isinstance(mem, InMemoryRound)
    # Check top-level fields
    assert mem.proposition == prop.id
    assert mem.persuader_id == user.id
    assert mem.target_id == user.id
    assert mem.human_persuader_id == user.id
    assert mem.human_target_id == user.id
    assert mem.target_initial_belief == pytest.approx(0.2)
    assert mem.target_final_belief == pytest.approx(0.8)
    # Check messages, transcripts, CoTs, reasoning_traces
    assert [m["content"] for m in mem.messages] == ["foo", "bar"]
    assert [t["content"] for t in mem.transcripts] == ["foo", "bar"]
    assert [c["content"] for c in mem.chains_of_thought] == ["think1", None]
    assert [r["content"] for r in mem.reasoning_traces] == ["trace1", None]

    # And we can call the in-memory evaluators
    assert mem.finished()
    assert mem.target_persuaded()  # persuaded, moved from .2 to .8 in support
    # It's a factual domain and prop is incorrect
    assert not mem.target_answered_correctly()  # .8 > .5 but proposition is false


def test_roundorm_messages_relationship(session: Session):
    # Test that rd.messages returns the SentMessage objects
    prop = Proposition(id="p-rel", proposition_is_correct=True)
    session.add(prop)
    user = ExternalUser(external_id="CCC")
    session.add(user)
    session.commit()
    session.refresh(prop)
    session.refresh(user)

    rd = RoundORM(
        persuader_id=user.id,
        target_id=user.id,
        proposition=prop.id,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    # Initially no messages
    assert rd.messages == []

    # add a new message
    m = SentMessage(
        audio="",
        transcript={"role": "persuader", "content": "hello"},
        message_content="hello",
        thought_content=None,
        reasoning_trace=None,
        flagged=False,
        flagged_response=None,
        is_target=False,
        round_id=rd.id,
    )
    session.add(m)
    session.commit()
    session.refresh(rd)

    # relationship should pick it up
    assert len(rd.messages) == 1
    assert rd.messages[0].message_content == "hello"


def test_is_roles_turn_handles_same_second_timestamps(session: Session):
    """Turn ordering should remain stable when messages share a timestamp."""
    prop = Proposition(id="p-same-sec", proposition_is_correct=True)
    session.add(prop)
    user = ExternalUser(external_id="DDD")
    session.add(user)
    session.commit()
    session.refresh(user)

    rd = RoundORM(
        persuader_id=user.id,
        target_id=user.id,
        proposition=prop.id,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    base = datetime.now(timezone.utc).replace(microsecond=0)
    shared = base + timedelta(seconds=1)

    msg1 = SentMessage(
        audio="",
        transcript={"role": "persuader", "content": "p1"},
        message_content="p1",
        thought_content=None,
        reasoning_trace=None,
        flagged=False,
        flagged_response=None,
        is_target=False,
        round_id=rd.id,
        created_at=base,
    )
    msg2 = SentMessage(
        audio="",
        transcript={"role": "target", "content": "t1"},
        message_content="t1",
        thought_content=None,
        reasoning_trace=None,
        flagged=False,
        flagged_response=None,
        is_target=True,
        round_id=rd.id,
        created_at=shared,
    )
    msg3 = SentMessage(
        audio="",
        transcript={"role": "persuader", "content": "p2"},
        message_content="p2",
        thought_content=None,
        reasoning_trace=None,
        flagged=False,
        flagged_response=None,
        is_target=False,
        round_id=rd.id,
        created_at=shared,
    )
    session.add_all([msg1, msg2, msg3])
    session.commit()

    rd_reloaded = session.get(RoundORM, rd.id)
    ordered = rd_reloaded.non_flagged_messages()
    assert [message.is_target for message in ordered] == [False, True, False]
    assert rd_reloaded.is_roles_turn(True) is True
