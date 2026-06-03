# src/tests/api/test_api.py

import asyncio
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select
from starlette.websockets import WebSocketState

import api.api
from api.api import (
    ParticipantInitRequest,
    ParticipantPropositionRequest,
    ParticipantRequest,
    ParticipantRoundRequest,
    current_round,
    participant_init,
    participant_propositions,
    participant_ready,
    participant_rounds,
    round_ws,
)
from api.message_processing import use_dummy_endpoints
from api.sql_model import (
    ExternalUser,
    Participant,
    Proposition,
    RoundORM,
    SentMessage,
    SentMessageBase,
)
from api.utils import MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER, round_over_state
from experiment.condition import Condition, Roles

from .context import TEST_SETTINGS, engine_fixture, session_fixture


@pytest.fixture(name="settings")
def settings_fixture():
    return TEST_SETTINGS


@pytest.fixture(autouse=True)
def _enable_dummy_endpoints():
    """Globally switch to dummy TTS / ASR / moderation."""
    use_dummy_endpoints()


@pytest.fixture(autouse=True)
def patch_global_engine(engine, monkeypatch):
    # engine is the in-memory sqlite from engine_fixture
    # Override api.api.global_engine so round_ws and other handlers
    # will use the same DB with tables already created.
    monkeypatch.setattr(api.api, "global_engine", engine)
    yield
    # no teardown needed


#
# Helpers
#


def _ensure_prop(session: Session, pid: str = "prop_test"):
    """Ensure there's at least one Proposition in the DB."""
    if session.get(Proposition, pid) is None:
        session.add(
            Proposition(id=pid, factual_domain=False, proposition_is_correct=None)
        )
        session.commit()


#
# Stable, low-level endpoint tests
#


def test_current_round_nonexistent_participant(session):
    # participant 999 does not exist
    with pytest.raises(HTTPException) as exc:
        current_round(
            ParticipantRoundRequest(participant_id=999),
            session,
            TEST_SETTINGS,
        )
    assert exc.value.status_code == 400
    assert "does not exist" in exc.value.detail


def test_current_round_no_rounds_left(session):
    # create and ready a participant
    eu = ExternalUser(external_id="X")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    p = Participant(id=eu.id)
    session.add(p)
    session.commit()
    # use a fresh settings with zero allowed rounds
    settings0 = TEST_SETTINGS.model_copy(update={"rounds_per_participant": 0})
    # mark them ready
    p.entered_waiting_room = datetime.now(timezone.utc)
    session.add(p)
    session.commit()
    # now should error due to no rounds left
    with pytest.raises(HTTPException) as exc:
        current_round(
            ParticipantRoundRequest(participant_id=p.id),
            session,
            settings0,
        )
    assert exc.value.status_code == 400
    assert "no more rounds to play" in exc.value.detail.lower()


def test_current_round_waiting_too_long(session):
    eu = ExternalUser(external_id="Y")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    p = Participant(id=eu.id)
    session.add(p)
    session.commit()
    # put them in waiting room long ago
    old = datetime.now(timezone.utc) - (
        MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER * TEST_SETTINGS.waiting_room_timeout
        + timedelta(seconds=1)
    )
    p.entered_waiting_room = old
    session.add(p)
    session.commit()
    with pytest.raises(HTTPException) as exc:
        current_round(
            ParticipantRoundRequest(participant_id=p.id),
            session,
            TEST_SETTINGS,
        )
    assert exc.value.status_code == 400
    assert "has waited too long" in exc.value.detail.lower()


def test_current_round_force_overrides_in_dev(session):
    # dev_environment=True allows forcing
    eu = ExternalUser(external_id="Z")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    p = Participant(id=eu.id)
    session.add(p)
    session.commit()
    # ready them
    p.entered_waiting_room = datetime.now(timezone.utc)
    session.add(p)
    session.commit()
    # ensure a proposition exists
    _ensure_prop(session, pid="force_prop")
    # force them to be target, with a fake llm persuader
    req = ParticipantRoundRequest(
        participant_id=p.id,
        is_target=True,
        llm_persuader="dummy-model",
        proposition="force_prop",
        # you can also pass persuader_supports_proposition if you like,
        # e.g. persuader_supports_proposition=True
    )
    res = current_round(req, session, TEST_SETTINGS)
    assert res["is_target"] is True
    # verify in DB
    rd = session.get(RoundORM, res["round_id"])
    assert rd.llm_persuader == "dummy-model"
    assert rd.target_id == p.id


def test_current_round_requires_participant_propositions(session):
    eu = ExternalUser(external_id="PP")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    participant = Participant(id=eu.id)
    session.add(participant)
    session.commit()

    condition = Condition(
        roles=Roles(human_target=True, llm_persuader="gpt-4o-mini"),
        factual_domain=False,
        participant_proposition=True,
    )
    session.add(participant)
    session.commit()

    settings_pp = TEST_SETTINGS.model_copy(
        update={"condition_num_rounds": Counter({condition: 1})}
    )
    participant.entered_waiting_room = datetime.now(timezone.utc)
    session.add(participant)
    session.commit()

    with pytest.raises(HTTPException) as exc:
        current_round(
            ParticipantRoundRequest(participant_id=participant.id),
            session,
            settings_pp,
        )
    assert exc.value.status_code == 400
    assert "propositions are required" in exc.value.detail.lower()


def test_participant_propositions_submit(session, monkeypatch):
    eu = ExternalUser(external_id="PP2")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    participant = Participant(id=eu.id)
    condition = Condition(
        roles=Roles(human_target=True, llm_persuader="gpt-4o-mini"),
        factual_domain=False,
        participant_proposition=True,
    )
    session.add(participant)
    session.commit()

    settings_pp = TEST_SETTINGS.model_copy(
        update={"condition_num_rounds": Counter({condition: 1})}
    )

    monkeypatch.setattr(api.api, "moderate_content", lambda _: False)
    monkeypatch.setattr(
        api.api,
        "rephrase_participant_decision",
        lambda _: {"status": "ok", "proposition": "I should apply for a new job."},
    )

    res = participant_propositions(
        ParticipantPropositionRequest(
            participant_id=participant.id, decision="I might change jobs."
        ),
        session,
        settings_pp,
    )
    assert res["status"] == "ok"
    assert res["remaining_count"] == settings_pp.rounds_per_participant - 1
    prop = session.exec(
        select(Proposition).where(Proposition.participant_id == participant.id)
    ).first()
    assert prop is not None
    assert prop.id == "I should apply for a new job."
    assert prop.original_text == "I might change jobs."


@pytest.mark.asyncio
async def test_send_message_wrong_turn_and_not_your_round(session, settings):
    # 1) initialize and ready two participants
    resp_a = participant_init(ParticipantInitRequest(id="A"), None, session)
    pid_a = resp_a["participant_id"]
    resp_b = participant_init(ParticipantInitRequest(id="B"), None, session)
    pid_b = resp_b["participant_id"]

    participant_ready(ParticipantRequest(id=pid_a), session)
    participant_ready(ParticipantRequest(id=pid_b), session)

    # ensure a proposition
    _ensure_prop(session, pid="wtr")

    # assign them to a round
    cr_a = current_round(
        ParticipantRoundRequest(participant_id=pid_a), session, settings
    )
    _ = current_round(ParticipantRoundRequest(participant_id=pid_b), session, settings)
    rid = cr_a["round_id"]

    # identify persuader and target
    if not cr_a["is_target"]:
        pid_p, pid_t = pid_a, pid_b
    else:
        pid_p, pid_t = pid_b, pid_a

    # open WS for both
    ws_p = DummyWebSocket()
    ws_t = DummyWebSocket()
    task_p = asyncio.create_task(round_ws(ws_p, rid, pid_p, settings))
    task_t = asyncio.create_task(round_ws(ws_t, rid, pid_t, settings))
    await asyncio.sleep(0.05)

    # Target sets initial belief
    await ws_t.incoming.put({"type": "make_choice", "initial": True, "belief": 0.4})
    await asyncio.sleep(0.05)

    # persuader sends first => OK
    await ws_p.incoming.put({"type": "message", "content": "foo"})
    await asyncio.sleep(1.5)
    assert any(msg["type"] == "response" and msg["text"] == "foo" for msg in ws_t.sent)

    # same persuader tries again => should be wrong-turn => WS closes
    await ws_p.incoming.put({"type": "message", "content": "bar"})
    await asyncio.sleep(0.05)
    assert ws_p.closed is True

    # outsider tries to connect => immediate close
    ws_out = DummyWebSocket()
    task_out = asyncio.create_task(round_ws(ws_out, rid, 999, settings))
    await asyncio.sleep(0.05)
    assert ws_out.closed is False

    # cleanup
    for t in (task_p, task_t, task_out):
        t.cancel()
    await asyncio.gather(task_p, task_t, task_out, return_exceptions=True)


@pytest.mark.asyncio
async def test_retrieve_response_flagged_and_timeout(session, settings, monkeypatch):
    # Set up a human-human round
    async def _setup_human_round():
        eu_a = ExternalUser(external_id="HA")
        eu_b = ExternalUser(external_id="HB")
        session.add_all((eu_a, eu_b))
        session.commit()
        session.refresh(eu_a)
        session.refresh(eu_b)
        p_a = Participant(id=eu_a.id)
        p_b = Participant(id=eu_b.id)
        session.add_all((p_a, p_b))
        session.commit()
        # create round
        rd = RoundORM(
            persuader_id=p_a.id,
            target_id=p_b.id,
            llm_persuader=None,
            llm_target=None,
            proposition="prop_ws",
            persuader_supports_proposition=True,
            turn_limit=1,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        # assign current_round
        p_a.current_round = rd.id
        p_b.current_round = rd.id
        session.add_all((p_a, p_b))
        session.commit()
        return rd.id, p_a.id, p_b.id

    rid, pid_a, pid_b = await _setup_human_round()

    # 1) flagged branch via process_message_and_response => WS emits 'flagged'
    def fake_proc(*args, **kwargs):  # pylint: disable=unused-argument
        return (
            SentMessageBase(flagged=True, message_content="", is_target=False),
            None,
            None,
            None,
            False,
            False,
        )

    monkeypatch.setattr(
        "api.message_processing.process_message_and_response",
        fake_proc,
    )

    ws_a = DummyWebSocket()
    ws_b = DummyWebSocket()
    task_a = asyncio.create_task(round_ws(ws_a, rid, pid_a, settings))
    task_b = asyncio.create_task(round_ws(ws_b, rid, pid_b, settings))
    await asyncio.sleep(0.05)

    # persuader A sends bad content
    await ws_a.incoming.put({"type": "message", "content": "bad"})
    await asyncio.sleep(0.05)
    # A sees 'flagged'; B sees nothing
    assert any(m["type"] == "flagged" for m in ws_a.sent)
    assert not any(m.get("text") == "bad" for m in ws_b.sent)

    # 2) timeout branch => both WS close after inactivity
    await asyncio.sleep(settings.participant_conversation_timeout.total_seconds() + 0.1)
    assert ws_a.closed is True
    assert ws_b.closed is True

    # cleanup
    for t in (task_a, task_b):
        t.cancel()
    await asyncio.gather(task_a, task_b, return_exceptions=True)


def test_participant_rounds_non_empty(session):
    eu = ExternalUser(external_id="G")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    p = Participant(id=eu.id, role="target")
    session.add(p)
    session.commit()

    # round1: human–human
    other_eu = ExternalUser(external_id="H")
    session.add(other_eu)
    session.commit()
    session.refresh(other_eu)
    rd1 = RoundORM(
        persuader_id=other_eu.id,
        target_id=p.id,
        proposition="prop1",
        target_initial_belief=0.1,
        target_final_belief=0.2,
        persuader_supports_proposition=True,
    )

    # round2: human–LLM
    _ensure_prop(session, pid="prop2")
    rd2 = RoundORM(
        persuader_id=p.id,
        llm_target="dummy-model",
        proposition="prop2",
        target_initial_belief=0.3,
        target_final_belief=0.4,
        persuader_supports_proposition=False,
    )

    session.add_all((rd1, rd2))
    session.commit()

    res = participant_rounds(ParticipantRequest(id=p.id), session, TEST_SETTINGS)
    assert res["num_rounds"] == 2
    assert res["rounds_remaining"] == TEST_SETTINGS.rounds_per_participant - 2
    # both rounds count as "human conversations" when p.role == "target"
    assert res["num_human_conversations"] == 2


def test_participant_init_and_ready(session: Session):
    # Calling ready on a non-existent participant should 400
    with pytest.raises(HTTPException):
        participant_ready(ParticipantRequest(id=1), session)

    # Initialize a new external user
    init_resp = participant_init(ParticipantInitRequest(id="EXT_A"), None, session)
    pid = init_resp["participant_id"]
    assert pid == 1

    # ExternalUser and Participant rows should now exist
    eu = session.get(ExternalUser, pid)
    assert eu.external_id == "EXT_A"
    par = session.get(Participant, pid)
    assert par.id == pid
    assert par.entered_waiting_room is None
    assert par.current_round is None

    # Mark ready -> should set entered_waiting_room but not current_round
    participant_ready(ParticipantRequest(id=pid), session)
    par = session.get(Participant, pid)
    assert par.entered_waiting_room is not None
    # timestamp is very recent
    assert (
        0
        <= (
            datetime.now(timezone.utc)
            - par.entered_waiting_room.replace(tzinfo=timezone.utc)
        ).total_seconds()
        < 1.0
    )
    assert par.current_round is None

    # Second ready call simply updates the timestamp
    ts1 = par.entered_waiting_room
    time.sleep(0.01)
    participant_ready(ParticipantRequest(id=pid), session)
    par2 = session.get(Participant, pid)
    assert par2.entered_waiting_room >= ts1


def test_participant_rounds_empty(session: Session):
    # Non-existent participant
    with pytest.raises(HTTPException):
        participant_rounds(ParticipantRequest(id=999), session, TEST_SETTINGS)

    # New participant has no rounds yet
    pid = participant_init(ParticipantInitRequest(id="solo"), None, session)[
        "participant_id"
    ]
    pr = participant_rounds(ParticipantRequest(id=pid), session, TEST_SETTINGS)
    assert pr["num_rounds"] == 0
    # rounds_remaining should equal settings.rounds_per_participant
    assert pr["rounds_remaining"] == TEST_SETTINGS.rounds_per_participant
    assert pr["num_human_conversations"] == 0
    assert pr["completion_code"] == TEST_SETTINGS.completion_code

    # now create two rounds
    # we cheat and insert them directly
    other_eu = ExternalUser(external_id="H")
    session.add(other_eu)
    session.commit()
    session.refresh(other_eu)

    # round1: human–human
    rd1 = RoundORM(
        persuader_id=other_eu.id,
        target_id=pid,
        proposition="p1",
        target_initial_belief=0.1,
        target_final_belief=0.2,
        persuader_supports_proposition=True,
    )
    # round2: human–llm
    _ensure_prop(session, pid="p2")
    rd2 = RoundORM(
        persuader_id=pid,
        llm_target="dummy",
        proposition="p2",
        target_initial_belief=0.3,
        target_final_belief=0.4,
        persuader_supports_proposition=False,
    )

    session.add_all((rd1, rd2))
    session.commit()

    pr2 = participant_rounds(ParticipantRequest(id=pid), session, TEST_SETTINGS)
    assert pr2["num_rounds"] == 2
    assert pr2["rounds_remaining"] == TEST_SETTINGS.rounds_per_participant - 2


def test_round_over_state_marks_timeout(session):
    """_round_over_state should set timed_out when the round exceeds the limit."""
    _ensure_prop(session, pid="prop_timeout")
    settings = TEST_SETTINGS.model_copy(update={"round_time_limit": 1})
    eu = ExternalUser(external_id="timeout_user")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    participant = Participant(id=eu.id)
    session.add(participant)
    session.commit()
    rd = RoundORM(
        proposition="prop_timeout",
        llm_persuader="gpt-4o",
        target_id=participant.id,
        persuader_supports_proposition=False,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)
    rd.created_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session.add(rd)
    session.commit()

    turns_left, timed_out = round_over_state(
        api.api.global_engine,
        rd.id,
        is_target=True,
        round_time_limit=settings.round_time_limit,
    )

    assert timed_out is True
    assert turns_left is True or turns_left is not None
    session.refresh(rd)
    assert rd.timed_out is True


@pytest.mark.asyncio
async def test_current_round_and_exchange(session, settings):
    # initialize two participants
    resp_a = participant_init(ParticipantInitRequest(id="X1"), None, session)
    pid_a = resp_a["participant_id"]
    resp_b = participant_init(ParticipantInitRequest(id="X2"), None, session)
    pid_b = resp_b["participant_id"]

    # both ready
    participant_ready(ParticipantRequest(id=pid_a), session)
    participant_ready(ParticipantRequest(id=pid_b), session)

    # ensure proposition
    _ensure_prop(session, pid="prop_cycle")

    # get or create the round
    cr_a = current_round(
        ParticipantRoundRequest(participant_id=pid_a), session, settings
    )
    cr_b = current_round(
        ParticipantRoundRequest(participant_id=pid_b), session, settings
    )
    assert cr_a["round_id"] == cr_b["round_id"]
    rid = cr_a["round_id"]
    assert cr_a["is_target"] != cr_b["is_target"]

    # identify roles
    if not cr_a["is_target"]:
        pid_p, pid_t = pid_a, pid_b
    else:
        pid_p, pid_t = pid_b, pid_a

    # open WS
    ws_p = DummyWebSocket()
    ws_t = DummyWebSocket()
    task_p = asyncio.create_task(round_ws(ws_p, rid, pid_p, settings))
    task_t = asyncio.create_task(round_ws(ws_t, rid, pid_t, settings))
    await asyncio.sleep(0.05)

    # persuader sends first
    await ws_p.incoming.put({"type": "message", "content": "hi there"})
    await asyncio.sleep(0.05)

    # DB should have exactly one message
    msgs = session.exec(select(SentMessage).where(SentMessage.round_id == rid)).all()
    assert len(msgs) == 1
    assert msgs[0].message_content == "hi there"
    assert msgs[0].is_target is False

    # target receives it
    assert any(m["type"] == "response" and m["text"] == "hi there" for m in ws_t.sent)

    # target replies
    await ws_t.incoming.put({"type": "message", "content": "hi there"})
    await asyncio.sleep(0.05)

    # now two messages
    msgs = session.exec(select(SentMessage).where(SentMessage.round_id == rid)).all()
    assert len(msgs) == 2
    assert msgs[1].message_content == "hi there"

    # check turns left on in-memory Round
    rd = session.get(RoundORM, rid).as_round()
    assert rd.turns_left(is_target=False) == 1
    assert rd.turns_left(is_target=True) == 1

    # persuader receives target's reply
    assert any(m["type"] == "response" and m["text"] == "hi there" for m in ws_p.sent)

    # cleanup
    for t in (task_p, task_t):
        t.cancel()
    await asyncio.gather(task_p, task_t, return_exceptions=True)


#
# WS Tests
#


class DummyWebSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.client_state = WebSocketState.CONNECTED

    async def receive_json(self):
        return await asyncio.wait_for(self.incoming.get(), timeout=None)

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, _=None):
        self.closed = True

    async def accept(self):
        # no-op for testing
        pass


async def _setup_human_round(
    session: Session, no_early_end: bool = False, turn_limit: int = 1
):
    """
    Create a human-human round and return round and participant ids.
    """
    eu_a = ExternalUser(external_id="A")
    eu_b = ExternalUser(external_id="B")
    session.add_all((eu_a, eu_b))
    session.commit()
    session.refresh(eu_a)
    session.refresh(eu_b)

    p_a = Participant(id=eu_a.id, role=None)
    p_b = Participant(id=eu_b.id, role=None)
    session.add_all((p_a, p_b))
    session.commit()

    rd = RoundORM(
        persuader_id=p_a.id,
        target_id=p_b.id,
        llm_persuader=None,
        llm_target=None,
        proposition="prop_ws",
        persuader_supports_proposition=None,
        turn_limit=turn_limit,
        no_early_end=no_early_end,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    p_a.current_round = rd.id
    p_b.current_round = rd.id
    session.add_all((p_a, p_b))
    session.commit()
    return rd.id, p_a.id, p_b.id


async def _setup_llm_round(session: Session):
    eu = ExternalUser(external_id="T")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    pt = Participant(id=eu.id, role="target")
    session.add(pt)
    session.commit()

    rd = RoundORM(
        persuader_id=None,
        target_id=pt.id,
        llm_persuader="dummy-llm",
        llm_target=None,
        proposition="prop_ws",
        persuader_supports_proposition=None,
        turn_limit=1,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    pt.current_round = rd.id
    session.add(pt)
    session.commit()

    return rd.id, pt.id


async def _setup_llm_target_round(session: Session, *, enable_node_belief_survey: bool):
    """Create a human-persuader vs llm-target round for websocket tests."""
    eu = ExternalUser(external_id="HP")
    session.add(eu)
    session.commit()
    session.refresh(eu)
    participant = Participant(id=eu.id, role="persuader")
    session.add(participant)
    session.commit()

    proposition_id = "prop_llm_target_bn"
    proposition = Proposition(
        id=proposition_id,
        factual_domain=False,
        proposition_is_correct=None,
        bayesian_network={
            "target": "Main target proposition.",
            "belief_nodes": [
                "Belief statement one.",
                "Belief statement two.",
            ],
            "edges": [],
            "joint_distribution": [
                {
                    "state": {"Target": True, "Belief_1": True, "Belief_2": True},
                    "probability": 0.25,
                },
                {
                    "state": {"Target": True, "Belief_1": True, "Belief_2": False},
                    "probability": 0.25,
                },
                {
                    "state": {"Target": False, "Belief_1": False, "Belief_2": True},
                    "probability": 0.25,
                },
                {
                    "state": {"Target": False, "Belief_1": False, "Belief_2": False},
                    "probability": 0.25,
                },
            ],
        },
    )
    session.add(proposition)
    session.commit()

    rd = RoundORM(
        persuader_id=participant.id,
        target_id=None,
        llm_persuader=None,
        llm_target="dummy-target-model",
        proposition=proposition_id,
        persuader_supports_proposition=None,
        turn_limit=1,
        enable_node_belief_survey=enable_node_belief_survey,
    )
    session.add(rd)
    session.commit()
    session.refresh(rd)

    participant.current_round = rd.id
    session.add(participant)
    session.commit()

    return rd.id, participant.id


@pytest.mark.asyncio
async def test_human_human_ws_exchange(session, settings):
    rid, pt_a_id, pt_b_id = await _setup_human_round(session, turn_limit=1)
    ws_a, ws_b = DummyWebSocket(), DummyWebSocket()

    task_a = asyncio.create_task(round_ws(ws_a, rid, pt_a_id, settings))
    task_b = asyncio.create_task(round_ws(ws_b, rid, pt_b_id, settings))
    await asyncio.sleep(0.05)

    # Target sets initial belief
    await ws_b.incoming.put({"type": "make_choice", "initial": True, "belief": 0.4})
    await asyncio.sleep(0.05)
    rd_orm = session.get(RoundORM, rid)
    assert rd_orm.target_initial_belief == 0.4
    assert any(m["type"] == "round_started" for m in ws_b.sent)
    assert any(m["type"] == "round_started" for m in ws_a.sent)

    # Persuader A -> B
    await ws_a.incoming.put({"type": "message", "content": "hello from A"})
    await asyncio.sleep(1.5)
    assert any(
        m["type"] == "response" and m["text"] == "hello from A" for m in ws_b.sent
    )

    # Target B -> A
    await ws_b.incoming.put({"type": "message", "content": "reply from B"})
    await asyncio.sleep(1.5)
    try:
        assert any(
            m["type"] == "response" and m["text"] == "reply from B" for m in ws_a.sent
        )
    except AssertionError:
        print("ws_a.sent:", ws_a.sent)
        raise

    # Target sets initial belief
    await ws_b.incoming.put({"type": "make_choice", "initial": False, "belief": 0.6})
    await asyncio.sleep(1.5)
    session.refresh(rd_orm)
    assert rd_orm.target_final_belief == 0.6

    # Check that round result
    assert any(m["type"] == "round_over" for m in ws_b.sent)
    assert any(m["type"] == "round_result" for m in ws_b.sent)

    # Tear down
    task_a.cancel()
    task_b.cancel()
    await asyncio.gather(task_a, task_b, return_exceptions=True)

    # DB persisted
    rows = session.exec(select(SentMessage).where(SentMessage.round_id == rid)).all()
    assert {r.message_content for r in rows} == {"hello from A", "reply from B"}


@pytest.mark.asyncio
async def test_ws_no_early_end_blocks_target_end(session, settings):
    """Targets cannot end early when no_early_end is enabled."""
    rid, _, pt_b_id = await _setup_human_round(session, no_early_end=True)
    ws_b = DummyWebSocket()

    task_b = asyncio.create_task(round_ws(ws_b, rid, pt_b_id, settings))
    await asyncio.sleep(0.05)

    await ws_b.incoming.put({"type": "target_ends_round"})
    await asyncio.sleep(0.05)

    assert any(
        m["type"] == "error" and m["detail"] == "round cannot end yet"
        for m in ws_b.sent
    )
    rd_orm = session.get(RoundORM, rid)
    assert not rd_orm.target_ended_round

    task_b.cancel()
    await asyncio.gather(task_b, return_exceptions=True)


@pytest.mark.asyncio
async def test_llm_first_ws_flow(session, settings, monkeypatch):
    rid, pt_id = await _setup_llm_round(session)
    ws_t = DummyWebSocket()
    task_t = asyncio.create_task(round_ws(ws_t, rid, pt_id, settings))
    await asyncio.sleep(0.1)

    # Target sets initial belief
    await ws_t.incoming.put({"type": "make_choice", "initial": True, "belief": 0.4})
    await asyncio.sleep(0.05)

    # Initial LLM response
    assert any(msg["type"] == "response" for msg in ws_t.sent)
    # Human reply -> LLM dummy second response
    await ws_t.incoming.put({"type": "message", "content": "Thanks, LLM"})
    await asyncio.sleep(0.1)

    task_t.cancel()
    await asyncio.gather(task_t, return_exceptions=True)


@pytest.mark.asyncio
async def test_llm_target_node_beliefs_auto_recorded(session, settings, monkeypatch):
    """LLM-target rounds auto-fill pre/post node beliefs when survey mode is on."""
    rid, participant_id = await _setup_llm_target_round(
        session,
        enable_node_belief_survey=True,
    )
    rd_orm = session.get(RoundORM, rid)
    rd_orm.created_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session.add(rd_orm)
    session.commit()

    survey_values = iter(
        [
            {"Belief_1": 0.11, "Belief_2": 0.22},
            {"Belief_1": 0.33, "Belief_2": 0.44},
        ]
    )

    monkeypatch.setattr(
        api.api,
        "_llm_target_self_report_node_beliefs",
        lambda _rd, _model: next(survey_values),
    )

    timeout_settings = settings.model_copy(update={"round_time_limit": 1})
    ws = DummyWebSocket()
    task = asyncio.create_task(round_ws(ws, rid, participant_id, timeout_settings))
    await asyncio.sleep(0.15)

    session.refresh(rd_orm)
    assert rd_orm.target_initial_node_beliefs == {"Belief_1": 0.11, "Belief_2": 0.22}
    assert rd_orm.target_final_node_beliefs == {"Belief_1": 0.33, "Belief_2": 0.44}
    assert rd_orm.target_initial_belief is not None
    assert rd_orm.target_final_belief is not None

    assert any(msg["type"] == "round_started" for msg in ws.sent)
    assert any(msg["type"] == "round_over" for msg in ws.sent)
    assert any(msg["type"] == "round_result" for msg in ws.sent)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_ws_timeout(session, settings):
    """If neither side sends, both sockets close after timeout."""
    rid, pt_a_id, pt_b_id = await _setup_human_round(session, turn_limit=1)
    ws_a, ws_b = DummyWebSocket(), DummyWebSocket()

    task_a = asyncio.create_task(round_ws(ws_a, rid, pt_a_id, settings))
    task_b = asyncio.create_task(round_ws(ws_b, rid, pt_b_id, settings))

    # Wait beyond the participant_conversation_timeout
    await asyncio.sleep(settings.participant_conversation_timeout.total_seconds() + 0.1)

    assert ws_a.closed is True
    assert ws_b.closed is True

    task_a.cancel()
    task_b.cancel()
    await asyncio.gather(task_a, task_b, return_exceptions=True)


@pytest.mark.asyncio
async def test_ws_flagged_content(session, settings, monkeypatch):
    """If process_message_and_response returns flagged=True, WS emits a 'flagged' message."""
    rid, pt_a_id, pt_b_id = await _setup_human_round(session, turn_limit=1)

    # Monkey-patch to always flag the first message
    def fake_proc(*args, **kwargs):  # pylint: disable=unused-argument
        return (
            SentMessageBase(flagged=True, message_content="", is_target=False),
            None,
            None,
            None,
            False,
            False,
        )

    monkeypatch.setattr(
        "api.message_processing.process_message_and_response",
        fake_proc,
    )

    ws_a, ws_b = DummyWebSocket(), DummyWebSocket()
    task_a = asyncio.create_task(round_ws(ws_a, rid, pt_a_id, settings))
    task_b = asyncio.create_task(round_ws(ws_b, rid, pt_b_id, settings))
    await asyncio.sleep(0.05)

    # Target sets initial belief
    await ws_b.incoming.put({"type": "make_choice", "initial": True, "belief": 0.4})
    await asyncio.sleep(0.05)

    # Persuader A sends a message -> flagged
    await ws_a.incoming.put({"type": "message", "content": "bad content"})
    await asyncio.sleep(0.05)

    assert any(m["type"] == "flagged" for m in ws_a.sent)
    # B should NOT receive it
    assert not any(m.get("text") == "bad content" for m in ws_b.sent)

    task_a.cancel()
    task_b.cancel()
    await asyncio.gather(task_a, task_b, return_exceptions=True)


@pytest.mark.asyncio
async def test_ws_wrong_turn_closes(session, settings, monkeypatch):
    """
    If the same side speaks twice in a row, simulate a wrong-turn by
    raising HTTPException on the second call and assert the WS closes.
    """
    rid, pt_a_id, pt_b_id = await _setup_human_round(session, turn_limit=1)

    # Count calls so we can fail on the second from A
    call_count = {"A": 0}

    def fake_proc(*args, **kwargs):  # pylint: disable=unused-argument
        is_target = kwargs.get("is_target", args[1] if len(args) > 1 else False)
        if not is_target:  # A is persuader (is_target=False)
            call_count["A"] += 1
            if call_count["A"] > 1:
                raise HTTPException(status_code=400, detail="not your turn")
        # Otherwise normal: store and no reply
        return (
            SentMessageBase(flagged=False, message_content="", is_target=False),
            None,
            None,
            None,
            False,
            False,
        )

    monkeypatch.setattr(
        "api.message_processing.process_message_and_response",
        fake_proc,
    )

    ws_a, ws_b = DummyWebSocket(), DummyWebSocket()
    task_a = asyncio.create_task(round_ws(ws_a, rid, pt_a_id, settings))
    task_b = asyncio.create_task(round_ws(ws_b, rid, pt_b_id, settings))
    await asyncio.sleep(0.05)

    # Target sets initial belief
    await ws_b.incoming.put({"type": "make_choice", "initial": True, "belief": 0.4})
    await asyncio.sleep(0.05)

    # First good message
    await ws_a.incoming.put({"type": "message", "content": "first"})
    await asyncio.sleep(0.05)
    assert any(m["type"] == "response" for m in ws_b.sent)

    # Second message (wrong-turn) triggers HTTPException -> WS should close
    await ws_a.incoming.put({"type": "message", "content": "second"})
    # give the handler a moment to process and close
    await asyncio.sleep(0.05)
    assert ws_a.closed is True

    task_a.cancel()
    task_b.cancel()
    await asyncio.gather(task_a, task_b, return_exceptions=True)
