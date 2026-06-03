"""
src/tests/api/test_message_processing.py

Author: Jared Moore
Date: July, 2025

Tests for message processing
"""

import pytest
from sqlmodel import Session, select

from api.message_processing import (
    _limit_llm_response_content,
    process_message_and_response,
    use_dummy_endpoints,
)
from api.sql_model import ExternalUser, Participant, Proposition, RoundORM, SentMessage

from .context import engine_fixture, session_fixture


@pytest.fixture(autouse=True)
def _enable_dummy_endpoints():
    """Globally switch to dummy TTS / ASR / moderation."""
    use_dummy_endpoints()


def _make_participant(session: Session, external_id: str) -> Participant:
    """Create an ExternalUser + Participant pair and return the Participant."""
    eu = ExternalUser(external_id=external_id)
    session.add(eu)
    session.commit()
    session.refresh(eu)

    p = Participant(id=eu.id, role="either")
    session.add(p)
    session.commit()
    session.refresh(p)
    session.expunge(p)
    return p


def _ensure_prop(session: Session, pid: str = "p1") -> None:
    """Ensure a Proposition with the given id exists."""
    if session.get(Proposition, pid) is None:
        prop = Proposition(id=pid, factual_domain=False, proposition_is_correct=None)
        session.add(prop)
        session.commit()


def _get_messages(session: Session, rid: int) -> list[SentMessage]:
    return session.exec(
        select(SentMessage).where(SentMessage.round_id == rid).order_by(SentMessage.id)
    ).all()


def test_llm_as_persuader(engine):
    with Session(engine) as session:
        _ensure_prop(session, pid="propA")
        # create a human target
        target = _make_participant(session, "T1")
        # create a round with an LLM persuader
        rd = RoundORM(
            llm_persuader="gpt-4o",
            target_id=target.id,
            proposition="propA",
            persuader_supports_proposition=False,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    # reload as in-memory Round
    with Session(engine) as session:
        rd_orm = session.get(RoundORM, rid)
        rd_in_mem = rd_orm.as_round()
        assert rd_in_mem.condition is not None

    # human target speaks
    process_message_and_response(
        content="dummy_audio_b64",
        is_target=True,
        round_id=rid,
        engine=engine,
    )

    # inspect DB
    with Session(engine) as session:
        rd2 = session.get(RoundORM, rid)
        msgs = _get_messages(session, rid)
        assert rd2 is not None

    # two messages: human -> “hello”, then LLM -> “hello”
    assert len(msgs) == 2
    assert [m.message_content for m in msgs] == ["hello", "hello"]
    assert msgs[0].is_target is True
    assert msgs[1].is_target is False


def test_llm_persuader_reasoning_effort_can_be_configured(engine, monkeypatch):
    """Configured persuader reasoning effort should be forwarded to call_llm."""
    monkeypatch.setattr("api.message_processing._DUMMY_MODE", False)
    monkeypatch.setattr(
        "api.message_processing.model_supports_reasoning", lambda _: True
    )
    captured: dict[str, object] = {}

    def fake_call_llm(*, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["kwargs"] = dict(kwargs)
        return {"text": "hello", "raw_response": {}}

    monkeypatch.setattr("api.message_processing.call_llm", fake_call_llm)

    with Session(engine) as session:
        _ensure_prop(session, pid="prop_reasoning_cfg")
        target = _make_participant(session, "T_reasoning_cfg")
        rd = RoundORM(
            llm_persuader="gpt-5",
            target_id=target.id,
            proposition="prop_reasoning_cfg",
            persuader_supports_proposition=False,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    process_message_and_response(
        content="hello",
        is_target=True,
        round_id=rid,
        engine=engine,
        use_audio=False,
        llm_persuader_reasoning_effort="medium",
    )

    kwargs = captured["kwargs"]
    assert kwargs["temperature"] == 1
    assert kwargs["reasoning_effort"] == "medium"


def test_llm_persuader_reasoning_effort_defaults_to_provider(engine, monkeypatch):
    """Unset persuader reasoning effort should omit reasoning_effort from call_llm."""
    monkeypatch.setattr("api.message_processing._DUMMY_MODE", False)
    monkeypatch.setattr(
        "api.message_processing.model_supports_reasoning", lambda _: True
    )
    captured: dict[str, object] = {}

    def fake_call_llm(*, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["kwargs"] = dict(kwargs)
        return {"text": "hello", "raw_response": {}}

    monkeypatch.setattr("api.message_processing.call_llm", fake_call_llm)

    with Session(engine) as session:
        _ensure_prop(session, pid="prop_reasoning_default")
        target = _make_participant(session, "T_reasoning_default")
        rd = RoundORM(
            llm_persuader="gpt-5",
            target_id=target.id,
            proposition="prop_reasoning_default",
            persuader_supports_proposition=False,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    process_message_and_response(
        content="hello",
        is_target=True,
        round_id=rid,
        engine=engine,
        use_audio=False,
        llm_persuader_reasoning_effort=None,
    )

    kwargs = captured["kwargs"]
    assert kwargs["temperature"] == 1
    assert "reasoning_effort" not in kwargs


def test_naive_llm_as_persuader(engine, monkeypatch):
    """Naive persuader model should generate deterministic non-LLM reply text."""
    monkeypatch.setattr("api.message_processing._DUMMY_MODE", False)
    proposition_id = "People should recycle"
    with Session(engine) as session:
        _ensure_prop(session, pid=proposition_id)
        target = _make_participant(session, "T_naive")
        rd = RoundORM(
            llm_persuader="naive",
            target_id=target.id,
            proposition=proposition_id,
            persuader_supports_proposition=True,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    process_message_and_response(
        content="dummy_audio_b64",
        is_target=True,
        round_id=rid,
        engine=engine,
    )

    with Session(engine) as session:
        msgs = _get_messages(session, rid)

    assert len(msgs) == 2
    assert msgs[0].is_target is True
    assert msgs[1].is_target is False
    assert msgs[1].message_content == "This proposition is true: People should recycle."


def test_llm_as_target(engine):
    with Session(engine) as session:
        _ensure_prop(session, pid="propB")
        # create a human persuader
        persuader = _make_participant(session, "P1")
        # create a round with an LLM target
        rd = RoundORM(
            llm_target="gpt-4o",
            persuader_id=persuader.id,
            proposition="propB",
            persuader_supports_proposition=False,
            target_initial_belief=0.4,
            turn_limit=2,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    with Session(engine) as session:
        rd_in_mem = session.get(RoundORM, rid).as_round()
        assert rd_in_mem.condition is not None

    # human persuader speaks
    process_message_and_response(
        content="dummy_audio_b64",
        is_target=False,
        round_id=rid,
        engine=engine,
    )

    with Session(engine) as session:
        rd2 = session.get(RoundORM, rid)
        msgs = _get_messages(session, rid)
        assert rd2 is not None

    assert len(msgs) == 2
    assert [m.message_content for m in msgs] == ["hello", "hello"]
    assert msgs[0].is_target is False
    assert msgs[1].is_target is True


def test_paired_human_exchange(engine):
    with Session(engine) as session:
        _ensure_prop(session, pid="propC")
        a = _make_participant(session, "A")
        b = _make_participant(session, "B")
        # create a paired-human round
        rd = RoundORM(
            persuader_id=a.id,
            target_id=b.id,
            proposition="propC",
            persuader_supports_proposition=False,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    # 1) persuader -> target
    with Session(engine) as session:
        rd_in_mem = session.get(RoundORM, rid).as_round()
        assert rd_in_mem.condition is not None

    process_message_and_response(
        content="dummy_b64_1",
        is_target=False,
        round_id=rid,
        engine=engine,
    )

    with Session(engine) as session:
        rd2 = session.get(RoundORM, rid)
        msgs = _get_messages(session, rid)
        assert rd2 is not None

    assert len(msgs) == 1
    assert msgs[0].message_content == "hello"
    assert msgs[0].is_target is False

    # 2) target -> persuader
    with Session(engine) as session:
        rd_in_mem = session.get(RoundORM, rid).as_round()
        assert rd_in_mem.condition is not None

    process_message_and_response(
        content="dummy_b64_2",
        is_target=True,
        round_id=rid,
        engine=engine,
    )

    with Session(engine) as session:
        rd3 = session.get(RoundORM, rid)
        msgs = _get_messages(session, rid)
        assert rd3 is not None

    assert len(msgs) == 2
    assert msgs[1].message_content == "hello"
    assert msgs[1].is_target is True


def test_llm_reply_truncates_by_char_limit(engine):
    with Session(engine) as session:
        _ensure_prop(session, pid="propD")
        target = _make_participant(session, "T2")
        rd = RoundORM(
            llm_persuader="gpt-4o",
            target_id=target.id,
            proposition="propD",
            persuader_supports_proposition=False,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    long_text = "x" * 50
    process_message_and_response(
        content=long_text,
        is_target=True,
        round_id=rid,
        engine=engine,
        use_audio=False,
        max_response_chars=10,
        max_audio_duration_s=None,
    )

    with Session(engine) as session:
        msgs = _get_messages(session, rid)

    assert len(msgs) == 2
    assert msgs[0].message_content == long_text
    assert msgs[1].message_content == "x" * 10


def test_llm_reply_truncates_by_audio_duration(engine):
    with Session(engine) as session:
        _ensure_prop(session, pid="propE")
        target = _make_participant(session, "T3")
        rd = RoundORM(
            llm_persuader="gpt-4o",
            target_id=target.id,
            proposition="propE",
            persuader_supports_proposition=False,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    long_text = ("word " * 200).strip()
    process_message_and_response(
        content=long_text,
        is_target=True,
        round_id=rid,
        engine=engine,
        use_audio=True,
        max_response_chars=5000,
        max_audio_duration_s=1,
    )

    with Session(engine) as session:
        msgs = _get_messages(session, rid)

    assert len(msgs) == 2
    assert msgs[0].message_content == "hello"
    assert msgs[1].message_content == "hello"

    trimmed = _limit_llm_response_content(
        long_text,
        use_audio=True,
        max_response_chars=5000,
        max_audio_duration_s=1,
    )
    assert trimmed
    assert len(trimmed) < len(long_text)


def test_llm_reply_skipped_if_timeout_after_generation(engine, monkeypatch):
    """Skip storing the LLM response if the round times out after generation."""
    with Session(engine) as session:
        _ensure_prop(session, pid="prop_timeout_llm")
        target = _make_participant(session, "T_timeout")
        rd = RoundORM(
            llm_persuader="gpt-4o",
            target_id=target.id,
            proposition="prop_timeout_llm",
            persuader_supports_proposition=False,
        )
        session.add(rd)
        session.commit()
        session.refresh(rd)
        rid = rd.id

    call_state = {"count": 0}

    def fake_round_over_state(_engine, _round_id, *, is_target, round_time_limit):
        call_state["count"] += 1
        if call_state["count"] >= 3:
            return True, True
        return True, False

    monkeypatch.setattr(
        "api.message_processing.round_over_state", fake_round_over_state
    )

    process_message_and_response(
        content="dummy_audio_b64",
        is_target=True,
        round_id=rid,
        engine=engine,
        use_audio=False,
        round_time_limit=1,
    )

    with Session(engine) as session:
        msgs = _get_messages(session, rid)

    assert len(msgs) == 1
    assert msgs[0].is_target is True
