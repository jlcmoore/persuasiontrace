"""
src/tests/api/test_api_http.py

Author: Jared Moore
Date: July, 2025

Tests for http aspects of the api
"""

# src/tests/api/test_api_http.py

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from api.api import app, get_session
from api.sql_model import ExternalUser, Participant, Proposition

from .context import engine_fixture, session_fixture


@pytest.fixture(name="client")
def client_fixture(session: Session):
    # Override the session dependency so that FastAPI uses our test session
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def _ensure_prop(session: Session, pid: str = "p_http"):
    # Ensure at least one proposition exists
    if session.get(Proposition, pid) is None:
        session.add(
            Proposition(id=pid, factual_domain=False, proposition_is_correct=None)
        )
        session.commit()


def test_participant_init_http(client: TestClient, session: Session):
    # First call creates a new ExternalUser+Participant
    r = client.post("/participant_init/", json={"id": "HTTP_USER"})
    assert r.status_code == 200
    data = r.json()
    assert "participant_id" in data and isinstance(data["participant_id"], int)

    pid = data["participant_id"]
    # Confirm rows exist
    eu = session.get(ExternalUser, pid)
    assert eu is not None and eu.external_id == "HTTP_USER"
    par = session.get(Participant, pid)
    assert par is not None and par.id == pid


def test_participant_ready_nonexistent(client: TestClient):
    # Nonexistent participant
    r = client.post("/participant_ready/", json={"id": 999})
    assert r.status_code == 400


def test_current_round_nonexistent(client: TestClient):
    r = client.post("/current_round/", json={"participant_id": 999})
    assert r.status_code == 400
