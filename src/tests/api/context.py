"""
src/tests/api/context.py

Author: Jared Moore
Date: July, 2025

Shared context around api tests
"""

from collections import Counter
from datetime import timedelta

import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from api.sql_model import CONNECT_ARGS, SQLITE_URL_FMT, Proposition
from api.utils import ServerSettings
from experiment.condition import PAIRED_HUMAN_ROLE, Condition
from experiment.utils import EXAMPLE_PROPOSITIONS_FILE

SQLITE_TEST_URL = SQLITE_URL_FMT.format(filename=":memory:")

TEST_SETTINGS = ServerSettings(
    condition_num_rounds=Counter(
        {
            Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False): 2,
        }
    ),
    conditions=None,
    waiting_room_timeout=timedelta(seconds=2),
    participant_conversation_timeout=timedelta(seconds=5),
    overassign_non_paired_conditions=True,
    dev_environment=True,
    propositions_filenames=[EXAMPLE_PROPOSITIONS_FILE],
)


@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine(
        SQLITE_TEST_URL,
        echo=False,
        connect_args=CONNECT_ARGS,
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="session")
def session_fixture(engine):
    with Session(engine) as session:
        if session.get(Proposition, "prop_ws") is None:
            session.add(
                Proposition(
                    id="prop_ws", factual_domain=False, proposition_is_correct=None
                )
            )
            session.commit()
        yield session
