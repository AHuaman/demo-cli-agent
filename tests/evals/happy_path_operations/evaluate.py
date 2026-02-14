from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import src.database
from src.agent import appointment_agent
from src.models import Appointment
from src.negotiator import negotiator_agent
from src.tools import Deps


@contextmanager
def evaluation_context():
    """
    Sets up a safe, isolated environment for evaluation.
    - Mocks the database to use an in-memory SQLite instance.
    - Seeds the database with test data.
    - Mocks datetime to a fixed point in time.
    """
    # 1. Setup In-Memory DB
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)

    # 2. Seed Data
    with Session(test_engine) as session:
        appts = [
            Appointment(
                title="Doctor's Appointment",
                start_time=datetime.fromisoformat("2026-02-12T14:00:00"),
                end_time=datetime.fromisoformat("2026-02-12T14:00:00")
                + timedelta(minutes=60),
            ),
            Appointment(
                title="Team Sync",
                start_time=datetime.fromisoformat("2026-02-12T09:00:00"),
                end_time=datetime.fromisoformat("2026-02-12T09:00:00")
                + timedelta(minutes=30),
            ),
            Appointment(
                title="Lunch with Sarah",
                start_time=datetime.fromisoformat("2026-02-13T12:00:00"),
                end_time=datetime.fromisoformat("2026-02-13T12:00:00")
                + timedelta(minutes=60),
            ),
            Appointment(
                title="Marketing Team Meeting",
                start_time=datetime.fromisoformat("2026-02-17T15:00:00"),
                end_time=datetime.fromisoformat("2026-02-17T15:00:00")
                + timedelta(minutes=60),
                location="Conference Room B",
            ),
            Appointment(
                title="Dentist Appointment",
                start_time=datetime.fromisoformat("2026-05-05T10:00:00"),
                end_time=datetime.fromisoformat("2026-05-05T10:00:00")
                + timedelta(minutes=60),
                location="Downtown Clinic",
            ),
        ]
        session.add_all(appts)
        session.commit()

    # 3. Define the mocked get_session
    @contextmanager
    def mock_get_session():
        with Session(test_engine) as session:
            yield session

    # 4. Mock Datetime (Fixed to Wed, Feb 11, 2026, 10:00 AM)
    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 11, 10, 0, 0)

    # Apply patches
    with (
        patch("src.database.get_session", side_effect=mock_get_session),
        patch("src.agent.datetime", MockDatetime),
        patch("src.negotiator.datetime", MockDatetime),
    ):
        yield


async def predict_fn(user_prompt: str) -> str:
    """Run the appointment agent and return its response."""
    with src.database.get_session() as session:
        deps = Deps(session=session, negotiator_agent=negotiator_agent)
        result = await appointment_agent.run(user_prompt, deps=deps)
        return result.output
