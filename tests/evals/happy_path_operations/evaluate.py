from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

import mlflow
from mlflow.genai.judges import make_judge
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import src.database
from src.agent import appointment_agent
from src.models import Appointment
from src.negotiator import negotiator_agent
from src.settings import settings
from src.tools import Deps

_MLFLOW_EXPERIMENT_NAME: str = "Happy Path Operations Eval"

_GROUNDEDNESS_JUDGE_INSTRUCTIONS: str = """You are evaluating whether an agent's response is grounded in the expected appointments.

Expected appointments that should be referenced:
{{expectations}}

Agent's response:
{{outputs}}

Score 1 if the response ONLY mentions appointments that appear in the expected list (or correctly indicates none exist).
Score 0 if the response mentions ANY appointment NOT in the expected list (hallucination).

Special cases:
- If expected_appointments is empty and the agent asks for confirmation, score 1.
- If expected_appointments is empty and the agent says "no appointments found", score 1.
- Be strict: inventing appointment titles not in expected_appointments = score 0.
"""

_CORRECTNESS_JUDGE_INSTRUCTIONS: str = """You are evaluating if an agent's response correctly addresses the user's request.

User request: {{inputs}}
Expected behavior (from expectations.expected_answer_contains): {{expectations}}
Agent's response: {{outputs}}

Score 1 if:
- The response addresses the user's request appropriately
- The response contains all expected key elements (or reasonable semantic equivalents)
- For CREATE/DELETE/UPDATE requests: asking for confirmation counts as correct behavior

Score 0 if:
- The response is missing expected information
- The response contains factually incorrect information
- The response doesn't address the user's request
"""

# Set MLflow experiment
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(_MLFLOW_EXPERIMENT_NAME)

# Define LLM Judges
groundedness_judge = make_judge(
    name="answer_groundedness",
    instructions=_GROUNDEDNESS_JUDGE_INSTRUCTIONS,
    model=f"openai:/{settings.MODEL_NAME}",
    feedback_value_type=bool,
)

correctness_judge = make_judge(
    name="answer_correctness",
    instructions=_CORRECTNESS_JUDGE_INSTRUCTIONS,
    model=f"openai:/{settings.MODEL_NAME}",
    feedback_value_type=bool,
)


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
