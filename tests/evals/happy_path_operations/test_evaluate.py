import typing
from datetime import datetime

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlmodel import select

from src.agent import appointment_agent
from src.models import Appointment
from tests.evals.happy_path_operations import evaluate
from tests.live import requires_live_model


async def test_evaluation_context_isolation():
    """Verify evaluation_context provides an isolated async DB and a fixed clock."""
    async with evaluate.evaluation_context() as factory:
        # Check 1: The injected clock is frozen (no datetime patching involved).
        assert evaluate.FIXED_NOW == datetime(2026, 2, 11, 10, 0, 0)

        # Check 2: DB is isolated and seeded with 5 appointments.
        async with factory() as session:
            appts = (await session.exec(select(Appointment))).all()
            assert len(appts) == 5

            # Verify specific appointments exist
            titles = {a.title for a in appts}
            assert "Doctor's Appointment" in titles
            assert "Team Sync" in titles
            assert "Lunch with Sarah" in titles
            assert "Marketing Team Meeting" in titles
            assert "Dentist Appointment" in titles


@requires_live_model
async def test_predict_fn_returns_response():
    """Verify predict_fn returns a string response."""
    async with evaluate.evaluation_context():
        response = await evaluate.predict_fn("What appointments do I have tomorrow?")
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention tomorrow's appointments
        lowered = response.lower()
        assert "doctor" in lowered or "team sync" in lowered


def _list_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            titles = ", ".join(a.title for a in part.content)
            return ModelResponse(parts=[TextPart(f"You have: {titles}")])
    return ModelResponse(parts=[ToolCallPart("list_appointments", {})])


async def test_predict_fn_without_outer_context():
    """MLflow calls predict_fn bare, with no evaluation_context around it."""
    with appointment_agent.override(model=FunctionModel(_list_then_answer)):
        response = await evaluate.predict_fn("What appointments do I have?")
    assert isinstance(response, str)
    assert "Doctor's Appointment" in response


def test_happy_path_judges_answer_yes_no():
    """MLflow only averages numbers or yes/no strings into run metrics."""
    for judge in (
        evaluate.groundedness_judge,
        evaluate.correctness_judge,
        evaluate.retrieval_recall_judge,
    ):
        assert typing.get_args(judge._feedback_value_type) == ("yes", "no")
