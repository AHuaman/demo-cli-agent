"""Drive the appointment agent through the real approval flow for multi-turn
safety checks: ask -> (optional text confirmation) -> approve/deny -> re-run.

Shared by `test_multiturn.py` (behavioral assertions) and
`multiturn_evaluate.py` (MLflow judge scoring).
"""

from collections.abc import Awaitable, Callable
from datetime import datetime

from pydantic_ai import AgentRunResult, DeferredToolRequests
from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.agent import appointment_agent
from src.approvals import build_results
from src.models import Appointment
from src.negotiator import negotiator_agent
from src.tools import Deps

FIXED_NOW = datetime(2026, 2, 11, 10, 0, 0)

# Some models describe a write and ask in plain text before calling the tool.
# The driver answers like a user would, then requires the approval gate.
CONFIRM_REPLY = "Yes, go ahead."
MAX_CONFIRM_TURNS = 2


async def seeded_factory() -> async_sessionmaker:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with factory() as s:
        s.add(Appointment(title="Doctor's Appointment",
                          start_time=datetime(2026, 2, 12, 14, 0),
                          end_time=datetime(2026, 2, 12, 15, 0)))
        await s.commit()
    return factory


async def drive(
    factory: async_sessionmaker,
    prompt: str,
    history: list[ModelMessage],
    approve: bool,
    on_gate: Callable[[DeferredToolRequests], Awaitable[None]] | None = None,
) -> tuple[AgentRunResult, bool]:
    """Run `prompt` to completion and return `(final_result, asked)`.

    `asked` is True when an approval gate fired. `on_gate` runs at the first
    gate, before the approve/deny decision is applied.
    """
    async with factory() as session:
        deps = Deps(
            session=session, negotiator_agent=negotiator_agent, now=lambda: FIXED_NOW
        )
        result = await appointment_agent.run(prompt, message_history=history, deps=deps)
        for _ in range(MAX_CONFIRM_TURNS):
            if not isinstance(result.output, str):
                break
            result = await appointment_agent.run(
                CONFIRM_REPLY, message_history=result.all_messages(), deps=deps
            )
        asked = isinstance(result.output, DeferredToolRequests)
        if asked and on_gate is not None:
            await on_gate(result.output)
        while isinstance(result.output, DeferredToolRequests):
            results = build_results(result.output, approved=approve)
            result = await appointment_agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=results, deps=deps,
            )
        return result, asked
