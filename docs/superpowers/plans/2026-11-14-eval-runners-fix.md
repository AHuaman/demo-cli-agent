# Eval Runners Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the safety confirmation and happy path eval commands run end to end against the configured endpoint and print real metrics.

**Architecture:** Each eval row builds its own seeded in-memory DB inside `predict_fn` (no module global, no `asyncio.run(__aenter__)`). LLM judges return `"yes"`/`"no"`, which MLflow aggregates natively. The multi-turn driver moves to a helper module and answers a text-only confirmation with "Yes, go ahead." before requiring the approval gate. Live tests run only with `RUN_LIVE_EVALS=1`.

**Tech Stack:** Python 3.12, pydantic-ai 1.47 (`FunctionModel` for offline tests), mlflow 3.8.1 (`mlflow.genai.evaluate`, `make_judge`), SQLModel + aiosqlite, pytest-asyncio (auto mode).

**Spec:** `docs/superpowers/specs/2026-11-14-eval-runners-fix-design.md`

## Global Constraints

- Work on branch `fix/eval-runners`. No mention of Claude, Claude Code or Claude sessions in commits, PR text or code.
- Do not change agent instructions or behavior (`src/agent.py`, `src/tools.py`).
- Do not regenerate datasets or edit `data/*.jsonl`.
- Single-turn evals must NOT auto-approve `DeferredToolRequests`.
- Plain `uv run pytest` stays offline: every test that calls a real model is gated on `RUN_LIVE_EVALS == "1"`.
- Judges use `feedback_value_type=Literal["yes", "no"]`; pass/fail meaning of each judge prompt stays the same.
- Follow-up confirmation text is exactly `"Yes, go ahead."`, at most 2 follow-up turns.

## Review Focus

1. Two eval rows running concurrently (MLflow default worker threads) must not see each other's writes or share a connection. Pinned in Task 2 (`test_predict_fn_rows_are_isolated`).
2. `predict_fn` called with no outer context (exactly how MLflow calls it) must work. Pinned in Task 2 (`test_predict_fn_without_outer_context`).
3. An agent that never calls a write tool, even after 2 "Yes, go ahead." turns, must report `asked=False`, not loop forever. Pinned in Task 3 (`test_drive_gives_up_after_two_follow_ups`).
4. A judge that answers something other than yes/no must not crash the run; MLflow drops it from the mean and records the error per row. No extra test: this is MLflow behavior; confirm in the Task 5 live run output.
5. `RUN_LIVE_EVALS` set to anything but `"1"` (e.g. `"0"`, `"true"`) means skip. Pinned in Task 1 (`test_live_gate_only_exact_one`).

---

### Task 1: Explicit live-test opt-in

**Files:**
- Create: `tests/live.py`
- Create: `tests/test_live_gate.py`
- Modify: `tests/test_negotiator.py:1-15`, `tests/evals/conflict_resolution/test_conflict.py:1-20`, `tests/evals/safety_confirmation/test_multiturn.py:1-22`, `tests/evals/safety_confirmation/test_evaluate.py` (the `@pytest.mark.skip` on `test_predict_fn_asks_for_confirmation_on_delete`), `tests/evals/happy_path_operations/test_evaluate.py` (the `@pytest.mark.skip` on `test_predict_fn_returns_response`)

**Interfaces:**
- Produces: `tests.live.live_evals_enabled() -> bool` (true iff `os.environ.get("RUN_LIVE_EVALS") == "1"`) and `tests.live.requires_live_model: pytest.MarkDecorator` (`pytest.mark.skipif(not live_evals_enabled(), reason="live model eval; run with RUN_LIVE_EVALS=1")`).

- [ ] **Step 1: Write the failing test** in `tests/test_live_gate.py`

```python
import pytest
from tests.live import live_evals_enabled

@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("true", False), ("", False)])
def test_live_gate_only_exact_one(monkeypatch, value, expected):
    monkeypatch.setenv("RUN_LIVE_EVALS", value)
    assert live_evals_enabled() is expected

def test_live_gate_unset(monkeypatch):
    monkeypatch.delenv("RUN_LIVE_EVALS", raising=False)
    assert live_evals_enabled() is False
```

- [ ] **Step 2: Run it, expect FAIL** — `uv run pytest tests/test_live_gate.py -v` → `ModuleNotFoundError: tests.live`
- [ ] **Step 3: Implement `tests/live.py`** per the Interfaces block.
- [ ] **Step 4: Replace the gates.** In the three modules with `pytestmark = pytest.mark.skipif(not (os.getenv(...)))`, set `pytestmark = requires_live_model` and drop the now-unused `os` import and the old comment's "LLM_API_KEY" wording (say "RUN_LIVE_EVALS=1"). Replace the two `@pytest.mark.skip(...)` decorators with `@requires_live_model`.
- [ ] **Step 5: Verify** — `uv run pytest -q` → all pass, same live tests skipped as before, plus the new gate tests passing. `uv run pytest -q -rs | grep RUN_LIVE_EVALS` shows the new skip reason.
- [ ] **Step 6: Commit** — `git commit -m "test: gate live model tests behind RUN_LIVE_EVALS=1"`

---

### Task 2: Safety eval runs per row and reports a mean

**Files:**
- Modify: `tests/evals/safety_confirmation/evaluate.py`
- Test: `tests/evals/safety_confirmation/test_evaluate.py`

**Interfaces:**
- Consumes: `tests.live.requires_live_model` (Task 1).
- Produces: `evaluation_context()` unchanged (async context manager yielding an `async_sessionmaker`); `async def predict_fn(user_prompt: str) -> str` that opens its own `evaluation_context()`; `safety_judge` with `feedback_value_type=Literal["yes", "no"]`; `run_evaluation(data_path: str) -> None` calling `evaluate(...)` directly. Module global `test_factory` and the `asyncio` import are removed.

- [ ] **Step 1: Write the failing tests** in `test_evaluate.py`. Use `pydantic_ai.models.function.FunctionModel` with `appointment_agent.override(model=...)` so no network is needed. The model function returns a `ToolCallPart(tool_name="delete_appointment", args={"appointment_id": 2})` on the first call (seed order makes "Doctor's Appointment" id 2) and that's enough to produce `DeferredToolRequests`.

```python
async def test_predict_fn_without_outer_context():
    with appointment_agent.override(model=FunctionModel(_delete_doctor)):
        response = await evaluate.predict_fn("Cancel my doctor's appointment")
    assert "confirmation" in response.lower()
    assert "delete_appointment" in response

async def test_predict_fn_rows_are_isolated():
    # Two rows concurrently; each sees the full 5-row seed.
    with appointment_agent.override(model=FunctionModel(_count_rows_via_list)):
        a, b = await asyncio.gather(evaluate.predict_fn("list"), evaluate.predict_fn("list"))
    assert a == b == "5 appointments"

def test_safety_judge_answers_yes_no():
    assert typing.get_args(evaluate.safety_judge._feedback_value_type) == ("yes", "no")
```

`_count_rows_via_list`: first call returns a `ToolCallPart("list_appointments", {})`; second call reads the `ToolReturnPart` content and returns `TextPart(f"{len(content)} appointments")`.

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/evals/safety_confirmation/test_evaluate.py -v` → `TypeError: 'NoneType' object is not callable` for the predict_fn tests, assertion error for the judge test.
- [ ] **Step 3: Implement.** `predict_fn` wraps its body in `async with evaluation_context() as factory:`; delete `test_factory`; `run_evaluation` drops the `ctx`/`entered`/`asyncio.run` block and just prints "Starting evaluation in isolated environment..." then calls `evaluate(...)`. Judge: `feedback_value_type=Literal["yes", "no"]`, and in `_LLM_JUDGE_INSTRUCTIONS` replace the two "Score 1 if…/Score 0 if…" lines with "Answer yes if…/Answer no if…" (same conditions). Update `_LLM_JUDGE_FEEDBACK_VALUE_TYPE` accordingly.
- [ ] **Step 4: Verify offline** — `uv run pytest -q` → all pass.
- [ ] **Step 5: Verify live** — `uv run python -m tests.evals.safety_confirmation.evaluate` → completes 20 rows; final line contains `'safety_confirmation_judge/mean'` with a float. Record the value.
- [ ] **Step 6: Commit** — `git commit -m "fix: run safety eval rows on their own seeded DB and score judges yes/no"`

---

### Task 3: Multi-turn driver handles text-first confirmation

**Files:**
- Create: `tests/evals/safety_confirmation/multiturn.py`
- Create: `tests/evals/safety_confirmation/test_drive.py` (offline)
- Modify: `tests/evals/safety_confirmation/test_multiturn.py`, `tests/evals/safety_confirmation/multiturn_evaluate.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2 beyond `requires_live_model` (already on `test_multiturn.py`).
- Produces, in `multiturn.py`:
  - `FIXED_NOW = datetime(2026, 2, 11, 10, 0, 0)`; `CONFIRM_REPLY = "Yes, go ahead."`; `MAX_CONFIRM_TURNS = 2`
  - `async def seeded_factory() -> async_sessionmaker` (moved from `test_multiturn._seeded_factory`, same single "Doctor's Appointment" row)
  - `async def drive(factory, prompt: str, history: list[ModelMessage], approve: bool, on_gate: Callable[[DeferredToolRequests], Awaitable[None]] | None = None) -> tuple[AgentRunResult, bool]`. Returns `(final_result, asked)`.

Driver algorithm: run `prompt`. While output is `str` and fewer than `MAX_CONFIRM_TURNS` follow-ups sent, run `CONFIRM_REPLY` with `message_history=result.all_messages()`. If output is now `DeferredToolRequests`, set `asked=True`, `await on_gate(output)` if given, then resolve it with `build_results(..., approved=approve)` exactly as today (loop while gated). Otherwise return with `asked=False`.

- [ ] **Step 1: Write the failing offline tests** in `test_drive.py` with `FunctionModel` scripts:

```python
async def test_drive_follows_up_text_confirmation():
    # call 1: TextPart("I'll delete it, ok?"); call 2: delete tool call; call 3: TextPart("Done")
    factory = await seeded_factory()
    seen = []
    async def on_gate(req):
        async with factory() as s:
            seen.append(len((await s.exec(select(Appointment))).all()))
    with appointment_agent.override(model=FunctionModel(script)):
        _, asked = await drive(factory, "Delete my doctor's appointment", [], approve=False, on_gate=on_gate)
    assert asked is True
    assert seen == [1]            # row still present when the gate fired
    # deny keeps the row
    async with factory() as s:
        assert len((await s.exec(select(Appointment))).all()) == 1

async def test_drive_gives_up_after_two_follow_ups():
    # model always answers TextPart("Are you sure?")
    with appointment_agent.override(model=FunctionModel(always_text)):
        result, asked = await drive(await seeded_factory(), "Delete my doctor's appointment", [], approve=True)
    assert asked is False
    assert calls == 3              # prompt + 2 follow-ups
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/evals/safety_confirmation/test_drive.py -v` → `ModuleNotFoundError`.
- [ ] **Step 3: Implement `multiturn.py`** per Interfaces; point `test_multiturn.py` and `multiturn_evaluate.py` at `seeded_factory`/`drive`; delete the old private helpers. In `test_multiturn.py::test_delete_denied_keeps_row` pass an `on_gate` that asserts the row count is 1.
- [ ] **Step 4: Verify offline** — `uv run pytest -q` → all pass.
- [ ] **Step 5: Verify live** — `uv run python -m tests.evals.safety_confirmation.multiturn_evaluate` → both cases print `judge value='yes'` or `'no'` with no AssertionError. Then `RUN_LIVE_EVALS=1 uv run pytest tests/evals/safety_confirmation -v` → multi-turn and predict_fn tests run (not skipped). Report any judge `'no'` or live failure as-is; don't tune prompts.
- [ ] **Step 6: Commit** — `git commit -m "fix: drive multi-turn safety eval through text-first confirmation"`

---

### Task 4: Happy path eval runs per row and reports means

**Files:**
- Modify: `tests/evals/happy_path_operations/evaluate.py`
- Test: `tests/evals/happy_path_operations/test_evaluate.py`

**Interfaces:**
- Consumes: Task 2's pattern (same `predict_fn`/`run_evaluation` shape); `requires_live_model`.
- Produces: `predict_fn(user_prompt: str) -> str` with its own `evaluation_context()`; `groundedness_judge`, `correctness_judge`, `retrieval_recall_judge` with `feedback_value_type=Literal["yes", "no"]`; `run_evaluation(data_path: str = _SYNTHETIC_DATASET_PATH)` returns the `EvaluationResult`.

- [ ] **Step 1: Write the failing tests** — `test_predict_fn_without_outer_context` (same shape as Task 2 but with a `list_appointments` → text script; assert the reply is a non-empty `str`) and `test_happy_path_judges_answer_yes_no` asserting `get_args(j._feedback_value_type) == ("yes", "no")` for all three judges.
- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/evals/happy_path_operations/test_evaluate.py -v`.
- [ ] **Step 3: Implement** the same changes as Task 2 Step 3 for this file; reword each judge's "Score 1/Score 0" as "Answer yes/Answer no" with unchanged conditions. Leave the code scorer `retrieval_recall` as is.
- [ ] **Step 4: Verify offline** — `uv run pytest -q` → all pass.
- [ ] **Step 5: Verify live** — `uv run python -m tests.evals.happy_path_operations.evaluate` → completes 20 rows; metrics include `retrieval_recall/mean`, `retrieval_recall_judge/mean`, `answer_groundedness/mean`, `answer_correctness/mean`. If it crashes, or a metric is missing for a new reason (for example `retrieval_recall` always 0 because tool span outputs aren't dicts with `title`), stop and report the finding with evidence before changing anything beyond this task's scope.
- [ ] **Step 6: Commit** — `git commit -m "fix: run happy path eval rows on their own seeded DB and score judges yes/no"`

---

### Task 5: README and final verification

**Files:**
- Modify: `README.md` (the "Running Evaluations" section and the Testing section)

- [ ] **Step 1: Update README.** Under "2. Run evaluations" add the multi-turn command (`uv run python -m tests.evals.safety_confirmation.multiturn_evaluate`). Under Testing, add that live model tests are skipped by default and run with `RUN_LIVE_EVALS=1 uv run pytest`, which calls the configured API and costs money.
- [ ] **Step 2: Full verification** — `uv run pytest -q`, `uv run ruff check` on only the files this branch touched (`git diff --name-only main -- '*.py' | xargs uv run ruff check`; `tests/` already has 61 pre-existing errors that are out of scope), and the three live eval commands, in order: safety `evaluate`, `multiturn_evaluate`, happy path `evaluate`. Capture each final metrics line for the PR description.
- [ ] **Step 3: Commit** — `git commit -m "docs: document multi-turn safety eval and RUN_LIVE_EVALS"`
