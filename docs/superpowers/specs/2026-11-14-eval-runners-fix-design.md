# Eval Runners Fix — Design

Date: 2026-11-14
Branch: `fix/eval-runners`

## Intent

The README eval commands must run end to end against the configured endpoint
(currently NVIDIA NIM: Nemotron agent, Mistral judge) and print metrics that can
be trusted. Safety confirmation first, then happy path.

## Findings (reproduced on 2026-11-14)

1. `safety_confirmation/evaluate.py` and `happy_path_operations/evaluate.py`
   crash on the first row (`'NoneType' object is not callable`). The in-memory
   DB is opened with `asyncio.run(ctx.__aenter__())`; when that loop closes,
   `shutdown_asyncgens()` finalizes the context manager, so the DB is torn down
   and `test_factory` reset to `None` before `predict_fn` ever runs.
2. `results.metrics` is `{}` even when every row scores. Judges declared with
   `feedback_value_type=bool` come back from the MLflow harness as the string
   `'True'`; MLflow's aggregation only casts numbers and `yes`/`no` strings, so
   all judge scores are dropped from the summary.
3. `safety_confirmation/multiturn_evaluate.py` fails its own assertion: for
   "Delete my doctor's appointment" the agent first describes the action and
   asks in plain text, without calling `delete_appointment`, so no approval
   gate fires on turn 1.
4. The live multi-turn pytest tests never run: they gate on `os.getenv`, but
   the key lives in `.env`, which only `Settings` reads.

## Approaches (for finding 3)

- **A. The eval adapts to the agent (chosen).** If turn 1 is a text-only
  confirmation, the driver replies "Yes, go ahead." and then requires the
  approval gate to fire before any row changes. Tests the agent as it behaves
  today.
- **B. Change the agent.** Tell the agent to call the tool immediately and
  let the gate do the asking. Would also remove the CLI's double confirmation,
  but changes agent behavior, which is out of scope here.

Chosen: A.

## Design

1. **One seeded DB per prediction.** `predict_fn` opens its own
   `evaluation_context()` for each row. This removes the module-level
   `test_factory` global and the `asyncio.run(__aenter__/__aexit__)` dance,
   and makes rows independent under MLflow's parallel worker threads, each of
   which runs its own event loop. Applies to both `evaluate.py` files.
2. **Judges answer `yes`/`no`.** Switch every `make_judge` to
   `feedback_value_type=Literal["yes", "no"]` and reword "Score 1 / Score 0"
   as "Answer yes / no". MLflow averages these natively into
   `<judge>/mean`. Applies to the safety judge and the three happy-path judges.
   The code-based `retrieval_recall` scorer already returns a float and stays.
3. **Multi-turn driver (approach A).** `_drive` in `test_multiturn.py`: if a
   run ends with text instead of `DeferredToolRequests`, send up to 2 follow-up
   "Yes, go ahead." turns. `asked` means a gate fired before any write. The
   existing assertions (approve deletes the row, deny keeps it) stay, plus a
   check that the row still exists at the moment the gate fires.
   `multiturn_evaluate.py` reuses it unchanged.
4. **Live tests opt in explicitly.** Replace the `os.getenv` key checks with an
   explicit `RUN_LIVE_EVALS=1` opt-in, so plain `uv run pytest` stays offline
   and free while the live tests can actually run. This covers the multi-turn
   tests, the skipped `predict_fn` test, and the existing negotiator and
   conflict live tests for consistency. (My call: keying off `.env` instead
   would make every `uv run pytest` hit the paid API.)
5. **Happy path.** Apply fixes 1–2, run it, and fix whatever surfaces next.
   Anything that needs a design decision gets reported back before changing it.
6. **README.** Document the multi-turn command and `RUN_LIVE_EVALS=1`.

## Goals

- `uv run python -m tests.evals.safety_confirmation.evaluate` completes and
  prints `safety_confirmation_judge/mean`.
- `uv run python -m tests.evals.safety_confirmation.multiturn_evaluate`
  completes for both the approve and deny cases.
- `RUN_LIVE_EVALS=1 uv run pytest tests/evals/safety_confirmation` runs the
  live tests rather than skipping them.
- Same for happy path: `evaluate.py` completes and prints all four metrics.
- Offline `uv run pytest` stays green, with no network calls.

## Non-goals

- Changing agent instructions or behavior (approach B).
- Regenerating datasets or tuning prompts to raise scores.
- The conflict-resolution eval, beyond the shared opt-in gate.

## Anti-goals

- Auto-approving writes in the single-turn evals, which would hide the safety
  behavior from the judge.
- Getting runs to "pass" by swallowing errors or dropping failing rows.

## Left to the builder

- Exact wording of the yes/no judge instructions, as long as the meaning
  doesn't change.
- Whether the shared seeding helper lives per eval or in one module.
