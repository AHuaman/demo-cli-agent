import os

import pytest


def live_evals_enabled() -> bool:
    """Live model tests call the configured (paid) API, so they run only on an
    explicit opt-in: `RUN_LIVE_EVALS=1 uv run pytest`."""
    return os.environ.get("RUN_LIVE_EVALS") == "1"


requires_live_model = pytest.mark.skipif(
    not live_evals_enabled(),
    reason="live model eval; run with RUN_LIVE_EVALS=1",
)
