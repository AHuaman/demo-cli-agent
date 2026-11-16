import os

# MLflow scores up to 10 rows at once by default. Slow hosted judges (e.g. large
# models on NVIDIA NIM's free tier) queue that many parallel requests until the
# run stalls, so evals default to 2. An explicit env value still wins.
_DEFAULT_MAX_WORKERS = "2"


def cap_eval_workers() -> None:
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", _DEFAULT_MAX_WORKERS)
