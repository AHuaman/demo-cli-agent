import pytest

from src import providers
from tests.evals.conflict_resolution import evaluate as conflict_eval
from tests.evals.happy_path_operations import evaluate as happy_eval
from tests.evals.safety_confirmation import evaluate as safety_eval

JUDGES = [
    conflict_eval.conflict_judge,
    happy_eval.groundedness_judge,
    happy_eval.correctness_judge,
    happy_eval.retrieval_recall_judge,
    safety_eval.safety_judge,
]


@pytest.mark.parametrize("judge", JUDGES, ids=lambda j: j.name)
def test_judge_uses_shared_endpoint(judge):
    assert judge.model == providers.judge_model_uri
    # Checked outside the assert so pytest never prints the params (API key).
    wired = judge.inference_params is providers.judge_inference_params
    assert wired, "judge must use providers.judge_inference_params"
