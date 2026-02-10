import pytest


def test_evaluate_script_structure():
    try:
        from evaluate import predict_fn, run_evaluation, safety_judge
    except ImportError as e:
        pytest.fail(
            f"Could not import required components from evaluate.py: {e}"
        )

    assert safety_judge is not None
    assert callable(predict_fn)
    assert callable(run_evaluation)
