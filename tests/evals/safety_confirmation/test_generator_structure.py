
import pytest
from pydantic import BaseModel


def test_generator_models_exist():
    # This import is expected to fail initially
    try:
        from tests.evals.safety_confirmation.generator import (
            SyntheticDataset,
            SyntheticExample,
        )
    except ImportError:
        pytest.fail("Could not import SyntheticExample or SyntheticDataset from tests.evals.safety_confirmation.generator")
    
    assert issubclass(SyntheticExample, BaseModel)
    assert issubclass(SyntheticDataset, BaseModel)
    
    # Verify fields of SyntheticExample
    fields = SyntheticExample.model_fields
    assert "user_prompt" in fields
    assert "expected_category" in fields
    assert "action_type" in fields

def test_generator_agent_exists():
    try:
        from tests.evals.safety_confirmation.generator import generator_agent
    except ImportError:
        pytest.fail("Could not import generator_agent from tests.evals.safety_confirmation.generator")

    assert generator_agent is not None
