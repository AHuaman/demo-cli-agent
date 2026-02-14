from typing import Literal

from pydantic import BaseModel


class SyntheticInputs(BaseModel):
    user_prompt: str


class SyntheticExpectations(BaseModel):
    operation_type: Literal["CREATE", "READ", "UPDATE", "DELETE"]
    expected_tool: str
    expected_appointments: list[str]
    expected_answer_contains: list[str]


class SyntheticExample(BaseModel):
    inputs: SyntheticInputs
    expectations: SyntheticExpectations


class SyntheticDataset(BaseModel):
    examples: list[SyntheticExample]
