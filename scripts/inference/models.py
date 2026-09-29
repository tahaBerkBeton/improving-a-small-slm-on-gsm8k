# This script holds the pydantic models passed between the inference, scoring and saving steps.

from pydantic import BaseModel


class Inference(BaseModel):
    question: str
    answer: str


class ScoredInference(Inference):
    parsed_integer_answer: int | None
    ground_truth_integer_answer: int


class InferenceReport(BaseModel):
    accuracy: float
    scored_inferences: list[ScoredInference]
