# This script scores model completions against the ground-truth integer answers of a dataset split.
# It parses the integer between the <answer> tags of each completion, computes the accuracy in percent,
# and saves the accuracy together with every question, completion, parsed integer and ground truth as a JSON report.

import re
from pathlib import Path

from scripts.inference.models import Inference, InferenceReport, ScoredInference


def save_scored_inference_report(
    inferences: list[Inference],
    ground_truth_integer_answers: list[int],
    output_path: Path,
) -> None:
    scored_inferences = [
        score_inference(inference, ground_truth_integer_answer)
        for inference, ground_truth_integer_answer in zip(inferences, ground_truth_integer_answers, strict=True)
    ]
    accuracy = compute_accuracy_percent(scored_inferences)

    inference_report = InferenceReport(accuracy=accuracy, scored_inferences=scored_inferences)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(inference_report.model_dump_json(indent=2))


def score_inference(inference: Inference, ground_truth_integer_answer: int) -> ScoredInference:
    return ScoredInference(
        question=inference.question,
        answer=inference.answer,
        parsed_integer_answer=parse_integer_answer(inference.answer),
        ground_truth_integer_answer=ground_truth_integer_answer,
    )


def compute_accuracy_percent(scored_inferences: list[ScoredInference]) -> float:
    correct_count = sum(
        scored_inference.parsed_integer_answer == scored_inference.ground_truth_integer_answer
        for scored_inference in scored_inferences
    )
    return 100 * correct_count / len(scored_inferences)


def parse_integer_answer(completion: str) -> int | None:
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if answer_match is None:
        return None

    answer_text = re.sub(r"\s", "", answer_match.group(1))
    if not re.fullmatch(r"-?\d+", answer_text):
        return None
    return int(answer_text)
