# This script runs GLM 5.3 Flash through the OpenRouter API with the teacher system prompt on the validation
# and train sets, scores each completion against the ground-truth integer answer, and saves one scored report
# per split to scripts/inference/results/glm53_val.json and glm53_train.json.
# The train report is the source of the reasoning traces used later for fine-tuning.

import json
import os
from pathlib import Path

from scripts.inference.batched_inference import run_inference_from_openrouter_api
from scripts.inference.parse_score_save_helpers import save_scored_inference_report
from scripts.inference.prompts import TEACHER_XML_SYSTEM_PROMPT


def main() -> None:
    openrouter_api_key = os.environ["OPENROUTER_API_KEY"]

    run_glm53_inference_and_save_report(
        openrouter_api_key,
        Path("dataset/processed/val.json"),
        Path("scripts/inference/results/glm53_val.json"),
    )
    run_glm53_inference_and_save_report(
        openrouter_api_key,
        Path("dataset/processed/train.json"),
        Path("scripts/inference/results/glm53_train.json"),
    )


def run_glm53_inference_and_save_report(openrouter_api_key: str, dataset_path: Path, output_path: Path) -> None:
    examples = json.loads(dataset_path.read_text())
    questions = [example["question"] for example in examples]
    ground_truth_integer_answers = [example["integer_answer"] for example in examples]

    inferences = run_inference_from_openrouter_api(
        openrouter_api_key,
        "z-ai/glm-5.3-flash",
        TEACHER_XML_SYSTEM_PROMPT,
        questions,
        preferred_providers=("z-ai",),
    )
    save_scored_inference_report(inferences, ground_truth_integer_answers, output_path)


if __name__ == "__main__":
    main()
