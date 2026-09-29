# This script runs Qwen2.5-1.5B-Instruct fine-tuned on the GLM 5.3 teacher traces (models/Qwen1_5_SFT_GLM) on the
# validation set with the XML system prompt, scores each completion against the ground-truth integer answer, and saves
# the scored report to scripts/inference/results/sft_glm53_val.json.

import json
from pathlib import Path

from scripts.inference.batched_inference import run_batch_inference_local
from scripts.inference.parse_score_save_helpers import save_scored_inference_report
from scripts.inference.prompts import XML_SYSTEM_PROMPT


def main() -> None:
    validation_examples = json.loads(Path("dataset/processed/val.json").read_text())
    questions = [validation_example["question"] for validation_example in validation_examples]
    ground_truth_integer_answers = [validation_example["integer_answer"] for validation_example in validation_examples]

    inferences = run_batch_inference_local(Path("models/Qwen1_5_SFT_GLM"), XML_SYSTEM_PROMPT, questions)
    save_scored_inference_report(inferences, ground_truth_integer_answers, Path("scripts/inference/results/sft_glm53_val.json"))


if __name__ == "__main__":
    main()
