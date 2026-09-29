# This script fine-tunes every parameter of Qwen2.5-1.5B-Instruct on the GLM 5.3 teacher traces that reached the
# correct answer. Each example uses the XML system prompt and the question as the prompt, and the teacher's
# <reasoning> and <answer> as the completion; the loss is computed on the completion only.
# Throughout training, the validation loss is measured on the correct validation traces and the validation accuracy
# is measured by generating answers to the validation questions. Training stops early once the validation accuracy
# no longer improves, and the most accurate model and its loss history are saved to models/Qwen1_5_SFT_GLM/.

import json
from pathlib import Path

from datasets import Dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from scripts.inference.models import InferenceReport, ScoredInference
from scripts.inference.prompts import XML_SYSTEM_PROMPT
from scripts.train.validation_accuracy_callback import ValidationAccuracyCallback

BASE_MODEL_DIRECTORY = Path("models/Qwen2.5-1.5B-Instruct")


def main() -> None:
    train_dataset = build_prompt_completion_dataset(Path("scripts/inference/results/glm53_train.json"))
    validation_dataset = build_prompt_completion_dataset(Path("scripts/inference/results/glm53_val.json"))
    validation_accuracy_callback = build_validation_accuracy_callback(Path("dataset/processed/val.json"))

    sft_trainer = build_sft_trainer(train_dataset, validation_dataset, validation_accuracy_callback)
    sft_trainer.train()

    output_directory = Path("models/Qwen1_5_SFT_GLM")
    sft_trainer.save_model(output_directory)
    save_loss_history(sft_trainer.state.log_history, output_directory / "loss_history.json")


def build_prompt_completion_dataset(inference_report_path: Path) -> Dataset:
    inference_report = InferenceReport.model_validate_json(inference_report_path.read_text())
    correct_scored_inferences = [
        scored_inference
        for scored_inference in inference_report.scored_inferences
        if scored_inference.parsed_integer_answer == scored_inference.ground_truth_integer_answer
    ]
    prompt_completion_examples = [
        convert_to_prompt_completion_example(scored_inference) for scored_inference in correct_scored_inferences
    ]
    return Dataset.from_list(prompt_completion_examples)


def convert_to_prompt_completion_example(scored_inference: ScoredInference) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": [
            {"role": "system", "content": XML_SYSTEM_PROMPT},
            {"role": "user", "content": scored_inference.question},
        ],
        "completion": [{"role": "assistant", "content": scored_inference.answer}],
    }


def build_validation_accuracy_callback(validation_examples_path: Path) -> ValidationAccuracyCallback:
    validation_examples = json.loads(validation_examples_path.read_text())
    questions = [validation_example["question"] for validation_example in validation_examples]
    ground_truth_integer_answers = [validation_example["integer_answer"] for validation_example in validation_examples]
    # The callback is our generation-based validation metric, which TRL does not provide for SFT: at every evaluation it
    # makes the model currently being trained answer all validation questions greedily, scores the answers with the same
    # <answer> parser as the inference reports, and publishes the result as eval_accuracy. That metric drives early
    # stopping and best checkpoint selection, and is saved in the loss history for the accuracy curve.
    # BASE_MODEL_DIRECTORY is only used to load the tokenizer; the weights that answer are the live training weights.
    return ValidationAccuracyCallback(BASE_MODEL_DIRECTORY, questions, ground_truth_integer_answers)


def build_sft_trainer(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    validation_accuracy_callback: ValidationAccuracyCallback,
) -> SFTTrainer:
    sft_config = SFTConfig(
        output_dir="models/Qwen1_5_SFT_GLM/checkpoints",
        model_init_kwargs={"attn_implementation": "eager"},
        completion_only_loss=True,  # cc: gabriel, here is the loss masking we discussed this morning, handled natively by trl
        max_length=None,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        eval_on_start=True,
        per_device_eval_batch_size=32,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="none",
    )
    return SFTTrainer(
        model=str(BASE_MODEL_DIRECTORY),
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[validation_accuracy_callback, EarlyStoppingCallback(early_stopping_patience=4)],
    )


def save_loss_history(log_history: list[dict[str, float]], output_path: Path) -> None:
    loss_history = {
        "train_loss": extract_curve(log_history, "loss"),
        "eval_loss": extract_curve(log_history, "eval_loss"),
        "eval_accuracy": extract_curve(log_history, "eval_accuracy"),
    }
    output_path.write_text(json.dumps(loss_history, indent=2))


def extract_curve(log_history: list[dict[str, float]], metric_name: str) -> list[dict[str, float]]:
    return [
        {"step": log_entry["step"], "value": log_entry[metric_name]}
        for log_entry in log_history
        if metric_name in log_entry
    ]


if __name__ == "__main__":
    main()
