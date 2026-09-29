# This script provides ValidationAccuracyCallback, a trainer callback that measures the model's accuracy on the
# validation questions every time the trainer evaluates. It keeps its own bfloat16 copy of the model, loads the
# current training weights into it, generates an answer greedily for every validation question using the XML system
# prompt, scores it with the same parser as the inference reports, and adds the accuracy to the evaluation metrics as
# eval_accuracy, so that early stopping, best model selection and the loss history can all use it. Generation uses
# continuous batching with a KV cache capped at half of the free GPU memory, so it neither waits on the longest answer
# nor competes with the training state for memory.

from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from scripts.inference.batched_inference import generate_greedy_completions_with_continuous_batching
from scripts.inference.models import Inference
from scripts.inference.parse_score_save_helpers import compute_accuracy_percent, score_inference
from scripts.inference.prompts import XML_SYSTEM_PROMPT


class ValidationAccuracyCallback(TrainerCallback):
    def __init__(self, model_directory: Path, questions: list[str], ground_truth_integer_answers: list[int]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.generation_model = AutoModelForCausalLM.from_pretrained(
            model_directory, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.questions = questions
        self.ground_truth_integer_answers = ground_truth_integer_answers

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        metrics: dict[str, float],
        **kwargs,
    ) -> None:
        accuracy = self.measure_validation_accuracy(model)
        metrics["eval_accuracy"] = accuracy
        state.log_history.append({"eval_accuracy": accuracy, "step": state.global_step})

    def measure_validation_accuracy(self, model: PreTrainedModel) -> float:
        self.generation_model.load_state_dict(model.state_dict())
        completions = generate_greedy_completions_with_continuous_batching(
            self.generation_model,
            self.tokenizer,
            XML_SYSTEM_PROMPT,
            self.questions,
            max_new_tokens=1024,
            max_kv_cache_memory_fraction=0.5,
        )

        scored_inferences = [
            score_inference(Inference(question=question, answer=completion), ground_truth_integer_answer)
            for question, completion, ground_truth_integer_answer in zip(
                self.questions, completions, self.ground_truth_integer_answers, strict=True
            )
        ]
        return compute_accuracy_percent(scored_inferences)
