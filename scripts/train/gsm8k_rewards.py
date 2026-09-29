# This script provides the GRPO reward functions for GSM8K answers written in the <reasoning> / <answer> format.
# They follow the reward set of Unsloth's Qwen2.5 GRPO notebook: a correct integer answer earns 2.0, and four format
# rewards of at most 0.5 each keep the model inside the expected format. Each function receives a batch of sampled
# completions and returns one reward per completion; GRPO adds the rewards of all functions together.
# Compared to the notebook, correctness compares integers with the same parser as our benchmark, the format patterns
# also match reasoning that spans several lines, and a completion may end right after </answer>.

import re

from scripts.inference.parse_score_save_helpers import parse_integer_answer


def reward_correct_integer_answer(
    completions: list[list[dict[str, str]]],
    ground_truth_integer_answer: list[int],
    **kwargs,
) -> list[float]:
    completion_texts = extract_completion_texts(completions)
    return [
        2.0 if parse_integer_answer(completion_text) == ground_truth else 0.0
        for completion_text, ground_truth in zip(completion_texts, ground_truth_integer_answer, strict=True)
    ]


def reward_integer_answer(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    completion_texts = extract_completion_texts(completions)
    return [0.5 if parse_integer_answer(completion_text) is not None else 0.0 for completion_text in completion_texts]


def reward_strict_xml_format(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    completion_texts = extract_completion_texts(completions)
    strict_xml_format = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s*"
    return [
        0.5 if re.fullmatch(strict_xml_format, completion_text, re.DOTALL) else 0.0
        for completion_text in completion_texts
    ]


def reward_soft_xml_format(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    completion_texts = extract_completion_texts(completions)
    soft_xml_format = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [
        0.5 if re.match(soft_xml_format, completion_text, re.DOTALL) else 0.0
        for completion_text in completion_texts
    ]


def reward_xml_tag_counts(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    completion_texts = extract_completion_texts(completions)
    return [score_xml_tag_counts(completion_text) for completion_text in completion_texts]


def score_xml_tag_counts(completion_text: str) -> float:
    well_formed_tag_count = sum(
        completion_text.count(xml_tag) == 1
        for xml_tag in ("<reasoning>\n", "\n</reasoning>\n", "\n<answer>\n", "\n</answer>")
    )
    text_after_answer = completion_text.split("</answer>")[-1] if "</answer>" in completion_text else ""
    return 0.125 * well_formed_tag_count - 0.001 * len(text_after_answer.strip())


def extract_completion_texts(completions: list[list[dict[str, str]]]) -> list[str]:
    return [completion[0]["content"] for completion in completions]


GSM8K_REWARD_FUNCTIONS = [
    reward_xml_tag_counts,
    reward_soft_xml_format,
    reward_strict_xml_format,
    reward_integer_answer,
    reward_correct_integer_answer,
]
