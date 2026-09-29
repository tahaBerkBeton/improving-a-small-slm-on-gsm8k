# This script picks the worked examples shown in the README from the scored validation reports. It finds validation
# questions that the baseline model got wrong but the teacher and the SFT model got right, and, when the GRPO report
# exists, questions that the baseline and the SFT model got wrong but the teacher and the GRPO model got right.
# Among the candidates it prefers the shortest completions, so the examples stay readable. It also renders the two
# hand-picked baseline-versus-zero examples (questions that only the zero model solved, chosen by hand for how clearly
# they show the change in reasoning) and writes everything to plot/readme_examples.md with the question and every
# model's completion.

import json
from pathlib import Path

from pydantic import BaseModel

RESULTS_DIRECTORY = Path("scripts/inference/results")
EXAMPLES_PER_SECTION = 2
HAND_PICKED_ZERO_EXAMPLE_INDICES = (464, 436)


class ScoredInference(BaseModel):
    question: str
    answer: str
    parsed_integer_answer: int | None
    ground_truth_integer_answer: int


def main() -> None:
    baseline = read_scored_inferences("prime_val.json")
    teacher = read_scored_inferences("glm53_val.json")
    sft = read_scored_inferences("sft_glm53_val.json")

    sections = [
        (
            "Fixed by SFT: baseline wrong, teacher and SFT right",
            {"Baseline (Qwen2.5-1.5B-Instruct)": baseline, "Teacher (GLM 5.3 Flash)": teacher, "SFT": sft},
            [index for index in range(len(baseline)) if not is_correct(baseline[index]) and is_correct(teacher[index]) and is_correct(sft[index])],
        )
    ]
    grpo_report_path = RESULTS_DIRECTORY / "sft_glm53_grpo_val.json"
    if grpo_report_path.exists():
        grpo = read_scored_inferences("sft_glm53_grpo_val.json")
        sections.append(
            (
                "Fixed by GRPO: baseline and SFT wrong, teacher and GRPO right",
                {"Baseline (Qwen2.5-1.5B-Instruct)": baseline, "Teacher (GLM 5.3 Flash)": teacher, "SFT": sft, "SFT + GRPO": grpo},
                [index for index in range(len(baseline)) if not is_correct(baseline[index]) and not is_correct(sft[index]) and is_correct(teacher[index]) and is_correct(grpo[index])],
            )
        )

    rendered_sections = [render_section(title, model_inferences, candidate_indices) for title, model_inferences, candidate_indices in sections]
    zero_report_path = RESULTS_DIRECTORY / "zero_val.json"
    if zero_report_path.exists():
        rendered_sections.append(render_zero_section(baseline, sft, read_scored_inferences("zero_val.json")))

    Path("plot/readme_examples.md").write_text("\n\n".join(rendered_sections) + "\n")


def read_scored_inferences(report_name: str) -> list[ScoredInference]:
    report = json.loads((RESULTS_DIRECTORY / report_name).read_text())
    return [ScoredInference.model_validate(scored) for scored in report["scored_inferences"]]


def is_correct(scored_inference: ScoredInference) -> bool:
    return scored_inference.parsed_integer_answer == scored_inference.ground_truth_integer_answer


def render_section(title: str, model_inferences: dict[str, list[ScoredInference]], candidate_indices: list[int]) -> str:
    def total_completion_length(index: int) -> int:
        return sum(len(inferences[index].answer) for inferences in model_inferences.values())

    chosen_indices = sorted(candidate_indices, key=total_completion_length)[:EXAMPLES_PER_SECTION]
    rendered_examples = [render_example(model_inferences, index) for index in chosen_indices]
    return f"## {title}\n\n({len(candidate_indices)} such questions; showing {len(chosen_indices)})\n\n" + "\n\n".join(rendered_examples)


def render_zero_section(baseline: list[ScoredInference], sft: list[ScoredInference], zero: list[ScoredInference]) -> str:
    grpo = read_scored_inferences("sft_glm53_grpo_val.json")
    only_zero_indices = [
        index
        for index in range(len(baseline))
        if is_correct(zero[index]) and not is_correct(baseline[index]) and not is_correct(sft[index]) and not is_correct(grpo[index])
    ]
    model_inferences = {"Baseline (Qwen2.5-1.5B-Instruct)": baseline, "Zero (GRPO alone)": zero}
    rendered_examples = [render_example(model_inferences, index) for index in HAND_PICKED_ZERO_EXAMPLE_INDICES]
    return (
        f"## Solved only by zero: baseline, SFT and SFT + GRPO wrong, zero right\n\n"
        f"({len(only_zero_indices)} such questions; showing {len(HAND_PICKED_ZERO_EXAMPLE_INDICES)} hand-picked)\n\n"
        + "\n\n".join(rendered_examples)
    )


def render_example(model_inferences: dict[str, list[ScoredInference]], index: int) -> str:
    first_inferences = next(iter(model_inferences.values()))
    question = first_inferences[index].question
    ground_truth = first_inferences[index].ground_truth_integer_answer
    parts = [f"**Question.** {question}\n\n**Gold answer.** {ground_truth}"]
    for model_name, inferences in model_inferences.items():
        scored = inferences[index]
        verdict = "correct" if is_correct(scored) else f"wrong (parsed {scored.parsed_integer_answer})"
        parts.append(f"**{model_name}** — {verdict}\n\n```\n{scored.answer.strip()}\n```")
    return "\n\n".join(parts)


if __name__ == "__main__":
    main()
