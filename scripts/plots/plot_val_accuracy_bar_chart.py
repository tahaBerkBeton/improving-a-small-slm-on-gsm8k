# This script draws a bar chart comparing validation accuracy across the models in this project: the untouched
# Qwen2.5-1.5B-Instruct, the GLM 5.3 Flash teacher, the Instruct model distilled on the teacher's traces (SFT), and
# the distilled model further trained with GRPO, and the Instruct model trained with GRPO alone ("zero", no SFT).
# Each bar is read from that model's scored report in
# scripts/inference/results/; a model whose report does not exist yet is left out. Every bar is labelled with its
# accuracy and its gain over the baseline, and the y axis starts just below the baseline so the gains are visible.
# The chart is saved to plot/val_accuracy_bar_chart.png.

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_REPORTS = {
    "Qwen2.5-1.5B-Instruct\n(baseline)": Path("scripts/inference/results/prime_val.json"),
    "GLM 5.3 Flash\n(teacher)": Path("scripts/inference/results/glm53_val.json"),
    "Qwen2.5-1.5B-Instruct\n+ SFT on teacher traces": Path("scripts/inference/results/sft_glm53_val.json"),
    "Qwen2.5-1.5B-Instruct\n+ SFT + GRPO": Path("scripts/inference/results/sft_glm53_grpo_val.json"),
    "Qwen2.5-1.5B-Instruct\n+ GRPO only (zero)": Path("scripts/inference/results/zero_val.json"),
}
BAR_COLORS = ["tab:gray", "tab:purple", "tab:blue", "tab:green", "tab:orange"]
Y_AXIS_MARGIN_BELOW_BASELINE = 5.0


def main() -> None:
    accuracies = read_available_accuracies()
    figure = build_bar_chart_figure(accuracies)
    figure.savefig(Path("plot/val_accuracy_bar_chart.png"), dpi=150, bbox_inches="tight")


def read_available_accuracies() -> dict[str, float]:
    return {
        model_name: json.loads(report_path.read_text())["accuracy"]
        for model_name, report_path in MODEL_REPORTS.items()
        if report_path.exists()
    }


def build_bar_chart_figure(accuracies: dict[str, float]) -> plt.Figure:
    baseline_accuracy = next(iter(accuracies.values()))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    bars = axis.bar(list(accuracies), list(accuracies.values()), color=BAR_COLORS[: len(accuracies)])
    axis.bar_label(bars, labels=[label_bar(accuracy, baseline_accuracy) for accuracy in accuracies.values()], padding=4)
    axis.axhline(baseline_accuracy, color="tab:gray", linestyle="--", linewidth=1)

    axis.set_ylabel("validation accuracy (%)")
    axis.set_ylim(baseline_accuracy - Y_AXIS_MARGIN_BELOW_BASELINE, 105)
    axis.set_title("GSM8K validation accuracy (1,319 questions)")
    axis.grid(axis="y", alpha=0.3)
    return figure


def label_bar(accuracy: float, baseline_accuracy: float) -> str:
    gain = accuracy - baseline_accuracy
    if gain == 0:
        return f"{accuracy:.2f}%"
    return f"{accuracy:.2f}%\n(+{gain:.1f} pts)"


if __name__ == "__main__":
    main()
