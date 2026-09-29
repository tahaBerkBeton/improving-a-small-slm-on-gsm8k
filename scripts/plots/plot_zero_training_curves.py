# This script plots the training curves of the GRPO-only run saved in models/Qwen2.5_zero/training_curves.json: the training
# reward and the sampled validation reward on the left axis and the greedy validation accuracy on the right axis, all
# against the training step. The accuracy curve starts from the untouched model's accuracy at step 0, and the plot
# marks the baseline and SFT+GRPO accuracies and the checkpoint with the best validation accuracy. It is saved to
# plot/zero_training_curves.png.

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_ACCURACY_PERCENT = 62.09
SFT_ACCURACY_PERCENT = 69.90
SFT_GRPO_ACCURACY_PERCENT = 71.87
MAX_REWARD = 4.0


def main() -> None:
    curves = json.loads(Path("models/Qwen2.5_zero/training_curves.json").read_text())
    figure = build_training_curves_figure(curves)
    figure.savefig(Path("plot/zero_training_curves.png"), dpi=150, bbox_inches="tight")


def build_training_curves_figure(curves: dict[str, list[dict[str, float]]]) -> plt.Figure:
    figure, reward_axis = plt.subplots(figsize=(9, 5))
    accuracy_axis = reward_axis.twinx()
    accuracy_points = [{"step": 0, "value": BASELINE_ACCURACY_PERCENT}, *curves["eval_accuracy"]]

    plot_curve(reward_axis, curves["train_reward"], label="train reward (sampled)", color="tab:blue", marker=None)
    plot_curve(reward_axis, curves["eval_reward"], label="validation reward (sampled)", color="tab:orange", marker="o")
    plot_curve(accuracy_axis, accuracy_points, label="validation accuracy", color="tab:green", marker="s")
    mark_reference_accuracies_and_best(accuracy_axis, accuracy_points)

    reward_axis.set_xlabel("training step")
    reward_axis.set_ylabel(f"reward (max {MAX_REWARD:.0f})")
    accuracy_axis.set_ylabel("validation accuracy (%)")
    accuracy_axis.set_ylim(BASELINE_ACCURACY_PERCENT - 4, 80)
    reward_axis.set_title("GRPO alone from the untouched Instruct model (zero)")
    reward_axis.grid(alpha=0.3)
    combine_legends(reward_axis, accuracy_axis)
    return figure


def plot_curve(axis: plt.Axes, points: list[dict[str, float]], label: str, color: str, marker: str | None) -> None:
    steps = [point["step"] for point in points]
    values = [point["value"] for point in points]
    axis.plot(steps, values, label=label, color=color, marker=marker)


def mark_reference_accuracies_and_best(accuracy_axis: plt.Axes, accuracy_points: list[dict[str, float]]) -> None:
    best_point = max(accuracy_points, key=lambda point: point["value"])
    accuracy_axis.axhline(BASELINE_ACCURACY_PERCENT, color="tab:red", linestyle="--", label="baseline accuracy")
    accuracy_axis.axhline(SFT_ACCURACY_PERCENT, color="tab:gray", linestyle=":", label="SFT accuracy")
    accuracy_axis.axhline(SFT_GRPO_ACCURACY_PERCENT, color="tab:green", linestyle=":", label="SFT + GRPO accuracy")
    accuracy_axis.annotate(
        f"best: {best_point['value']:.2f}% at step {best_point['step']:.0f}\n"
        f"(+{best_point['value'] - BASELINE_ACCURACY_PERCENT:.1f} pts vs baseline)",
        xy=(best_point["step"], best_point["value"]),
        xytext=(15, -40),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )


def combine_legends(reward_axis: plt.Axes, accuracy_axis: plt.Axes) -> None:
    reward_handles, reward_labels = reward_axis.get_legend_handles_labels()
    accuracy_handles, accuracy_labels = accuracy_axis.get_legend_handles_labels()
    reward_axis.legend(reward_handles + accuracy_handles, reward_labels + accuracy_labels, loc="lower right")


if __name__ == "__main__":
    main()
