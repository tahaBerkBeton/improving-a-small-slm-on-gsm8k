# This script plots the SFT training curves saved in models/Qwen1_5_SFT_GLM/loss_history.json: the training and
# validation loss on the left axis and the validation accuracy on the right axis, all against the training step.
# The plot marks the baseline accuracy and the checkpoint with the best validation accuracy, and is saved to
# plot/sft_training_curves.png.

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_ACCURACY_PERCENT = 62.09


def main() -> None:
    curves = json.loads(Path("models/Qwen1_5_SFT_GLM/loss_history.json").read_text())
    figure = build_training_curves_figure(curves)
    figure.savefig(Path("plot/sft_training_curves.png"), dpi=150, bbox_inches="tight")


def build_training_curves_figure(curves: dict[str, list[dict[str, float]]]) -> plt.Figure:
    figure, loss_axis = plt.subplots(figsize=(9, 5))
    accuracy_axis = loss_axis.twinx()

    plot_curve(loss_axis, curves["train_loss"], label="train loss", color="tab:blue", marker=None)
    plot_curve(loss_axis, curves["eval_loss"], label="validation loss", color="tab:orange", marker="o")
    plot_curve(accuracy_axis, curves["eval_accuracy"], label="validation accuracy", color="tab:green", marker="s")
    mark_baseline_and_best_accuracy(accuracy_axis, curves["eval_accuracy"])

    loss_axis.set_xlabel("training step")
    loss_axis.set_ylabel("loss (completion tokens)")
    accuracy_axis.set_ylabel("validation accuracy (%)")
    accuracy_axis.set_ylim(BASELINE_ACCURACY_PERCENT - 8, 74)
    loss_axis.set_title("SFT of Qwen2.5-1.5B-Instruct on GLM 5.3 teacher traces")
    loss_axis.grid(alpha=0.3)
    combine_legends(loss_axis, accuracy_axis)
    return figure


def plot_curve(axis: plt.Axes, points: list[dict[str, float]], label: str, color: str, marker: str | None) -> None:
    steps = [point["step"] for point in points]
    values = [point["value"] for point in points]
    axis.plot(steps, values, label=label, color=color, marker=marker)


def mark_baseline_and_best_accuracy(accuracy_axis: plt.Axes, accuracy_points: list[dict[str, float]]) -> None:
    best_point = max(accuracy_points, key=lambda point: point["value"])
    accuracy_axis.axhline(BASELINE_ACCURACY_PERCENT, color="tab:red", linestyle="--", label="baseline accuracy")
    accuracy_axis.annotate(
        f"best: {best_point['value']:.2f}% at step {best_point['step']:.0f}\n"
        f"(+{best_point['value'] - BASELINE_ACCURACY_PERCENT:.1f} pts vs baseline)",
        xy=(best_point["step"], best_point["value"]),
        xytext=(15, -40),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )


def combine_legends(loss_axis: plt.Axes, accuracy_axis: plt.Axes) -> None:
    loss_handles, loss_labels = loss_axis.get_legend_handles_labels()
    accuracy_handles, accuracy_labels = accuracy_axis.get_legend_handles_labels()
    loss_axis.legend(loss_handles + accuracy_handles, loss_labels + accuracy_labels, loc="lower right")


if __name__ == "__main__":
    main()
