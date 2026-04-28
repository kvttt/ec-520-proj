"""
File Name: script_fig_9.py
Author: Jiatong
Function: Plots DnCNN and DINOv3-ViT training loss and validation curves.
Reference: None.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_CONFIGS = {
    "dncnn": {
        "label": "DnCNN",
        "history": "artifacts/dncnn_color_sigma25/history.csv",
        "output": "figures/fig9/fig9.png",
    },
    "dinov3": {
        "label": "DINOv3-ViT",
        "history": "artifacts/dinov3_vits16_sigma25/history.csv",
        "output": "figures/fig_dinov3_sigma25/fig_dinov3_sigma25.png",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learned denoiser training curves.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="dncnn",
        help="Which learned denoiser history to plot.",
    )
    parser.add_argument("--history", type=Path, default=None, help="Optional training history CSV override.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output figure path override.")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    project_root = Path(__file__).resolve().parent.parent
    config = MODEL_CONFIGS[args.model]
    history = args.history or project_root / config["history"]
    output = args.output or project_root / config["output"]
    return history, output, config["label"]


def plot_training_curves(history: Path, output: Path, label: str) -> None:
    df = pd.read_csv(history)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    panels = [
        ("Training Loss", "train_loss", "MSE Loss", "o", None),
        ("Validation PSNR", "val_psnr", "PSNR (dB)", "o", "tab:blue"),
        ("Validation L1", "val_l1", "L1", "s", "tab:orange"),
    ]

    for ax, (title, column, ylabel, marker, color) in zip(axes, panels, strict=True):
        ax.plot(df["step"], df[column], marker=marker, color=color)
        ax.set_title(f"{label} {title}")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {output}")


def main(argv: list[str] | None = None) -> None:
    history, output, label = resolve_paths(parse_args(argv))
    plot_training_curves(history, output, label)


if __name__ == "__main__":
    main()
