"""
File Name: script_dinov3_eval.py
Author: Jiatong
Function: Evaluates fine-tuned DINOv3-ViT denoiser on held-out BSD100 and Urban100 images.
Reference: DINOv3, https://github.com/facebookresearch/dinov3; timm, https://github.com/huggingface/pytorch-image-models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from script_dinov3_denoiser import load_dinov3_model
from utils import imnoise_salt_pepper, mse, perceptual, psnr, ssim


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 denoiser")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "artifacts" / "dinov3_vits16_sigma25" / "checkpoints" / "best.pt",
        help="Checkpoint written by script_dinov3_train.py.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=project_root / "data" / "dncnn_split",
        help="Root directory created by script_dncnn_dataset.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "tables" / "tab_dinov3_sigma25",
        help="Where to write raw and summary CSV files.",
    )
    parser.add_argument("--sigma", type=int, default=25, help="AWGN sigma on the 0-255 scale.")
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=("gaussian", "salt_pepper"),
        default="gaussian",
        help="Synthetic corruption model used on the evaluation split.",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=0.1,
        help="Salt-and-pepper corruption probability when --noise-type=salt_pepper.",
    )
    parser.add_argument(
        "--salt-vs-pepper",
        type=float,
        default=0.5,
        help="Salt probability among corrupted pixels when --noise-type=salt_pepper.",
    )
    parser.add_argument("--seed", type=int, default=520, help="Random seed for synthetic test noise.")
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto | cuda | cpu.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def add_noise(
    clean: np.ndarray,
    noise_type: str,
    sigma: int,
    amount: float,
    salt_vs_pepper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_type == "gaussian":
        return clean + (sigma / 255.0) * rng.standard_normal(clean.shape)
    return imnoise_salt_pepper(clean, amount=amount, rng=rng, salt_vs_pepper=salt_vs_pepper)


def to_tensor(image: np.ndarray, device) -> torch.Tensor:
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).float()
    return tensor.to(device)


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()[0]
    return np.transpose(array, (1, 2, 0))


def summarize_and_save(df: pd.DataFrame, dataset_name: str, output_root: Path) -> None:
    metric_names = ["MSE", "PSNR", "SSIM", "Perceptual"]
    summary = {"Method": []}
    for metric_name in metric_names:
        summary[f"{metric_name} (mean)"] = []
        summary[f"{metric_name} (std)"] = []

    for method in ["Noisy", "DINOv3-ViT"]:
        subset = df[(df["Dataset"] == dataset_name) & (df["Method"] == method)]
        summary["Method"].append(method)
        for metric_name in metric_names:
            summary[f"{metric_name} (mean)"].append(float(subset[metric_name].mean()))
            summary[f"{metric_name} (std)"].append(float(subset[metric_name].std()))

    pd.DataFrame(summary).to_csv(output_root / f"tab_stats_{dataset_name.lower()}.csv", index=False)


def main() -> None:
    args = parse_args()
    model, device = load_dinov3_model(checkpoint=args.checkpoint, device=args.device)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset_name in ["BSD100", "Urban100"]:
        rng = np.random.default_rng(args.seed)
        dataset_root = args.split_root / "test" / dataset_name
        paths = sorted(dataset_root.glob("*.png"))
        if args.limit_per_dataset is not None:
            paths = paths[: args.limit_per_dataset]

        for path in paths:
            clean = load_image(path)
            noisy = add_noise(
                clean=clean,
                noise_type=args.noise_type,
                sigma=args.sigma,
                amount=args.amount,
                salt_vs_pepper=args.salt_vs_pepper,
                rng=rng,
            )

            with torch.no_grad():
                denoised = from_tensor(model(to_tensor(noisy, device)))

            rows.append(
                {
                    "Dataset": dataset_name,
                    "Image ID": path.stem,
                    "Method": "Noisy",
                    "MSE": mse(clean, noisy),
                    "PSNR": psnr(clean, noisy),
                    "SSIM": ssim(clean, noisy),
                    "Perceptual": perceptual(clean, noisy),
                }
            )
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Image ID": path.stem,
                    "Method": "DINOv3-ViT",
                    "MSE": mse(clean, denoised),
                    "PSNR": psnr(clean, denoised),
                    "SSIM": ssim(clean, denoised),
                    "Perceptual": perceptual(clean, denoised),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_root / "tab_raw.csv", index=False)
    summarize_and_save(df, "BSD100", args.output_root)
    summarize_and_save(df, "Urban100", args.output_root)
    print(f"Wrote evaluation tables to: {args.output_root}")


if __name__ == "__main__":
    main()
