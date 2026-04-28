"""
File Name: script_tab_3.py
Author: Jiatong & Kaibo
Function: Evaluates salt-and-pepper denoising methods on merged validation and test subsets.
Reference: None.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from baseline import my_Huber_Markov, my_gaussian, my_median, my_wiener, my_bayesian_markov
from bf import bilateral_filter_numba as bf
from nlm import nlm_numba as nlm
from script_dinov3_denoiser import dinov3_denoise
from script_dncnn_eval import dncnn_denoise
from utils import get_result_rgb, imnoise_salt_pepper


METHOD_ORDER = [
    "Noisy",
    "Gaussian",
    "Median",
    "Wiener",
    "Bayesian Markov",
    "Huber Markov",
    "NLM",
    "BF",
    "DnCNN",
    "DINOv3-ViT",
]
CLASSICAL_METHODS = [
    ("Noisy", lambda x: x, {}),
    ("Gaussian", my_gaussian, {}),
    ("Median", my_median, {}),
    ("Wiener", my_wiener, {}),
    ("Bayesian Markov", my_bayesian_markov, {}),
    ("Huber Markov", my_Huber_Markov, {}),
    ("NLM", nlm, {"h": 0.15}),
    ("BF", bf, {"sigma_spatial": 1.5, "sigma_range": 0.10}),
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate denoisers under salt-and-pepper noise.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data",
        help="Root containing BSD100 and Urban100.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=project_root / "data" / "dncnn_split",
        help="Held-out split used for appendix evaluation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "tables" / "tab3",
        help="Where to write the salt-and-pepper appendix tables.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "artifacts" / "dncnn_color_sp10" / "checkpoints" / "best.pt",
        help="DnCNN checkpoint evaluated in the appendix tables.",
    )
    parser.add_argument(
        "--dinov3-checkpoint",
        type=Path,
        default=project_root / "artifacts" / "dinov3_vits16_sp10" / "checkpoints" / "best.pt",
        help="DINOv3-ViT checkpoint evaluated in the appendix tables.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for learned model inference: auto | cuda | cpu.",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=0.1,
        help="Salt-and-pepper corruption probability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for synthetic salt-and-pepper corruption.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--dinov3-only",
        action="store_true",
        help="Only evaluate DINOv3-ViT, then combine it with existing classical and DnCNN raw CSVs.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def summarize(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    metric_names = ["MSE", "PSNR", "SSIM", "Perceptual"]
    rows = []
    subset = df[df["Dataset"] == dataset_name]

    for method in METHOD_ORDER:
        method_df = subset[subset["Method"] == method]
        if method_df.empty:
            continue
        row = {
            "Method": method,
            "Split": method_df["Split"].iloc[0],
            "Count": int(method_df.shape[0]),
        }
        for metric_name in metric_names:
            row[f"{metric_name} (mean)"] = float(method_df[metric_name].mean())
            row[f"{metric_name} (std)"] = float(method_df[metric_name].std())
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_classical_dataset(
    dataset_name: str,
    paths: list[Path],
    amount: float,
    seed: int,
    split_name: str,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for path in paths:
        clean = load_image(path)
        noisy = imnoise_salt_pepper(clean, amount, rng)
        image_id = path.stem

        for method_name, handle, kwargs in CLASSICAL_METHODS:
            _, mse_val, psnr_val, ssim_val, perceptual_val = get_result_rgb(handle, noisy, clean, **kwargs)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Image ID": image_id,
                    "Split": split_name,
                    "Method": method_name,
                    "MSE": mse_val,
                    "PSNR": psnr_val,
                    "SSIM": ssim_val,
                    "Perceptual": perceptual_val,
                }
            )

    return rows


def evaluate_dncnn_dataset(
    dataset_name: str,
    paths: list[Path],
    amount: float,
    seed: int,
    checkpoint: Path,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for path in paths:
        clean = load_image(path)
        noisy = imnoise_salt_pepper(clean, amount, rng)
        _, mse_val, psnr_val, ssim_val, perceptual_val = get_result_rgb(
            dncnn_denoise,
            noisy,
            clean,
            checkpoint=checkpoint,
        )
        rows.append(
            {
                "Dataset": dataset_name,
                "Image ID": path.stem,
                "Split": "heldout",
                "Method": "DnCNN",
                "MSE": mse_val,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "Perceptual": perceptual_val,
            }
        )

    return rows


def evaluate_dinov3_dataset(
    dataset_name: str,
    paths: list[Path],
    amount: float,
    seed: int,
    checkpoint: Path,
    device: str,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for path in paths:
        clean = load_image(path)
        noisy = imnoise_salt_pepper(clean, amount, rng)
        _, mse_val, psnr_val, ssim_val, perceptual_val = get_result_rgb(
            dinov3_denoise,
            noisy,
            clean,
            checkpoint=checkpoint,
            device=device,
        )
        rows.append(
            {
                "Dataset": dataset_name,
                "Image ID": path.stem,
                "Split": "heldout",
                "Method": "DINOv3-ViT",
                "MSE": mse_val,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "Perceptual": perceptual_val,
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    classical_rows: list[dict[str, object]] = []
    dncnn_rows: list[dict[str, object]] = []
    dinov3_rows: list[dict[str, object]] = []

    if args.dinov3_only:
        classical_df = pd.read_csv(args.output_root / "tab3_raw_classical.csv")
        dncnn_df = pd.read_csv(args.output_root / "tab3_raw_dncnn.csv")
    else:
        classical_df = None
        dncnn_df = None

    for dataset_name in ["BSD100", "Urban100"]:
        paths = sorted((args.split_root / "val" / dataset_name).glob("*.png")) + sorted(
            (args.split_root / "test" / dataset_name).glob("*.png")
        )
        if args.limit_per_dataset is not None:
            paths = paths[: args.limit_per_dataset]

        if not args.dinov3_only:
            classical_rows.extend(
                evaluate_classical_dataset(
                    dataset_name=dataset_name,
                    paths=paths,
                    amount=args.amount,
                    seed=args.seed,
                    split_name="heldout",
                )
            )
            dncnn_rows.extend(
                evaluate_dncnn_dataset(
                    dataset_name=dataset_name,
                    paths=paths,
                    amount=args.amount,
                    seed=args.seed,
                    checkpoint=args.checkpoint,
                )
            )
        dinov3_rows.extend(
            evaluate_dinov3_dataset(
                dataset_name=dataset_name,
                paths=paths,
                amount=args.amount,
                seed=args.seed,
                checkpoint=args.dinov3_checkpoint,
                device=args.device,
            )
        )

    if classical_df is None:
        classical_df = pd.DataFrame(classical_rows)
        classical_df.to_csv(args.output_root / "tab3_raw_classical.csv", index=False)
    if dncnn_df is None:
        dncnn_df = pd.DataFrame(dncnn_rows)
        dncnn_df.to_csv(args.output_root / "tab3_raw_dncnn.csv", index=False)

    dinov3_df = pd.DataFrame(dinov3_rows)
    combined_df = pd.concat([classical_df, dncnn_df, dinov3_df], ignore_index=True)

    dinov3_df.to_csv(args.output_root / "tab3_raw_dinov3.csv", index=False)

    for dataset_name, filename in [("BSD100", "tab3_stats_bsd100.csv"), ("Urban100", "tab3_stats_urban100.csv")]:
        summary_df = summarize(combined_df, dataset_name)
        ordered_methods = [method for method in METHOD_ORDER if method in set(summary_df["Method"])]
        summary_df = summary_df.set_index("Method").loc[ordered_methods]
        summary_df = summary_df.reset_index()
        summary_df.to_csv(args.output_root / filename, index=False)
        print(summary_df)


if __name__ == "__main__":
    main()
