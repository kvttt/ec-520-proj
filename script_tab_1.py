"""
File Name: script_tab_1.py
Author: Kaibo & Jiatong
Function: Evaluates AWGN denoising methods on merged validation and test subsets.
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
from script_dncnn_eval import dncnn_denoise
from utils import get_result_rgb, imnoise


METHOD_ORDER = ["Noisy", "Gaussian", "Median", "Wiener", "Bayesian Markov", "Huber Markov", "NLM", "BF", "DnCNN"]
CLASSICAL_METHODS = [
    ("Noisy", lambda x: x, {}),
    ("Gaussian", my_gaussian, {}),
    ("Median", my_median, {}),
    ("Wiener", my_wiener, {}),
    ("Bayesian Markov", my_bayesian_markov, {}),
    ("Huber Markov", my_Huber_Markov, {}),
    ("NLM", nlm, {"h": 0.18}),
    ("BF", bf, {"sigma_spatial": 1.5, "sigma_range": 0.25}),
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate all AWGN denoisers on the merged held-out split.")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=project_root / "data" / "dncnn_split",
        help="Root containing the fixed train/val/test split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "artifacts" / "dncnn_color_sigma25" / "checkpoints" / "best.pt",
        help="DnCNN checkpoint evaluated in Table 1/2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "tables" / "tab1",
        help="Where to write the held-out Table 1/2 CSV files.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Gaussian noise standard deviation on the normalized [0, 1] scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for synthetic AWGN generation.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
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


def evaluate_dataset(
    dataset_name: str,
    paths: list[Path],
    sigma: float,
    seed: int,
    checkpoint: Path,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for path in paths:
        clean = load_image(path)
        noisy = imnoise(clean, sigma, rng)
        image_id = path.stem

        for method_name, handle, kwargs in CLASSICAL_METHODS:
            _, mse_val, psnr_val, ssim_val, perceptual_val = get_result_rgb(handle, noisy, clean, **kwargs)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Image ID": image_id,
                    "Split": "heldout",
                    "Method": method_name,
                    "MSE": mse_val,
                    "PSNR": psnr_val,
                    "SSIM": ssim_val,
                    "Perceptual": perceptual_val,
                }
            )

        _, mse_val, psnr_val, ssim_val, perceptual_val = get_result_rgb(
            dncnn_denoise,
            noisy,
            clean,
            checkpoint=checkpoint,
        )
        rows.append(
            {
                "Dataset": dataset_name,
                "Image ID": image_id,
                "Split": "heldout",
                "Method": "DnCNN",
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

    rows: list[dict[str, object]] = []

    for dataset_name in ["BSD100", "Urban100"]:
        paths = sorted((args.split_root / "val" / dataset_name).glob("*.png")) + sorted(
            (args.split_root / "test" / dataset_name).glob("*.png")
        )
        if args.limit_per_dataset is not None:
            paths = paths[: args.limit_per_dataset]
        rows.extend(
            evaluate_dataset(
                dataset_name=dataset_name,
                paths=paths,
                sigma=args.sigma,
                seed=args.seed,
                checkpoint=args.checkpoint,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_root / "tab1_raw.csv", index=False)

    for dataset_name, filename in [("BSD100", "tab1_stats_bsd100.csv"), ("Urban100", "tab1_stats_urban100.csv")]:
        summary_df = summarize(df, dataset_name)
        ordered_methods = [method for method in METHOD_ORDER if method in set(summary_df["Method"])]
        summary_df = summary_df.set_index("Method").loc[ordered_methods].reset_index()
        summary_df.to_csv(args.output_root / filename, index=False)
        print(summary_df)


if __name__ == "__main__":
    main()
