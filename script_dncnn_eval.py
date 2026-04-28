"""
File Name: script_dncnn_eval.py
Author: Jiatong
Function: Loads and evaluates trained DnCNN checkpoints on held-out denoising data.
Reference: DnCNN, https://doi.org/10.1109/TIP.2017.2662206; KAIR, https://github.com/cszn/KAIR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from utils import imnoise_salt_pepper, mse, perceptual, psnr, ssim


_DNCNN_MODELS = {}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate a trained DnCNN checkpoint on held-out project data.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "artifacts" / "dncnn_color_sigma25" / "checkpoints" / "best.pt",
        help="Checkpoint written by script_dncnn_train.py.",
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
        default=project_root / "tables" / "tab2",
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
        help="Salt probability among corrupted pixels",
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


def add_kair_to_path(project_root: Path) -> None:
    import sys

    kair_root = project_root / "external" / "KAIR"
    if str(kair_root) not in sys.path:
        sys.path.insert(0, str(kair_root))


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).float()
    return tensor.to(device)


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()[0]
    return np.transpose(array, (1, 2, 0))


def load_dncnn_model(checkpoint: Path | str | None = None, device: str = "auto"):
    root = project_root()
    checkpoint_path = Path(checkpoint) if checkpoint is not None else (
        root / "artifacts" / "dncnn_color_sigma25" / "checkpoints" / "best.pt"
    )
    device_obj = resolve_device(device)
    cache_key = (str(checkpoint_path.resolve()), str(device_obj))

    if cache_key not in _DNCNN_MODELS:
        add_kair_to_path(root)
        from models.network_dncnn import DnCNN

        checkpoint_data = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=17, act_mode="BR").to(device_obj)
        model.load_state_dict(checkpoint_data["model_state"], strict=True)
        model.eval()
        _DNCNN_MODELS[cache_key] = model

    return _DNCNN_MODELS[cache_key], device_obj


def dncnn_denoise(u_noisy: np.ndarray, checkpoint: Path | str | None = None, device: str = "auto") -> np.ndarray:
    model, device_obj = load_dncnn_model(checkpoint=checkpoint, device=device)
    is_gray = u_noisy.ndim == 2
    model_input = np.stack([u_noisy, u_noisy, u_noisy], axis=-1) if is_gray else u_noisy

    with torch.no_grad():
        denoised = model(to_tensor(model_input, device_obj)).clamp(0.0, 1.0).cpu().numpy()[0]

    denoised = np.transpose(denoised, (1, 2, 0))
    if is_gray:
        return np.mean(denoised, axis=-1)
    return denoised


def summarize_and_save(df: pd.DataFrame, dataset_name: str, output_root: Path) -> None:
    metric_names = ["MSE", "PSNR", "SSIM", "Perceptual"]
    summary = {"Method": []}
    for metric_name in metric_names:
        summary[f"{metric_name} (mean)"] = []
        summary[f"{metric_name} (std)"] = []

    for method in ["Noisy", "DnCNN"]:
        subset = df[(df["Dataset"] == dataset_name) & (df["Method"] == method)]
        summary["Method"].append(method)
        for metric_name in metric_names:
            summary[f"{metric_name} (mean)"].append(float(subset[metric_name].mean()))
            summary[f"{metric_name} (std)"].append(float(subset[metric_name].std()))

    pd.DataFrame(summary).to_csv(output_root / f"tab2_stats_{dataset_name.lower()}.csv", index=False)


def main() -> None:
    args = parse_args()
    model, device = load_dncnn_model(checkpoint=args.checkpoint, device=args.device)

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
                    "Method": "DnCNN",
                    "MSE": mse(clean, denoised),
                    "PSNR": psnr(clean, denoised),
                    "SSIM": ssim(clean, denoised),
                    "Perceptual": perceptual(clean, denoised),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_root / "tab2_raw.csv", index=False)
    summarize_and_save(df, "BSD100", args.output_root)
    summarize_and_save(df, "Urban100", args.output_root)
    print(f"Wrote evaluation tables to: {args.output_root}")


if __name__ == "__main__":
    main()
