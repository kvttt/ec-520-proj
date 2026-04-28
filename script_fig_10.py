"""
File Name: script_fig_10.py
Author: Jiatong
Function: Generates the grayscale salt-and-pepper qualitative comparison figure and subplot NPZ files.
Reference: None.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baseline import my_Huber_Markov, my_gaussian, my_median, my_wiener, my_bayesian_markov
from bf import bilateral_filter_numba as bf
from nlm import nlm_numba as nlm
from script_dncnn_eval import dncnn_denoise
from utils import (
    barbara,
    get_result_gray,
    imnoise_salt_pepper,
    save_panel_npz_outputs,
    save_comparison_grid_with_error_maps,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Plot the grayscale salt-and-pepper appendix figure.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "artifacts" / "dncnn_color_sp10" / "checkpoints" / "best.pt",
        help="DnCNN checkpoint used for the appendix figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "figures" / "fig10" / "fig10.png",
        help="Where to save the figure.",
    )
    parser.add_argument("--amount", type=float, default=0.1, help="Salt-and-pepper corruption probability.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for the synthetic corruption.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    nlm_h = 0.10
    bf_sigma_spatial = 2.0
    bf_sigma_range = 0.10

    u_clean = barbara()
    u_noisy = imnoise_salt_pepper(u_clean, args.amount, rng)
    u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_gray(lambda x: x, u_noisy, u_clean)
    u_gaussian, mse_gaussian, psnr_gaussian, ssim_gaussian, perceptual_gaussian = get_result_gray(my_gaussian, u_noisy, u_clean)
    u_median, mse_median, psnr_median, ssim_median, perceptual_median = get_result_gray(my_median, u_noisy, u_clean)
    u_wiener, mse_wiener, psnr_wiener, ssim_wiener, perceptual_wiener = get_result_gray(my_wiener, u_noisy, u_clean)
    u_bayesian_markov, mse_bayesian_markov, psnr_bayesian_markov, ssim_bayesian_markov, perceptual_bayesian_markov = get_result_gray(my_bayesian_markov, u_noisy, u_clean)
    u_huber_markov, mse_huber_markov, psnr_huber_markov, ssim_huber_markov, perceptual_huber_markov = get_result_gray(my_Huber_Markov, u_noisy, u_clean)
    u_nlm, mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm = get_result_gray(nlm, u_noisy, u_clean, h=nlm_h)
    u_bf, mse_bf, psnr_bf, ssim_bf, perceptual_bf = get_result_gray(
        bf,
        u_noisy,
        u_clean,
        sigma_spatial=bf_sigma_spatial,
        sigma_range=bf_sigma_range,
    )
    u_dncnn, mse_dncnn, psnr_dncnn, ssim_dncnn, perceptual_dncnn = get_result_gray(
        dncnn_denoise,
        u_noisy,
        u_clean,
        checkpoint=args.checkpoint,
    )

    panels = [
        ("Clean", u_clean, None),
        ("Salt-and-pepper noisy", u_noisy, (mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy)),
        ("Gaussian", u_gaussian, (mse_gaussian, psnr_gaussian, ssim_gaussian, perceptual_gaussian)),
        ("Median", u_median, (mse_median, psnr_median, ssim_median, perceptual_median)),
        ("Wiener", u_wiener, (mse_wiener, psnr_wiener, ssim_wiener, perceptual_wiener)),
        ("DnCNN", u_dncnn, (mse_dncnn, psnr_dncnn, ssim_dncnn, perceptual_dncnn)),
        ("Bayesian-Markov", u_bayesian_markov, (mse_bayesian_markov, psnr_bayesian_markov, ssim_bayesian_markov, perceptual_bayesian_markov)),
        ("Huber-Markov", u_huber_markov, (mse_huber_markov, psnr_huber_markov, ssim_huber_markov, perceptual_huber_markov)),
        ("Non-local means", u_nlm, (mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm)),
        ("Bilateral filtering", u_bf, (mse_bf, psnr_bf, ssim_bf, perceptual_bf)),
    ]

    save_panel_npz_outputs(
        panels=panels,
        u_clean=u_clean,
        output_dir=args.output.parent / "npz",
        image_mode="gray",
        prefix=args.output.stem,
    )

    save_comparison_grid_with_error_maps(
        panels=panels,
        u_clean=u_clean,
        output_path=args.output,
        image_mode="gray",
    )
    print(f"Saved figure to: {args.output}")
    print(f"Saved subplot NPZ files to: {args.output.parent / 'npz'}")


if __name__ == "__main__":
    main()
