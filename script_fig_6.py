"""
File Name: script_fig_6.py
Author: Kaibo
Function: Generates the color AWGN qualitative comparison figure and subplot NPZ files.
Reference: None.
"""

import numpy as np
from pathlib import Path

from utils import (
    bu2010,
    imnoise,
    get_result_rgb,
    save_panel_npz_outputs,
    save_comparison_grid_with_error_maps,
)
from script_dncnn_eval import dncnn_denoise
from baseline import (
    my_Huber_Markov, my_gaussian, my_median, my_wiener, my_bayesian_markov,
)
from nlm import nlm_numba as nlm
from bf import bilateral_filter_numba as bf


rng = np.random.default_rng(0)
sigma = 0.1


u_clean = bu2010()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_rgb(lambda x: x, u_noisy, u_clean)
u_gaussian, mse_gaussian, psnr_gaussian, ssim_gaussian, perceptual_gaussian = get_result_rgb(my_gaussian, u_noisy, u_clean)
u_median, mse_median, psnr_median, ssim_median, perceptual_median = get_result_rgb(my_median, u_noisy, u_clean)
u_wiener, mse_wiener, psnr_wiener, ssim_wiener, perceptual_wiener = get_result_rgb(my_wiener, u_noisy, u_clean)
u_bayesian_markov, mse_bayesian_markov, psnr_bayesian_markov, ssim_bayesian_markov, perceptual_bayesian_markov = get_result_rgb(my_bayesian_markov, u_noisy, u_clean)
u_huber_markov, mse_huber_markov, psnr_huber_markov, ssim_huber_markov, perceptual_huber_markov = get_result_rgb(my_Huber_Markov, u_noisy, u_clean)
u_nlm, mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm = get_result_rgb(nlm, u_noisy, u_clean, h=0.18)
u_bf, mse_bf, psnr_bf, ssim_bf, perceptual_bf = get_result_rgb(bf, u_noisy, u_clean, sigma_spatial=1.5, sigma_range=0.25)
u_dncnn, mse_dncnn, psnr_dncnn, ssim_dncnn, perceptual_dncnn = get_result_rgb(dncnn_denoise, u_noisy, u_clean)

panels = [
    ("Clean", u_clean, None),
    ("Noisy", u_noisy, (mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy)),
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
    output_dir=Path("figures/fig6/npz"),
    image_mode="rgb",
    prefix="fig6",
)

result = save_comparison_grid_with_error_maps(
    panels=panels,
    u_clean=u_clean,
    output_path=Path("figures/fig6/fig6.png"),
    image_mode="rgb",
    max_bytes=50 * 1024 * 1024,
)

print(f"Saved figure to: figures/fig6/fig6.png (dpi={result['dpi']}, bytes={result['bytes']})")
