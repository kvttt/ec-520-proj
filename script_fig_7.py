"""
File Name: script_fig_7.py
Author: Kaibo
Function: Generates the BSD100 AWGN sample comparison figure and subplot NPZ files.
Reference: None.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

from script_dncnn_eval import dncnn_denoise
from utils import get_result_rgb, imnoise, save_panel_npz_outputs
from nlm import nlm_numba as nlm
from bf import bilateral_filter_numba as bf


rng = np.random.default_rng(0)
sigma = 0.1


fn = "data/BSD100/img_003_SRF_4_HR.png"
img = np.array(Image.open(fn)).astype(np.float64) / 255.0
u_clean = img
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_rgb(lambda x: x, u_noisy, u_clean)
u_nlm, mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm = get_result_rgb(nlm, u_noisy, u_clean, h=0.18)
u_bf, mse_bf, psnr_bf, ssim_bf, perceptual_bf = get_result_rgb(bf, u_noisy, u_clean, sigma_spatial=1.5, sigma_range=0.25)
u_dncnn, mse_dncnn, psnr_dncnn, ssim_dncnn, perceptual_dncnn = get_result_rgb(dncnn_denoise, u_noisy, u_clean)

panels = [
    ("Clean", u_clean, None),
    ("Noisy", u_noisy, (mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy)),
    ("Non-local means", u_nlm, (mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm)),
    ("Bilateral filtering", u_bf, (mse_bf, psnr_bf, ssim_bf, perceptual_bf)),
    ("DnCNN", u_dncnn, (mse_dncnn, psnr_dncnn, ssim_dncnn, perceptual_dncnn)),
]

save_panel_npz_outputs(
    panels=panels,
    u_clean=u_clean,
    output_dir=Path("figures/fig7/npz"),
    image_mode="rgb",
    prefix="fig7",
    include_error_maps=False,
)

fig, axes = plt.subplots(1, 5, figsize=(16, 6), layout="constrained")

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

axes[0].imshow(np.clip(u_clean, 0.0, 1.0))
axes[0].set_title("Clean")
axes[1].imshow(np.clip(u_noisy, 0.0, 1.0))
axes[1].set_title(f"Noisy\nMSE: {mse_noisy:.4f}, PSNR: {psnr_noisy:.2f} dB\nSSIM: {ssim_noisy:.4f}, Perceptual: {perceptual_noisy:.4f}")
axes[2].imshow(np.clip(u_nlm, 0.0, 1.0))
axes[2].set_title(f"Non-local means\nMSE: {mse_nlm:.4f}, PSNR: {psnr_nlm:.2f} dB\nSSIM: {ssim_nlm:.4f}, Perceptual: {perceptual_nlm:.4f}")
axes[3].imshow(np.clip(u_bf, 0.0, 1.0))
axes[3].set_title(f"Bilateral filtering\nMSE: {mse_bf:.4f}, PSNR: {psnr_bf:.2f} dB\nSSIM: {ssim_bf:.4f}, Perceptual: {perceptual_bf:.4f}")
axes[4].imshow(np.clip(u_dncnn, 0.0, 1.0))
axes[4].set_title(f"DnCNN\nMSE: {mse_dncnn:.4f}, PSNR: {psnr_dncnn:.2f} dB\nSSIM: {ssim_dncnn:.4f}, Perceptual: {perceptual_dncnn:.4f}")

fig.savefig("figures/fig7/fig7.png", dpi=600, bbox_inches="tight")
plt.show()
