import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    barbara, bu2010, 
    imnoise, 
    get_result_gray, get_result_rgb, get_result_lab,
)
from baseline import (
    my_gaussian, my_median, my_wiener, my_bayesian_markov,
)
from nlm import nlm_numba as nlm
from bf import bilateral_filter_numba as bf


rng = np.random.default_rng(0)
sigma = 0.1


u_clean = bu2010()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_gray(lambda x: x, u_noisy, u_clean)
u_gaussian, mse_gaussian, psnr_gaussian, ssim_gaussian, perceptual_gaussian = get_result_rgb(my_gaussian, u_noisy, u_clean)
u_median, mse_median, psnr_median, ssim_median, perceptual_median = get_result_rgb(my_median, u_noisy, u_clean)
u_wiener, mse_wiener, psnr_wiener, ssim_wiener, perceptual_wiener = get_result_rgb(my_wiener, u_noisy, u_clean)
u_bayesian_markov, mse_bayesian_markov, psnr_bayesian_markov, ssim_bayesian_markov, perceptual_bayesian_markov = get_result_rgb(my_bayesian_markov, u_noisy, u_clean)
u_nlm, mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm = get_result_rgb(nlm, u_noisy, u_clean, h=0.18)
u_bf, mse_bf, psnr_bf, ssim_bf, perceptual_bf = get_result_rgb(bf, u_noisy, u_clean, sigma_spatial=1.5, sigma_range=0.25)


fig, axes = plt.subplots(2, 4, figsize=(16, 6), layout='constrained')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

axes[0, 0].imshow(u_clean, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title("Clean")
axes[0, 1].imshow(u_gaussian, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f"Gaussian\nMSE: {mse_gaussian:.4f}, PSNR: {psnr_gaussian:.2f} dB,\nSSIM: {ssim_gaussian:.4f}, Perceptual: {perceptual_gaussian:.4f}")
axes[0, 2].imshow(u_median, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f"Median\nMSE: {mse_median:.4f}, PSNR: {psnr_median:.2f} dB,\nSSIM: {ssim_median:.4f}, Perceptual: {perceptual_median:.4f}")
axes[0, 3].imshow(u_wiener, cmap='gray', vmin=0, vmax=1)
axes[0, 3].set_title(f"Wiener\nMSE: {mse_wiener:.4f}, PSNR: {psnr_wiener:.2f} dB,\nSSIM: {ssim_wiener:.4f}, Perceptual: {perceptual_wiener:.4f}")

axes[1, 0].imshow(u_noisy, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title(f"Noisy\nMSE: {mse_noisy:.4f}, PSNR: {psnr_noisy:.2f} dB,\nSSIM: {ssim_noisy:.4f}, Perceptual: {perceptual_noisy:.4f}")
axes[1, 1].imshow(u_bayesian_markov, cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title(f"Bayesian Markov\nMSE: {mse_bayesian_markov:.4f}, PSNR: {psnr_bayesian_markov:.2f} dB,\nSSIM: {ssim_bayesian_markov:.4f}, Perceptual: {perceptual_bayesian_markov:.4f}")
axes[1, 2].imshow(u_nlm, cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title(f"NLM\nMSE: {mse_nlm:.4f}, PSNR: {psnr_nlm:.2f} dB,\nSSIM: {ssim_nlm:.4f}, Perceptual: {perceptual_nlm:.4f}")
axes[1, 3].imshow(u_bf, cmap='gray', vmin=0, vmax=1)
axes[1, 3].set_title(f"BF\nMSE: {mse_bf:.4f}, PSNR: {psnr_bf:.2f} dB,\nSSIM: {ssim_bf:.4f}, Perceptual: {perceptual_bf:.4f}")

fig.savefig("figures/fig6/fig6.png", dpi=300)
plt.show()
