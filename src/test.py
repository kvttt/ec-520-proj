from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import wiener
from skimage.filters import gaussian, median
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from nlm import nlm_numba
from bf import bilateral_filter

SCRIPT_DIR = Path(__file__).resolve().parent
BARBARA_PATH = SCRIPT_DIR / "barbara.tif"
FIG_DIR = SCRIPT_DIR.parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


def get_result(handle, u_noisy, u_clean, **kwargs):
    u_hat = handle(u_noisy, **kwargs)
    mse = mean_squared_error(u_clean, u_hat)
    psnr = peak_signal_noise_ratio(u_clean, u_hat, data_range=1)
    ssim = structural_similarity(u_clean, u_hat, data_range=1)
    return u_hat, mse, psnr, ssim


rng = np.random.default_rng(0)
sigma = 0.1
u_clean = np.array(Image.open(BARBARA_PATH).convert("L")).astype(np.float64) / 255.0
u_noisy = u_clean + sigma * rng.standard_normal(u_clean.shape)
u_noisy = np.clip(u_noisy, 0, 1)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy = get_result(lambda x: x, u_noisy, u_clean)
u_gaussian, mse_gaussian, psnr_gaussian, ssim_gaussian = get_result(gaussian, u_noisy, u_clean)
u_median, mse_median, psnr_median, ssim_median = get_result(median, u_noisy, u_clean)
u_wiener, mse_wiener, psnr_wiener, ssim_wiener = get_result(wiener, u_noisy, u_clean)
u_nlm, mse_nlm, psnr_nlm, ssim_nlm = get_result(nlm_numba, u_noisy, u_clean)
u_bilateral, mse_bilateral, psnr_bilateral, ssim_bilateral = get_result(bilateral_filter, u_noisy, u_clean)

plt.style.use('default')
fig, axes = plt.subplots(4, 4, figsize=(12, 12), layout='constrained')
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
axes[0, 0].imshow(u_clean, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title("Clean")
axes[0, 1].imshow(u_noisy, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title("Noisy")
axes[1, 1].imshow(u_noisy - u_clean, cmap='gray', vmin=-0.1, vmax=0.1)
axes[1, 1].set_title(f"MSE = {mse_noisy:.4f}\nPSNR = {psnr_noisy:.2f} dB\nSSIM = {ssim_noisy:.4f}")
axes[0, 2].imshow(u_gaussian, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title("Gaussian")
axes[1, 2].imshow(u_gaussian - u_clean, cmap='gray', vmin=-0.1, vmax=0.1)
axes[1, 2].set_title(f"MSE = {mse_gaussian:.4f}\nPSNR = {psnr_gaussian:.2f} dB\nSSIM = {ssim_gaussian:.4f}")
axes[0, 3].imshow(u_median, cmap='gray', vmin=0, vmax=1)
axes[0, 3].set_title("Median")
axes[1, 3].imshow(u_median - u_clean, cmap='gray', vmin=-0.1, vmax=0.1)
axes[1, 3].set_title(f"MSE = {mse_median:.4f}\nPSNR = {psnr_median:.2f} dB\nSSIM = {ssim_median:.4f}")
axes[2, 1].imshow(u_wiener, cmap='gray', vmin=0, vmax=1)
axes[2, 1].set_title("Wiener")
axes[3, 1].imshow(u_wiener - u_clean, cmap='gray', vmin=-0.1, vmax=0.1)
axes[3, 1].set_title(f"MSE = {mse_wiener:.4f}\nPSNR = {psnr_wiener:.2f} dB\nSSIM = {ssim_wiener:.4f}")
axes[2, 2].imshow(u_nlm, cmap='gray', vmin=0, vmax=1)
axes[2, 2].set_title("NLM")
axes[3, 2].imshow(u_nlm - u_clean, cmap='gray', vmin=-0.1, vmax=0.1)
axes[3, 2].set_title(f"MSE = {mse_nlm:.4f}\nPSNR = {psnr_nlm:.2f} dB\nSSIM = {ssim_nlm:.4f}")
axes[2, 3].imshow(u_bilateral, cmap='gray', vmin=0, vmax=1)
axes[2, 3].set_title("Bilateral")
axes[3, 3].imshow(u_bilateral - u_clean, cmap='gray', vmin=-0.1, vmax=0.1)
axes[3, 3].set_title(f"MSE = {mse_bilateral:.4f}\nPSNR = {psnr_bilateral:.2f} dB\nSSIM = {ssim_bilateral:.4f}")
fig.savefig(FIG_DIR / "fig1.png", dpi=300)
plt.show()


# NLM 
# fix patch_distance = 11, h = 0.1, and vary patch_size
mse_nlm_lst, psnr_nlm_lst, ssim_nlm_lst = [], [], []
for patch_size in [3, 5, 7, 9, 11]:
    u_nlm, mse_nlm, psnr_nlm, ssim_nlm = get_result(nlm_numba, u_noisy, u_clean, patch_size=patch_size)
    mse_nlm_lst.append(mse_nlm)
    psnr_nlm_lst.append(psnr_nlm)
    ssim_nlm_lst.append(ssim_nlm)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
axes[0].plot([3, 5, 7, 9, 11], mse_nlm_lst, marker='o', color='C0')
axes[0].set_xlabel("Patch size")
axes[0].set_xticks([3, 5, 7, 9, 11])
axes[0].set_ylabel("MSE")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].plot([3, 5, 7, 9, 11], psnr_nlm_lst, marker='o', color='C1')
axes[1].set_xlabel("Patch size")
axes[1].set_xticks([3, 5, 7, 9, 11])
axes[1].set_ylabel("PSNR (dB)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[2].plot([3, 5, 7, 9, 11], ssim_nlm_lst, marker='o', color='C2')
axes[2].set_xlabel("Patch size")
axes[2].set_xticks([3, 5, 7, 9, 11])
axes[2].set_ylabel("SSIM")
axes[2].grid(True, linestyle='--', alpha=0.5)
fig.suptitle("Performance of NLM at different patch sizes")
fig.savefig(FIG_DIR / "fig2.png", dpi=300)
plt.show()

# fix patch_size = 7, h = 0.1, and vary patch_distance
mse_nlm_lst, psnr_nlm_lst, ssim_nlm_lst = [], [], []
for patch_distance in [7, 9, 11, 13, 15]:
    u_nlm, mse_nlm, psnr_nlm, ssim_nlm = get_result(nlm_numba, u_noisy, u_clean, patch_distance=patch_distance)
    mse_nlm_lst.append(mse_nlm)
    psnr_nlm_lst.append(psnr_nlm)
    ssim_nlm_lst.append(ssim_nlm)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
axes[0].plot([7, 9, 11, 13, 15], mse_nlm_lst, marker='o', color='C0')
axes[0].set_xlabel("Patch distance")
axes[0].set_xticks([7, 9, 11, 13, 15])
axes[0].set_ylabel("MSE")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].plot([7, 9, 11, 13, 15], psnr_nlm_lst, marker='o', color='C1')
axes[1].set_xlabel("Patch distance")
axes[1].set_xticks([7, 9, 11, 13, 15])
axes[1].set_ylabel("PSNR (dB)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[2].plot([7, 9, 11, 13, 15], ssim_nlm_lst, marker='o', color='C2')
axes[2].set_xlabel("Patch distance")
axes[2].set_xticks([7, 9, 11, 13, 15])
axes[2].set_ylabel("SSIM")
axes[2].grid(True, linestyle='--', alpha=0.5)
fig.suptitle("Performance of NLM at different patch distances")
fig.savefig(FIG_DIR / "fig3.png", dpi=300)
plt.show()

# fix patch_size = 7, patch_distance = 11, and vary h
mse_nlm_lst, psnr_nlm_lst, ssim_nlm_lst = [], [], []
for h in [0.05, 0.1, 0.15, 0.2, 0.25]:
    u_nlm, mse_nlm, psnr_nlm, ssim_nlm = get_result(nlm_numba, u_noisy, u_clean, h=h)
    mse_nlm_lst.append(mse_nlm)
    psnr_nlm_lst.append(psnr_nlm)
    ssim_nlm_lst.append(ssim_nlm)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
axes[0].plot([0.05, 0.1, 0.15, 0.2, 0.25], mse_nlm_lst, marker='o', color='C0')
axes[0].set_xlabel("h")
axes[0].set_xticks([0.05, 0.1, 0.15, 0.2, 0.25])
axes[0].set_ylabel("MSE")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].plot([0.05, 0.1, 0.15, 0.2, 0.25], psnr_nlm_lst, marker='o', color='C1')
axes[1].set_xlabel("h")
axes[1].set_xticks([0.05, 0.1, 0.15, 0.2, 0.25])
axes[1].set_ylabel("PSNR (dB)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[2].plot([0.05, 0.1, 0.15, 0.2, 0.25], ssim_nlm_lst, marker='o', color='C2')
axes[2].set_xlabel("h")
axes[2].set_xticks([0.05, 0.1, 0.15, 0.2, 0.25])
axes[2].set_ylabel("SSIM")
axes[2].grid(True, linestyle='--', alpha=0.5)
fig.suptitle("Performance of NLM at different h")
fig.savefig(FIG_DIR / "fig4.png", dpi=300)
plt.show()


# Bilateral filtering
# fix sigma_range = 0.1, and vary sigma_spatial
mse_bilateral_lst, psnr_bilateral_lst, ssim_bilateral_lst = [], [], []
sigma_spatial_values = [1.0, 1.5, 2.0, 2.5, 3.0]
for sigma_spatial in sigma_spatial_values:
    u_bilateral, mse_bilateral, psnr_bilateral, ssim_bilateral = get_result(
        bilateral_filter,
        u_noisy,
        u_clean,
        sigma_spatial=sigma_spatial,
        sigma_range=0.1,
    )
    mse_bilateral_lst.append(mse_bilateral)
    psnr_bilateral_lst.append(psnr_bilateral)
    ssim_bilateral_lst.append(ssim_bilateral)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
axes[0].plot(sigma_spatial_values, mse_bilateral_lst, marker='o', color='C0')
axes[0].set_xlabel("sigma_spatial")
axes[0].set_xticks(sigma_spatial_values)
axes[0].set_ylabel("MSE")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].plot(sigma_spatial_values, psnr_bilateral_lst, marker='o', color='C1')
axes[1].set_xlabel("sigma_spatial")
axes[1].set_xticks(sigma_spatial_values)
axes[1].set_ylabel("PSNR (dB)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[2].plot(sigma_spatial_values, ssim_bilateral_lst, marker='o', color='C2')
axes[2].set_xlabel("sigma_spatial")
axes[2].set_xticks(sigma_spatial_values)
axes[2].set_ylabel("SSIM")
axes[2].grid(True, linestyle='--', alpha=0.5)
fig.suptitle("Performance of Bilateral Filtering at different sigma_spatial")
fig.savefig(FIG_DIR / "fig5.png", dpi=300)
plt.show()

# fix sigma_spatial = 2.0, and vary sigma_range
mse_bilateral_lst, psnr_bilateral_lst, ssim_bilateral_lst = [], [], []
sigma_range_values = [0.05, 0.075, 0.1, 0.125, 0.15]
for sigma_range in sigma_range_values:
    u_bilateral, mse_bilateral, psnr_bilateral, ssim_bilateral = get_result(
        bilateral_filter,
        u_noisy,
        u_clean,
        sigma_spatial=2.0,
        sigma_range=sigma_range,
    )
    mse_bilateral_lst.append(mse_bilateral)
    psnr_bilateral_lst.append(psnr_bilateral)
    ssim_bilateral_lst.append(ssim_bilateral)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
axes[0].plot(sigma_range_values, mse_bilateral_lst, marker='o', color='C0')
axes[0].set_xlabel("sigma_range")
axes[0].set_xticks(sigma_range_values)
axes[0].set_ylabel("MSE")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].plot(sigma_range_values, psnr_bilateral_lst, marker='o', color='C1')
axes[1].set_xlabel("sigma_range")
axes[1].set_xticks(sigma_range_values)
axes[1].set_ylabel("PSNR (dB)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[2].plot(sigma_range_values, ssim_bilateral_lst, marker='o', color='C2')
axes[2].set_xlabel("sigma_range")
axes[2].set_xticks(sigma_range_values)
axes[2].set_ylabel("SSIM")
axes[2].grid(True, linestyle='--', alpha=0.5)
fig.suptitle("Performance of Bilateral Filtering at different sigma_range")
fig.savefig(FIG_DIR / "fig6.png", dpi=300)
plt.show()

# fix sigma_spatial = 2.0, sigma_range = 0.1, and vary radius
mse_bilateral_lst, psnr_bilateral_lst, ssim_bilateral_lst = [], [], []
radius_values = [2, 4, 6, 8, 10]
for radius in radius_values:
    u_bilateral, mse_bilateral, psnr_bilateral, ssim_bilateral = get_result(
        bilateral_filter,
        u_noisy,
        u_clean,
        sigma_spatial=2.0,
        sigma_range=0.1,
        radius=radius,
    )
    mse_bilateral_lst.append(mse_bilateral)
    psnr_bilateral_lst.append(psnr_bilateral)
    ssim_bilateral_lst.append(ssim_bilateral)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
axes[0].plot(radius_values, mse_bilateral_lst, marker='o', color='C0')
axes[0].set_xlabel("radius")
axes[0].set_xticks(radius_values)
axes[0].set_ylabel("MSE")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].plot(radius_values, psnr_bilateral_lst, marker='o', color='C1')
axes[1].set_xlabel("radius")
axes[1].set_xticks(radius_values)
axes[1].set_ylabel("PSNR (dB)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[2].plot(radius_values, ssim_bilateral_lst, marker='o', color='C2')
axes[2].set_xlabel("radius")
axes[2].set_xticks(radius_values)
axes[2].set_ylabel("SSIM")
axes[2].grid(True, linestyle='--', alpha=0.5)
fig.suptitle("Performance of Bilateral Filtering at different radius")
fig.savefig(FIG_DIR / "fig7.png", dpi=300)
plt.show()
