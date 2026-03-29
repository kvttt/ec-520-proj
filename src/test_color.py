from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from bf_color import add_gaussian_noise, color_bilateral_filter, load_color_image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BSDS_TEST_DIR = PROJECT_ROOT / "BSDS300" / "images" / "test"
FIG_DIR = PROJECT_ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)


def default_color_image_path():
    candidates = sorted(BSDS_TEST_DIR.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No test images found in {BSDS_TEST_DIR}")
    return candidates[0]


def get_result(handle, u_noisy, u_clean, **kwargs):
    u_hat = handle(u_noisy, **kwargs)
    mse = mean_squared_error(u_clean, u_hat)
    psnr = peak_signal_noise_ratio(u_clean, u_hat, data_range=1)
    ssim = structural_similarity(u_clean, u_hat, data_range=1, channel_axis=-1)
    return u_hat, mse, psnr, ssim


def main():
    image_path = default_color_image_path()
    u_clean = load_color_image(image_path)
    u_noisy = add_gaussian_noise(u_clean, sigma=20.0, seed=0)

    u_noisy, mse_noisy, psnr_noisy, ssim_noisy = get_result(lambda x: x, u_noisy, u_clean)
    u_gaussian, mse_gaussian, psnr_gaussian, ssim_gaussian = get_result(
        gaussian,
        u_noisy,
        u_clean,
        sigma=1.0,
        channel_axis=-1,
        preserve_range=True,
    )
    u_bilateral, mse_bilateral, psnr_bilateral, ssim_bilateral = get_result(
        color_bilateral_filter,
        u_noisy,
        u_clean,
        sigma_spatial=3.0,
        sigma_range=12.0,
    )

    plt.style.use("default")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[0, 0].imshow(u_clean)
    axes[0, 0].set_title("Clean")
    axes[0, 1].imshow(u_noisy)
    axes[0, 1].set_title("Noisy")
    axes[0, 2].imshow(u_gaussian)
    axes[0, 2].set_title("Gaussian")
    axes[0, 3].imshow(u_bilateral)
    axes[0, 3].set_title("Color Bilateral")

    residual_scale = 0.2
    axes[1, 0].axis("off")
    axes[1, 1].imshow(np.clip((u_noisy - u_clean) / residual_scale + 0.5, 0.0, 1.0))
    axes[1, 1].set_title(
        f"MSE = {mse_noisy:.4f}\nPSNR = {psnr_noisy:.2f} dB\nSSIM = {ssim_noisy:.4f}"
    )
    axes[1, 2].imshow(np.clip((u_gaussian - u_clean) / residual_scale + 0.5, 0.0, 1.0))
    axes[1, 2].set_title(
        f"MSE = {mse_gaussian:.4f}\nPSNR = {psnr_gaussian:.2f} dB\nSSIM = {ssim_gaussian:.4f}"
    )
    axes[1, 3].imshow(np.clip((u_bilateral - u_clean) / residual_scale + 0.5, 0.0, 1.0))
    axes[1, 3].set_title(
        f"MSE = {mse_bilateral:.4f}\nPSNR = {psnr_bilateral:.2f} dB\nSSIM = {ssim_bilateral:.4f}"
    )

    fig.suptitle(f"Color Denoising on {image_path.name}")
    fig.savefig(FIG_DIR / "fig_color1.png", dpi=300)
    plt.close(fig)

    sigma_spatial_values = [1.5, 2.0, 3.0, 4.0, 5.0]
    mse_lst, psnr_lst, ssim_lst = [], [], []
    for sigma_spatial in sigma_spatial_values:
        _, mse, psnr, ssim = get_result(
            color_bilateral_filter,
            u_noisy,
            u_clean,
            sigma_spatial=sigma_spatial,
            sigma_range=12.0,
        )
        mse_lst.append(mse)
        psnr_lst.append(psnr)
        ssim_lst.append(ssim)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].plot(sigma_spatial_values, mse_lst, marker="o", color="C0")
    axes[0].set_xlabel("sigma_spatial")
    axes[0].set_xticks(sigma_spatial_values)
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[1].plot(sigma_spatial_values, psnr_lst, marker="o", color="C1")
    axes[1].set_xlabel("sigma_spatial")
    axes[1].set_xticks(sigma_spatial_values)
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[2].plot(sigma_spatial_values, ssim_lst, marker="o", color="C2")
    axes[2].set_xlabel("sigma_spatial")
    axes[2].set_xticks(sigma_spatial_values)
    axes[2].set_ylabel("SSIM")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Performance of Color Bilateral Filtering at different sigma_spatial")
    fig.savefig(FIG_DIR / "fig_color2.png", dpi=300)
    plt.close(fig)

    sigma_range_values = [8.0, 10.0, 12.0, 15.0, 20.0]
    mse_lst, psnr_lst, ssim_lst = [], [], []
    for sigma_range in sigma_range_values:
        _, mse, psnr, ssim = get_result(
            color_bilateral_filter,
            u_noisy,
            u_clean,
            sigma_spatial=3.0,
            sigma_range=sigma_range,
        )
        mse_lst.append(mse)
        psnr_lst.append(psnr)
        ssim_lst.append(ssim)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].plot(sigma_range_values, mse_lst, marker="o", color="C0")
    axes[0].set_xlabel("sigma_range")
    axes[0].set_xticks(sigma_range_values)
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[1].plot(sigma_range_values, psnr_lst, marker="o", color="C1")
    axes[1].set_xlabel("sigma_range")
    axes[1].set_xticks(sigma_range_values)
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[2].plot(sigma_range_values, ssim_lst, marker="o", color="C2")
    axes[2].set_xlabel("sigma_range")
    axes[2].set_xticks(sigma_range_values)
    axes[2].set_ylabel("SSIM")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Performance of Color Bilateral Filtering at different sigma_range")
    fig.savefig(FIG_DIR / "fig_color3.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
