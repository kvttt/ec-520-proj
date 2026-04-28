"""
File Name: utils.py
Author: Kaibo & Jiatong
Function: Provides shared image loading, noise generation, metrics, plotting, and export utilities.
Reference: LPIPS, https://github.com/richzhang/PerceptualSimilarity; scikit-image metrics, https://scikit-image.org/docs/stable/api/skimage.metrics.html.
"""

import tempfile
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import torch

import warnings
warnings.filterwarnings("ignore")


_LPIPS_MODEL = None
_TORCHVISION_COMPAT_LIB = None


def barbara():
    return np.array(Image.open(_project_root() / "data" / "barbara.tif")).astype(np.float64) / 255.0


def bu2010():
    return np.array(Image.open(_project_root() / "data" / "bu2010.tif")).astype(np.float64) / 255.0


def imnoise(u, sigma, rng):
    return u + sigma * rng.standard_normal(u.shape)


def imnoise_salt_pepper(u, amount, rng, salt_vs_pepper=0.5):
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must lie in [0, 1].")
    if not 0.0 <= salt_vs_pepper <= 1.0:
        raise ValueError("salt_vs_pepper must lie in [0, 1].")

    noisy = np.array(u, dtype=np.float64, copy=True)
    mask = rng.random(noisy.shape[:2])
    pepper = mask < amount * (1.0 - salt_vs_pepper)
    salt = (mask >= amount * (1.0 - salt_vs_pepper)) & (mask < amount)

    if noisy.ndim == 2:
        noisy[pepper] = 0.0
        noisy[salt] = 1.0
    else:
        noisy[pepper, :] = 0.0
        noisy[salt, :] = 1.0
    return noisy


def mse(u_clean, u_hat):
    return mean_squared_error(u_clean, u_hat)


def psnr(u_clean, u_hat):
    return peak_signal_noise_ratio(u_clean, u_hat, data_range=1)


def ssim(u_clean, u_hat):
    return structural_similarity(u_clean, u_hat, data_range=1, channel_axis=-1)


def get_result_gray(handle, u_noisy, u_clean, **kwargs):
    u_hat = handle(u_noisy, **kwargs)
    mse_val = mse(u_clean, u_hat)
    psnr_val = psnr(u_clean, u_hat)
    ssim_val = ssim(u_clean, u_hat)
    perceptual_val = perceptual(u_clean, u_hat)
    return u_hat, mse_val, psnr_val, ssim_val, perceptual_val


def get_result_rgb(handle, u_noisy, u_clean, **kwargs):
    u_hat = handle(u_noisy, **kwargs)
    mse_val = mse(u_clean, u_hat)
    psnr_val = psnr(u_clean, u_hat)
    ssim_val = ssim(u_clean, u_hat)
    perceptual_val = perceptual(u_clean, u_hat)
    return u_hat, mse_val, psnr_val, ssim_val, perceptual_val


def get_result_lab(handle, u_noisy, u_clean, **kwargs):
    u_lab = rgb2lab(u_noisy)
    u_lab_hat = handle(u_lab, **kwargs)
    u_hat = lab2rgb(u_lab_hat)
    mse_val = mse(u_clean, u_hat)
    psnr_val = psnr(u_clean, u_hat)
    ssim_val = ssim(u_clean, u_hat)
    perceptual_val = perceptual(u_clean, u_hat)
    return u_hat, mse_val, psnr_val, ssim_val, perceptual_val


def _style_panel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _format_metrics(metrics):
    mse_val, psnr_val, ssim_val, perceptual_val = metrics
    return (
        f"MSE = {mse_val:.4f}\n"
        f"PSNR = {psnr_val:.2f} dB\n"
        f"SSIM = {ssim_val:.4f}\n"
        f"Perceptual = {perceptual_val:.4f}"
    )


def _gray_residual_map(u_hat, u_clean):
    return np.asarray(u_hat, dtype=np.float64) - np.asarray(u_clean, dtype=np.float64)


def _rgb_residual_map(u_hat, u_clean, residual_scale=0.2):
    residual = np.asarray(u_hat, dtype=np.float64) - np.asarray(u_clean, dtype=np.float64)
    return np.clip(residual / residual_scale + 0.5, 0.0, 1.0)


def _as_hwc3(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W) or (H, W, 3), got {image.shape}")
    return np.clip(image, 0.0, 1.0)


def _panel_metrics_array(metrics):
    if metrics is None:
        return np.full(4, np.nan, dtype=np.float64)
    return np.asarray(metrics, dtype=np.float64)


def _safe_panel_name(name):
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return "_".join(part for part in safe.split("_") if part)


def _save_subplot_npz(path, name, kind, image, metrics):
    metrics_array = _panel_metrics_array(metrics)
    np.savez_compressed(
        path,
        image=_as_hwc3(image),
        metrics=metrics_array,
        metric_names=np.asarray(["MSE", "PSNR", "SSIM", "Perceptual"]),
        mse=metrics_array[0],
        psnr=metrics_array[1],
        ssim=metrics_array[2],
        perceptual=metrics_array[3],
        title=np.asarray(name),
        kind=np.asarray(kind),
    )


def save_panel_npz_outputs(panels, u_clean, output_dir, image_mode, prefix, include_error_maps=True):
    """Save each visible figure subplot as an HxWx3 array plus metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (name, image, metrics) in enumerate(panels):
        safe_name = _safe_panel_name(name)
        base = f"{prefix}_{idx:02d}_{safe_name}"
        _save_subplot_npz(output_dir / f"{base}_image.npz", name, "image", image, metrics)

        if metrics is None or not include_error_maps:
            continue
        if image_mode == "gray":
            error = np.clip((_gray_residual_map(image, u_clean) + 0.1) / 0.2, 0.0, 1.0)
        else:
            error = _rgb_residual_map(image, u_clean)
        _save_subplot_npz(output_dir / f"{base}_error.npz", name, "error", error, metrics)


def _draw_result_pair(ax_img, ax_err, name, image, metrics, u_clean, image_mode, title_fontsize, metric_fontsize):
    _style_panel(ax_img)
    _style_panel(ax_err)

    if image_mode == "gray":
        ax_img.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax_img.imshow(np.clip(image, 0.0, 1.0))
    ax_img.set_title(name, fontsize=title_fontsize)

    if metrics is None:
        ax_err.axis("off")
        return

    if image_mode == "gray":
        ax_err.imshow(_gray_residual_map(image, u_clean), cmap="gray", vmin=-0.1, vmax=0.1)
    else:
        ax_err.imshow(_rgb_residual_map(image, u_clean))
    ax_err.set_title(_format_metrics(metrics), fontsize=metric_fontsize)


def _save_figure_with_max_bytes(fig, output_path, dpi, max_bytes, min_dpi=72, **savefig_kwargs):
    output_path = Path(output_path)

    if max_bytes is None:
        fig.savefig(output_path, dpi=dpi, **savefig_kwargs)
        return dpi, output_path.stat().st_size

    def save_trial(trial_dpi):
        with tempfile.NamedTemporaryFile(suffix=output_path.suffix, dir=output_path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            fig.savefig(tmp_path, dpi=trial_dpi, **savefig_kwargs)
            return tmp_path.stat().st_size, tmp_path
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    too_large_dpi = dpi
    too_large_path = None
    too_large_size = None
    acceptable = None

    trial_dpi = dpi
    while True:
        trial_size, trial_path = save_trial(trial_dpi)
        if trial_size <= max_bytes:
            acceptable = (trial_dpi, trial_size, trial_path)
            break
        too_large_dpi = trial_dpi
        too_large_size = trial_size
        if too_large_path is not None and too_large_path.exists():
            too_large_path.unlink()
        too_large_path = trial_path
        if trial_dpi <= min_dpi:
            acceptable = (trial_dpi, trial_size, trial_path)
            break
        next_dpi = max(min_dpi, int(trial_dpi * np.sqrt(max_bytes / trial_size) * 0.98))
        if next_dpi >= trial_dpi:
            next_dpi = trial_dpi - 1
        trial_dpi = max(min_dpi, next_dpi)

    best_dpi, best_size, best_path = acceptable
    low = best_dpi
    high = too_large_dpi - 1 if too_large_size is not None else best_dpi

    while low <= high:
        mid = (low + high) // 2
        trial_size, trial_path = save_trial(mid)
        if trial_size <= max_bytes:
            if best_path.exists():
                best_path.unlink()
            best_dpi, best_size, best_path = mid, trial_size, trial_path
            low = mid + 1
        else:
            trial_path.unlink()
            high = mid - 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.replace(output_path)
    if too_large_path is not None and too_large_path.exists():
        too_large_path.unlink()
    return best_dpi, best_size


def save_comparison_grid_with_error_maps(
    panels,
    u_clean,
    output_path,
    image_mode,
    figsize=(20, 21),
    dpi=300,
    max_bytes=None,
    min_dpi=72,
):
    if len(panels) != 10:
        raise ValueError("Expected exactly 10 panels arranged as five rows of two method blocks.")

    title_fontsize = 15
    metric_fontsize = 12
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(5, 4)

    for row in range(5):
        left_panel = panels[2 * row]
        right_panel = panels[2 * row + 1]
        left_img = fig.add_subplot(grid[row, 0])
        left_err = fig.add_subplot(grid[row, 1])
        right_img = fig.add_subplot(grid[row, 2])
        right_err = fig.add_subplot(grid[row, 3])
        _draw_result_pair(left_img, left_err, *left_panel, u_clean, image_mode, title_fontsize, metric_fontsize)
        _draw_result_pair(right_img, right_err, *right_panel, u_clean, image_mode, title_fontsize, metric_fontsize)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual_dpi, actual_size = _save_figure_with_max_bytes(
        fig,
        output_path,
        dpi=dpi,
        max_bytes=max_bytes,
        min_dpi=min_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)
    return {"dpi": actual_dpi, "bytes": actual_size}


def to_tensor(u):
    if u.ndim == 2:
        u = np.stack([u, u, u], axis=-1)
    return torch.tensor(u).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1


def _ensure_torchvision_nms_schema():
    global _TORCHVISION_COMPAT_LIB

    if _TORCHVISION_COMPAT_LIB is not None:
        return

    try:
        import torchvision  # noqa: F401

        _TORCHVISION_COMPAT_LIB = True
        return
    except Exception as exc:
        if "torchvision::nms" not in str(exc):
            return
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)

    try:
        from torch.library import Library

        _TORCHVISION_COMPAT_LIB = Library("torchvision", "DEF")
        _TORCHVISION_COMPAT_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


def perceptual(u_clean, u_hat):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        _ensure_torchvision_nms_schema()
        import lpips

        _LPIPS_MODEL = lpips.LPIPS(verbose=False)
    return _LPIPS_MODEL(to_tensor(u_clean), to_tensor(u_hat)).item()


def _project_root():
    return Path(__file__).resolve().parent.parent
