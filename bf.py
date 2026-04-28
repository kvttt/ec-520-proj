"""
File Name: bf.py
Author: Jiatong
Function: Implements bilateral filtering for grayscale, RGB, and CIE-Lab image denoising.
Reference: Tomasi and Manduchi (1998), https://doi.org/10.1109/ICCV.1998.710815.
"""

from __future__ import annotations

import math

from numba import njit, prange
import numpy as np


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def gaussian_spatial_kernel(radius: int, sigma_spatial: float) -> np.ndarray:
    _validate_positive("sigma_spatial", sigma_spatial)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist2 = xx * xx + yy * yy
    return np.exp(-0.5 * dist2 / (sigma_spatial * sigma_spatial))


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(np.clip(rgb, 0.0, None), 1.0 / 2.4) - 0.055,
    )


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_to_lab expects an array of shape (H, W, 3)")

    rgb_linear = _srgb_to_linear(np.clip(rgb, 0.0, 1.0)).astype(np.float64)

    xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = rgb_linear @ xyz_matrix.T

    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz_scaled = xyz / white

    delta = 6.0 / 29.0
    delta3 = delta ** 3
    linear_term = xyz_scaled / (3.0 * delta * delta) + 4.0 / 29.0
    f_xyz = np.where(xyz_scaled > delta3, np.cbrt(xyz_scaled), linear_term)

    l = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack([l, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("lab_to_rgb expects an array of shape (H, W, 3)")

    lab = lab.astype(np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    delta = 6.0 / 29.0
    f_stack = np.stack([fx, fy, fz], axis=-1)
    xyz_scaled = np.where(
        f_stack > delta,
        f_stack ** 3,
        3.0 * delta * delta * (f_stack - 4.0 / 29.0),
    )

    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz = xyz_scaled * white

    rgb_matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    rgb_linear = xyz @ rgb_matrix.T
    rgb = _linear_to_srgb(rgb_linear)
    return np.clip(rgb, 0.0, 1.0)


def _should_clip_output(image: np.ndarray, use_lab: bool) -> bool:
    if use_lab or image.ndim == 2:
        return True
    return bool(np.min(image) >= 0.0 and np.max(image) <= 1.0)


def bilateral_filter(
    image: np.ndarray,
    sigma_spatial: float,
    sigma_range: float,
    radius: int | None = None,
    use_lab: bool = False,
) -> np.ndarray:
    """
    Args:
        image: Input image as float array in [0, 1]. Shape (H, W) or (H, W, 3).
        sigma_spatial: Standard deviation of the Gaussian spatial kernel.
        sigma_range: Standard deviation of the Gaussian range kernel.
        radius: Window radius.
        use_lab: For RGB input, compute range similarity in Lab space.

    Returns:
        Filtered image with the same shape as the input.
    """
    _validate_positive("sigma_spatial", sigma_spatial)
    _validate_positive("sigma_range", sigma_range)

    image = np.asarray(image, dtype=np.float64)
    clip_output = _should_clip_output(image, use_lab)
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, 3)")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("color image must have shape (H, W, 3)")

    radius = int(math.ceil(3 * sigma_spatial)) if radius is None else int(radius)
    spatial_weights = gaussian_spatial_kernel(radius, sigma_spatial)
    inv_two_sigma_range2 = -0.5 / (sigma_range * sigma_range)

    if image.ndim == 2:
        work_image = image
        pad_width = ((radius, radius), (radius, radius))
    else:
        work_image = rgb_to_lab(image) if use_lab else image
        pad_width = ((radius, radius), (radius, radius), (0, 0))

    padded = np.pad(work_image, pad_width, mode="reflect")
    filtered = np.empty_like(work_image)
    height, width = work_image.shape[:2]

    if work_image.ndim == 2:
        for row in range(height):
            row_start = row
            row_end = row + 2 * radius + 1
            for col in range(width):
                col_start = col
                col_end = col + 2 * radius + 1
                patch = padded[row_start:row_end, col_start:col_end]
                center = padded[row + radius, col + radius]
                range_dist2 = (patch - center) ** 2
                range_weights = np.exp(range_dist2 * inv_two_sigma_range2)
                weights = spatial_weights * range_weights
                filtered[row, col] = np.sum(weights * patch) / np.sum(weights)
        return np.clip(filtered, 0.0, 1.0)

    for row in range(height):
        row_start = row
        row_end = row + 2 * radius + 1
        for col in range(width):
            col_start = col
            col_end = col + 2 * radius + 1
            patch = padded[row_start:row_end, col_start:col_end, :]
            center = padded[row + radius, col + radius, :]
            range_dist2 = np.sum((patch - center) ** 2, axis=-1)
            range_weights = np.exp(range_dist2 * inv_two_sigma_range2)
            weights = spatial_weights * range_weights
            filtered[row, col, :] = (
                np.sum(weights[..., None] * patch, axis=(0, 1)) / np.sum(weights)
            )

    if use_lab:
        return lab_to_rgb(filtered)
    if clip_output:
        return np.clip(filtered, 0.0, 1.0)
    return filtered


@njit(parallel=True, cache=True)
def _bilateral_filter_numba_core(
    padded: np.ndarray,
    spatial_weights: np.ndarray,
    radius: int,
    inv_two_sigma_range2: float,
) -> np.ndarray:
    p_height, p_width, n_channels = padded.shape
    height, width = p_height - 2 * radius, p_width - 2 * radius
    window = 2 * radius + 1
    filtered = np.empty((height, width, n_channels), dtype=np.float64)

    for row in prange(height):
        for col in range(width):
            center_row, center_col = row + radius, col + radius
            norm = 0.0
            acc = np.zeros(n_channels, dtype=np.float64)
            for dy in range(window):
                py = row + dy
                for dx in range(window):
                    px = col + dx
                    range_dist2 = 0.0
                    for ch in range(n_channels):
                        d = padded[py, px, ch] - padded[center_row, center_col, ch]
                        range_dist2 += d * d
                    w = spatial_weights[dy, dx] * math.exp(
                        range_dist2 * inv_two_sigma_range2
                    )
                    norm += w
                    for ch in range(n_channels):
                        acc[ch] += w * padded[py, px, ch]
            for ch in range(n_channels):
                filtered[row, col, ch] = acc[ch] / norm

    return filtered


def bilateral_filter_numba(
    image: np.ndarray,
    sigma_spatial: float,
    sigma_range: float,
    radius: int | None = None,
    use_lab: bool = False,
) -> np.ndarray:
    _validate_positive("sigma_spatial", sigma_spatial)
    _validate_positive("sigma_range", sigma_range)

    image = np.asarray(image, dtype=np.float64)
    clip_output = _should_clip_output(image, use_lab)
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, 3)")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("color image must have shape (H, W, 3)")

    radius = int(math.ceil(3 * sigma_spatial)) if radius is None else int(radius)
    spatial_weights = gaussian_spatial_kernel(radius, sigma_spatial)
    inv_two_sigma_range2 = -0.5 / (sigma_range * sigma_range)

    squeeze_output = False
    work_image = image
    if image.ndim == 2:
        work_image = image[:, :, np.newaxis]
        squeeze_output = True
    elif use_lab:
        work_image = rgb_to_lab(image)

    padded = np.pad(
        work_image,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="reflect",
    )
    filtered = _bilateral_filter_numba_core(
        padded,
        spatial_weights,
        radius,
        inv_two_sigma_range2,
    )

    if squeeze_output:
        return np.clip(filtered[:, :, 0], 0.0, 1.0)
    if use_lab:
        return lab_to_rgb(filtered)
    if clip_output:
        return np.clip(filtered, 0.0, 1.0)
    return filtered
