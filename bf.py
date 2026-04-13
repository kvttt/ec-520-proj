"""Bilateral filtering following Tomasi and Manduchi (ICCV 1998).

This module implements the bilateral filter as a normalized weighted average
with a Gaussian spatial kernel and a Gaussian range kernel:

    h(x) = 1 / k(x) * sum_xi f(xi) c(xi, x) s(f(xi), f(x))

where c measures spatial closeness and s measures photometric similarity.

For RGB images, the paper recommends computing range similarity in CIE-Lab
space. This implementation therefore converts RGB inputs to Lab, performs the
filter in Lab, and converts the result back to RGB.
"""

from __future__ import annotations

# import argparse
import math
# import time
# from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def gaussian_spatial_kernel(radius: int, sigma_spatial: float) -> np.ndarray:
    """Return the Gaussian spatial kernel c used by the bilateral filter."""
    _validate_positive("sigma_spatial", sigma_spatial)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist2 = xx * xx + yy * yy
    return np.exp(-0.5 * dist2 / (sigma_spatial * sigma_spatial))


def bilateral_filter(
    image: np.ndarray,
    sigma_spatial: float,
    sigma_range: float,
    radius: int | None = None,
    # use_lab: bool = True,
) -> np.ndarray:
    """Apply bilateral filtering to a grayscale or RGB image.

    Args:
        image: Input image as float array in [0, 1]. Shape (H, W) or (H, W, 3).
        sigma_spatial: Standard deviation of the Gaussian spatial kernel.
        sigma_range: Standard deviation of the Gaussian range kernel.
            For RGB with ``use_lab=True``, this is measured in Lab units.
        radius: Window radius. If None, use ceil(3 * sigma_spatial).
        # use_lab: For RGB input, compute range similarity in Lab space.

    Returns:
        Filtered image with the same shape as the input.
    """
    _validate_positive("sigma_spatial", sigma_spatial)
    _validate_positive("sigma_range", sigma_range)

    image = np.asarray(image, dtype=np.float64)
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
        work_image = image
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

    return filtered
