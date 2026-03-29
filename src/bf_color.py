import math

import numpy as np
from PIL import Image

from bilateral_filter import lab_to_rgb, rgb_to_lab


def _validate_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _gaussian_spatial_kernel(radius, sigma_spatial):
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist2 = xx * xx + yy * yy
    return np.exp(-0.5 * dist2 / (sigma_spatial * sigma_spatial))


def load_color_image(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def save_color_image(path, image):
    array = np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def add_gaussian_noise(image, sigma=20.0, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma / 255.0, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def color_bilateral_filter(image, sigma_spatial=3.0, sigma_range=12.0, radius=None, use_lab=True):
    """Apply bilateral filtering to an RGB image in [0, 1].

    Args:
        image: RGB image with shape (H, W, 3).
        sigma_spatial: Spatial Gaussian standard deviation in pixels.
        sigma_range: Range Gaussian standard deviation. If ``use_lab=True``,
            this value is interpreted in CIE-Lab units.
        radius: Window radius. Defaults to ceil(3 * sigma_spatial).
        use_lab: If True, compute color distance and averaging in Lab space.

    Returns:
        Filtered RGB image in [0, 1].
    """
    _validate_positive("sigma_spatial", sigma_spatial)
    _validate_positive("sigma_range", sigma_range)

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("color_bilateral_filter expects an RGB image with shape (H, W, 3)")

    radius = int(math.ceil(3 * sigma_spatial)) if radius is None else int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    spatial = _gaussian_spatial_kernel(radius, sigma_spatial)
    work = rgb_to_lab(img) if use_lab else img
    padded = np.pad(work, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    height, width = img.shape[:2]
    numerator = np.zeros_like(work)
    denominator = np.zeros((height, width), dtype=np.float64)
    inv_two_sigma_range2 = -0.5 / (sigma_range * sigma_range)

    for dy in range(2 * radius + 1):
        y0 = dy
        y1 = dy + height
        for dx in range(2 * radius + 1):
            shifted = padded[y0:y1, dx:dx + width, :]
            diff2 = np.sum((shifted - work) ** 2, axis=-1)
            weights = spatial[dy, dx] * np.exp(diff2 * inv_two_sigma_range2)
            numerator += weights[..., None] * shifted
            denominator += weights

    filtered = numerator / denominator[..., None]
    return lab_to_rgb(filtered) if use_lab else np.clip(filtered, 0.0, 1.0)
