import math

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - optional dependency
    njit = None
    prange = range

NUMBA_AVAILABLE = njit is not None


def _validate_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _gaussian_spatial_kernel(radius, sigma_spatial):
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist2 = xx * xx + yy * yy
    return np.exp(-0.5 * dist2 / (sigma_spatial * sigma_spatial))


def bilateral_filter_numpy(image, sigma_spatial=2.0, sigma_range=0.1, radius=None):
    """Bilateral filter for 2D grayscale images in [0, 1]."""
    _validate_positive("sigma_spatial", sigma_spatial)
    _validate_positive("sigma_range", sigma_range)

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("bilateral_filter expects a 2D grayscale image")

    radius = int(math.ceil(3 * sigma_spatial)) if radius is None else int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    spatial = _gaussian_spatial_kernel(radius, sigma_spatial)
    padded = np.pad(img, radius, mode="reflect")
    height, width = img.shape
    numerator = np.zeros_like(img)
    denominator = np.zeros_like(img)
    inv_two_sigma_range2 = -0.5 / (sigma_range * sigma_range)

    for dy in range(2 * radius + 1):
        y0 = dy
        y1 = dy + height
        for dx in range(2 * radius + 1):
            shifted = padded[y0:y1, dx:dx + width]
            diff2 = (shifted - img) ** 2
            weights = spatial[dy, dx] * np.exp(diff2 * inv_two_sigma_range2)
            numerator += weights * shifted
            denominator += weights

    return np.clip(numerator / denominator, 0.0, 1.0)


if njit is not None:

    @njit(parallel=True, cache=True)
    def _bilateral_numba(padded, spatial, radius, inv_two_sigma_range2):
        height = padded.shape[0] - 2 * radius
        width = padded.shape[1] - 2 * radius
        out = np.empty((height, width), dtype=np.float64)

        for row in prange(height):
            for col in range(width):
                center = padded[row + radius, col + radius]
                numerator = 0.0
                denominator = 0.0
                for dy in range(2 * radius + 1):
                    y = row + dy
                    for dx in range(2 * radius + 1):
                        value = padded[y, col + dx]
                        diff = value - center
                        weight = spatial[dy, dx] * np.exp(diff * diff * inv_two_sigma_range2)
                        numerator += weight * value
                        denominator += weight
                out[row, col] = numerator / denominator

        return out


def bilateral_filter_numba(image, sigma_spatial=2.0, sigma_range=0.1, radius=None):
    """Numba-accelerated bilateral filter if numba is installed."""
    if not NUMBA_AVAILABLE:
        return bilateral_filter_numpy(
            image,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            radius=radius,
        )

    _validate_positive("sigma_spatial", sigma_spatial)
    _validate_positive("sigma_range", sigma_range)

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("bilateral_filter expects a 2D grayscale image")

    radius = int(math.ceil(3 * sigma_spatial)) if radius is None else int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    spatial = _gaussian_spatial_kernel(radius, sigma_spatial)
    padded = np.pad(img, radius, mode="reflect")
    out = _bilateral_numba(
        padded,
        spatial,
        radius,
        -0.5 / (sigma_range * sigma_range),
    )
    return np.clip(out, 0.0, 1.0)


def bilateral_filter(image, sigma_spatial=2.0, sigma_range=0.1, radius=None, implementation="auto"):
    """Dispatch to the available bilateral filter implementation.

    Args:
        image: 2D grayscale image normalized to [0, 1].
        sigma_spatial: Spatial Gaussian standard deviation.
        sigma_range: Range Gaussian standard deviation.
        radius: Window radius. Defaults to ceil(3 * sigma_spatial).
        implementation: One of ``auto``, ``numpy``, or ``numba``.
    """
    if implementation == "numpy":
        return bilateral_filter_numpy(
            image,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            radius=radius,
        )

    if implementation == "numba":
        if not NUMBA_AVAILABLE:
            raise ImportError("numba is not installed, so implementation='numba' is unavailable")
        return bilateral_filter_numba(
            image,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            radius=radius,
        )

    if implementation != "auto":
        raise ValueError("implementation must be 'auto', 'numpy', or 'numba'")

    if not NUMBA_AVAILABLE:
        return bilateral_filter_numpy(
            image,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            radius=radius,
        )

    return bilateral_filter_numba(
        image,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        radius=radius,
    )
