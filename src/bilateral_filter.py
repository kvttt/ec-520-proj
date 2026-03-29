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

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
    """Convert an RGB image in [0, 1] to CIE-Lab with D65 white point."""
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
    """Convert a CIE-Lab image to RGB in [0, 1] with D65 white point."""
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("lab_to_rgb expects an array of shape (H, W, 3)")

    lab = lab.astype(np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    delta = 6.0 / 29.0
    xyz_scaled = np.empty_like(lab)
    f_stack = np.stack([fx, fy, fz], axis=-1)
    xyz_scaled[:] = np.where(
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


def bilateral_filter(
    image: np.ndarray,
    sigma_spatial: float,
    sigma_range: float,
    radius: int | None = None,
    use_lab: bool = True,
) -> np.ndarray:
    """Apply bilateral filtering to a grayscale or RGB image.

    Args:
        image: Input image as float array in [0, 1]. Shape (H, W) or (H, W, 3).
        sigma_spatial: Standard deviation of the Gaussian spatial kernel.
        sigma_range: Standard deviation of the Gaussian range kernel.
            For RGB with ``use_lab=True``, this is measured in Lab units.
        radius: Window radius. If None, use ceil(3 * sigma_spatial).
        use_lab: For RGB input, compute range similarity in Lab space.

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

    return lab_to_rgb(filtered) if use_lab else np.clip(filtered, 0.0, 1.0)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as float RGB in [0, 1]."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save a float image in [0, 1] to disk."""
    array = np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def add_gaussian_noise(image: np.ndarray, noise_std_255: float, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise to an image. The std is specified on a 0-255 scale."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std_255 / 255.0, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def create_demo_figure(
    clean: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a side-by-side visualization for quick qualitative inspection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, image, name in zip(
        axes,
        [clean, noisy, filtered],
        ["Original", "Noisy", "Bilateral Filtered"],
        strict=True,
    ):
        ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def default_demo_image(project_root: Path) -> Path:
    candidates = sorted((project_root / "BSDS300" / "images" / "test").glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError("No BSDS300 test images were found.")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bilateral filtering demo.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path. Defaults to the first BSDS300 test image.",
    )
    parser.add_argument(
        "--sigma-spatial",
        type=float,
        default=3.0,
        help="Spatial Gaussian sigma.",
    )
    parser.add_argument(
        "--sigma-range",
        type=float,
        default=12.0,
        help="Range Gaussian sigma. For RGB + Lab, this is in Lab units.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Neighborhood radius. Defaults to ceil(3 * sigma_spatial).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=20.0,
        help="Gaussian noise std on the 0-255 scale used for the demo.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Resize factor for the demo image to keep runtime reasonable.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("bilateral_filtered.png"),
        help="Where to save the filtered image.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("bilateral_demo.png"),
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic noise.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert the input to grayscale before filtering.",
    )
    parser.add_argument(
        "--no-lab",
        action="store_true",
        help="For RGB images, filter in RGB instead of Lab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    image_path = args.image if args.image is not None else default_demo_image(project_root)

    clean_rgb = load_image(image_path)
    if args.downsample > 1:
        height, width = clean_rgb.shape[:2]
        resized = Image.fromarray((clean_rgb * 255.0).astype(np.uint8)).resize(
            (width // args.downsample, height // args.downsample),
            Image.Resampling.BILINEAR,
        )
        clean_rgb = np.asarray(resized, dtype=np.float64) / 255.0

    noisy_rgb = add_gaussian_noise(clean_rgb, noise_std_255=args.noise_std, seed=args.seed)

    if args.grayscale:
        clean = np.mean(clean_rgb, axis=-1)
        noisy = np.mean(noisy_rgb, axis=-1)
        sigma_range = args.sigma_range / 255.0 if args.sigma_range > 1.0 else args.sigma_range
        use_lab = False
    else:
        clean = clean_rgb
        noisy = noisy_rgb
        sigma_range = args.sigma_range
        use_lab = not args.no_lab

    start = time.time()
    filtered = bilateral_filter(
        noisy,
        sigma_spatial=args.sigma_spatial,
        sigma_range=sigma_range,
        radius=args.radius,
        use_lab=use_lab,
    )
    elapsed = time.time() - start

    if args.grayscale:
        filtered_to_save = np.stack([filtered, filtered, filtered], axis=-1)
        clean_for_fig = np.stack([clean, clean, clean], axis=-1)
        noisy_for_fig = np.stack([noisy, noisy, noisy], axis=-1)
    else:
        filtered_to_save = filtered
        clean_for_fig = clean
        noisy_for_fig = noisy

    save_image(args.output_image, filtered_to_save)
    create_demo_figure(
        clean_for_fig,
        noisy_for_fig,
        filtered_to_save,
        args.output_figure,
        title=(
            f"Bilateral filter on {image_path.name} | "
            f"sigma_spatial={args.sigma_spatial}, sigma_range={sigma_range:.4g}, "
            f"elapsed={elapsed:.2f}s"
        ),
    )

    print(f"Input image: {image_path}")
    print(f"Filtered image saved to: {args.output_image}")
    print(f"Comparison figure saved to: {args.output_figure}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
