import time
from pathlib import Path

import numpy as np
from PIL import Image

from bf import NUMBA_AVAILABLE, bilateral_filter, bilateral_filter_numba, bilateral_filter_numpy
from bilateral_filter import bilateral_filter as bilateral_filter_reference

try:  # pragma: no cover - optional dependency
    from skimage.restoration import denoise_bilateral as bilateral_skimage
except ImportError:  # pragma: no cover - optional dependency
    bilateral_skimage = None


SCRIPT_DIR = Path(__file__).resolve().parent
BARBARA_PATH = SCRIPT_DIR / "barbara.tif"


def load_barbara():
    return np.asarray(Image.open(BARBARA_PATH).convert("L"), dtype=np.float64) / 255.0


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def psnr(x, y):
    err = np.mean((x - y) ** 2)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / err)


def time_filter(fn, n_runs, warmup=True, **kwargs):
    if warmup:
        fn(**kwargs)
    times = []
    out = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(**kwargs)
        times.append(time.perf_counter() - t0)
    return np.mean(times), out


def main():
    clean = load_barbara()
    rng = np.random.default_rng(0)
    noise_sigma = 0.1
    noisy = np.clip(clean + noise_sigma * rng.standard_normal(clean.shape), 0.0, 1.0)

    sigma_spatial = 2.0
    sigma_range = 0.1
    radius = None

    t_numpy, out_numpy = time_filter(
        bilateral_filter_numpy,
        n_runs=3,
        warmup=True,
        image=noisy,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        radius=radius,
    )

    print(f"bilateral_filter_numpy: {t_numpy * 1000:.2f} ms")
    print(f"bilateral_filter_numpy PSNR: {psnr(clean, out_numpy):.2f} dB")

    if NUMBA_AVAILABLE:
        t_numba, out_numba = time_filter(
            bilateral_filter_numba,
            n_runs=3,
            warmup=True,
            image=noisy,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            radius=radius,
        )
        print(f"bilateral_filter_numba: {t_numba * 1000:.2f} ms")
        print(f"bilateral_filter_numba PSNR: {psnr(clean, out_numba):.2f} dB")
        print(f"numba vs numpy RMSE: {rmse(out_numba, out_numpy):.2e}")
    else:
        print("bilateral_filter_numba: skipped because numba is not installed")

    t_auto, out_auto = time_filter(
        bilateral_filter,
        n_runs=3,
        warmup=True,
        image=noisy,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        radius=radius,
    )

    print(f"bilateral_filter(auto): {t_auto * 1000:.2f} ms")
    print(f"bilateral_filter(auto) PSNR: {psnr(clean, out_auto):.2f} dB")
    print(f"auto vs numpy RMSE: {rmse(out_auto, out_numpy):.2e}")

    crop = noisy[192:320, 192:320]
    t_ref, out_ref = time_filter(
        bilateral_filter_reference,
        n_runs=1,
        warmup=False,
        image=crop,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        radius=radius,
        use_lab=False,
    )
    out_crop = bilateral_filter_numpy(
        crop,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        radius=radius,
    )

    print(f"reference bilateral_filter.py on 128x128 crop: {t_ref * 1000:.2f} ms")
    print(f"numpy vs reference crop RMSE: {rmse(out_crop, out_ref):.2e}")

    if bilateral_skimage is not None:
        t_skimage, out_skimage = time_filter(
            bilateral_skimage,
            n_runs=3,
            warmup=True,
            image=noisy,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_range,
            bins=10000,
            mode="reflect",
        )
        print(f"skimage: {t_skimage * 1000:.2f} ms")
        print(f"skimage PSNR: {psnr(clean, out_skimage):.2f} dB")
        print(f"numpy vs skimage RMSE: {rmse(out_numpy, out_skimage):.2e}")
    else:
        print("skimage bilateral benchmark: skipped because scikit-image is not installed")


if __name__ == "__main__":
    main()
