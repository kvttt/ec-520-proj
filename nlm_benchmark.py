import time

import numpy as np
from skimage.data import camera
from skimage.restoration import denoise_nl_means as nlm_skimage

from nlm import nlm_numpy, nlm_numba


img = camera().astype(np.float64) / 255.0
noisy = img + 0.1 * np.random.randn(*img.shape)
noisy = np.clip(noisy, 0, 1)


def time_nlm(fn, n_runs, warmup=True, **kwargs):
    if warmup:
        fn(**kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(**kwargs)
        times.append(time.perf_counter() - t0)
    return np.mean(times), out


t_skimage, out_skimage = time_nlm(
    nlm_skimage, n_runs=10, warmup=True, 
    image=noisy, h=0.1, patch_size=7, patch_distance=11, preserve_range=True, fast_mode=False,
)
t_numpy, out_numpy = time_nlm(
    nlm_numpy, n_runs=10, warmup=True, 
    image=noisy, patch_size=7, patch_distance=11, h=0.1,
)
t_numba, out_numba = time_nlm(
    nlm_numba, n_runs=10, warmup=True, 
    image=noisy, patch_size=7, patch_distance=11, h=0.1,
)


print(f"skimage: {t_skimage * 1000:.2f} ms")
print(f"nlm_numpy: {t_numpy * 1000:.2f} ms")
print(f"nlm_numba: {t_numba * 1000:.2f} ms")

print(f"nlm_numpy vs skimage: RMSE = {np.sqrt(np.mean((out_numpy - out_skimage) ** 2)):.2e}")
print(f"nlm_numba vs skimage: RMSE = {np.sqrt(np.mean((out_numba - out_skimage) ** 2)):.2e}")
