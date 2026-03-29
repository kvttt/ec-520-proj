from numba import njit, prange
import numpy as np
from scipy.ndimage import convolve


def _gauss_kernel(s):
    # https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/restoration/_nl_means_denoising.pyx#L164
    c = np.arange(s) - s // 2
    k = np.exp(-0.5 * (c / ((s - 1) / 4.0)) ** 2)
    k2d = np.outer(k, k)
    return k2d / k2d.sum()


def nlm_numpy(image, patch_size=7, patch_distance=11, h=0.1):
    kernel = _gauss_kernel(patch_size).astype(np.float64)
    img = np.asarray(image, dtype=np.float64)
    s, p = patch_distance, patch_size // 2
    padded = np.pad(img, s + p, mode="reflect")
    H, W = img.shape
    h2 = h * h
    C = np.zeros((H, W))
    I  = np.zeros((H, W))
    for dy2 in range(2 * s + 1):
        for dy1 in range(2 * s + 1):
            idx2 = dy2 + p
            idx1 = dy1 + p
            shifted = padded[idx2:idx2 + H, idx1:idx1 + W]
            dist = convolve((img - shifted) ** 2, kernel, mode="reflect")
            w = np.exp(-np.maximum(dist, 0.0) / h2)
            C += w
            I += w * shifted
    return np.clip(I / C, 0.0, 1.0)


@njit(parallel=True, cache=True)
def _nlm(padded, kernel, s, p, h2):
    pH, pW = padded.shape
    pad = s + p
    H, W = pH - 2 * pad, pW - 2 * pad
    out = np.empty((H, W), dtype=np.float64)
    for x2 in prange(H):
        for x1 in range(W):  # index the output pixel NL[u][x1, x2]
            x2pp, x1pp = x2 + pad, x1 + pad
            C, I = 0.0, 0.0  # normalizing constant, integral
            for dy2 in range(-s, s + 1):
                for dy1 in range(-s, s + 1):  # for evaluating the intergral
                    y2pp, y1pp = x2pp + dy2, x1pp + dy1
                    dist = 0.0
                    for kz2 in range(-p, p + 1):
                        for kz1 in range(-p, p + 1):  # for evaluating the exponential
                            d = padded[x2pp + kz2, x1pp + kz1] - padded[y2pp + kz2, y1pp + kz1]
                            dist += kernel[kz2 + p, kz1 + p] * d * d
                    w = np.exp(-max(dist, 0.0) / h2)
                    C += w
                    I += w * padded[y2pp, y1pp]
            out[x2, x1] = I / C
    return out


def nlm_numba(image, patch_size=7, patch_distance=11, h=0.1):
    kernel = _gauss_kernel(patch_size).astype(np.float64)
    img = np.asarray(image, dtype=np.float64)
    s, p = patch_distance, patch_size // 2
    padded = np.pad(img, s + p, mode="reflect")
    out = _nlm(padded, kernel, s, p, h * h)
    return np.clip(out, 0.0, 1.0)
