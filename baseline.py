import numpy as np
from scipy.ndimage import correlate
from skimage.filters import median

from nlm import _gauss_kernel


def my_median(u_noisy):  # TODO: need to implement from scratch
    squeeze_output = False
    if u_noisy.ndim == 2:
        u_noisy = u_noisy[:, :, np.newaxis]
        squeeze_output = True
    u_hat = np.zeros_like(u_noisy)
    for c in range(u_noisy.shape[2]):
        u_hat[:, :, c] = median(u_noisy[:, :, c])
    if squeeze_output:
        u_hat = u_hat[:, :, 0]
    return u_hat


def my_bayesian_markov(u_noisy, lambda_reg=1.0, n_iters=3):  # TODO: need to implement from scratch
    return u_noisy


def _wiener(img, radius=1):
    size = 2 * radius + 1
    k = np.full((size, size), 1.0 / (size * size), dtype=np.float64)
    mean = correlate(img, k[:, :, np.newaxis], mode="reflect")
    mean2 = correlate(img * img, k[:, :, np.newaxis], mode="reflect")
    var = np.maximum(mean2 - mean * mean, 0.0)
    noise = var.mean(axis=(0, 1), keepdims=True)
    gain = np.maximum(var - noise, 0.0) / (var + 1e-12)
    return mean + gain * (img - mean)


def _gaussian(img, s):
    k2d = _gauss_kernel(s)
    return correlate(img, k2d[:, :, np.newaxis], mode="reflect")


def my_wiener(u_noisy):
    img = np.asarray(u_noisy, dtype=np.float64)
    squeeze_output = False
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze_output = True
    u_hat = _wiener(img, radius=1)
    if squeeze_output:
        u_hat = u_hat[:, :, 0]
    return u_hat


def my_gaussian(u_noisy, sigma=1.0):
    img = np.asarray(u_noisy, dtype=np.float64)
    squeeze_output = False
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze_output = True
    s = max(3, 2 * int(2.0 * float(sigma)) + 1)
    u_hat = _gaussian(img, s)
    if squeeze_output:
        u_hat = u_hat[:, :, 0]
    return u_hat
