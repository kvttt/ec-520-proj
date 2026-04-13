import numpy as np
from skimage.filters import gaussian, median
from scipy.signal import wiener


def my_wiener(u_noisy):  # TODO: need to implement from scratch
    squeeze_output = False
    if u_noisy.ndim == 2:
        u_noisy = u_noisy[:, :, np.newaxis]
        squeeze_output = True
    u_hat = np.zeros_like(u_noisy)
    for c in range(u_noisy.shape[2]):
        u_hat[:, :, c] = wiener(u_noisy[:, :, c])
    if squeeze_output:
        u_hat = u_hat[:, :, 0]
    return u_hat


def my_gaussian(u_noisy, sigma=1.0):  # TODO: need to implement from scratch
    if u_noisy.ndim == 2:
        return gaussian(u_noisy, sigma=sigma)
    else:
        return gaussian(u_noisy, sigma=sigma, channel_axis=-1)
    

def my_median(u_noisy, patch_size=3):  # TODO: need to implement from scratch
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


def my_bayesian_markov(u_noisy):  # TODO: need to implement from scratch
    return u_noisy
