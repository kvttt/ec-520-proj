"""
File Name: baseline.py
Author: Jiatong & Kaibo
Function: Implements classical baseline denoisers including median, Wiener, Gaussian, Bayesian-Markov, and Huber-Markov methods.
Reference: Geman and Geman (1984), https://doi.org/10.1109/TPAMI.1984.4767596; Huber (1964), https://doi.org/10.1214/aoms/1177703732.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import correlate
from numba import njit
from nlm import _gauss_kernel


# ''''''''''''''''''''''''''''''''''Median''''''''''''''''''''''''''''''''''

def _median_filter_2d(channel):
    padded = np.pad(channel, 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    flat_windows = windows.reshape(channel.shape[0], channel.shape[1], 9)
    return np.partition(flat_windows, 4, axis=-1)[..., 4]


def my_median(u_noisy):
    u_noisy = np.asarray(u_noisy, dtype=np.float64)
    squeeze_output = False
    if u_noisy.ndim == 2:
        u_noisy = u_noisy[:, :, np.newaxis]
        squeeze_output = True
    u_hat = np.zeros_like(u_noisy)
    for c in range(u_noisy.shape[2]):
        u_hat[:, :, c] = _median_filter_2d(u_noisy[:, :, c])
    if squeeze_output:
        u_hat = u_hat[:, :, 0]
    return u_hat


# ''''''''''''''''''''''''''''''''''Bayesian_Markov''''''''''''''''''''''''''''''''''

def _continuous_abs_local_minimizer(v_ij, neighbors, lambda_reg):
    neighbors = np.asarray(neighbors, dtype=np.float64)
    if neighbors.size == 0:
        return np.clip(v_ij, 0.0, 1.0)

    sorted_neighbors = np.sort(neighbors)
    m = sorted_neighbors.size
    
    # derivative of the object function: 2(u-v_ij)+lambda*(2k-m)=0
    # u = v_ij - lambda/2 * (2k-m)
    roots = v_ij - 0.5 * lambda_reg * (2 * np.arange(m + 1) - m)
    left_edges = np.concatenate(([-np.inf], sorted_neighbors))
    right_edges = np.concatenate((sorted_neighbors, [np.inf]))
    valid_roots = roots[(left_edges - 1e-12 <= roots) & (roots <= right_edges + 1e-12)]
    in_range_neighbors = sorted_neighbors[(0.0 <= sorted_neighbors) & (sorted_neighbors <= 1.0)]

    candidates = np.concatenate((
        np.array([0.0, 1.0], dtype=np.float64),
        in_range_neighbors,
        np.clip(valid_roots, 0.0, 1.0),
    ))
    costs = (v_ij - candidates) ** 2 + lambda_reg * np.abs(candidates[:, np.newaxis] - neighbors).sum(axis=1)
    return candidates[np.argmin(costs)]


def _quantized_abs_local_minimizer(v_ij, neighbors, lambda_reg):
    cont = _continuous_abs_local_minimizer(v_ij, neighbors, lambda_reg)
    low_idx = int(np.floor(cont * 255.0))
    high_idx = int(np.ceil(cont * 255.0))

    best_val = low_idx / 255.0
    best_cost = (v_ij - best_val) ** 2 + lambda_reg * np.abs(best_val - neighbors).sum()

    if high_idx != low_idx:
        cand_val = high_idx / 255.0
        cand_cost = (v_ij - cand_val) ** 2 + lambda_reg * np.abs(cand_val - neighbors).sum()
        if cand_cost < best_cost:
            best_val = cand_val

    return best_val


def _bayesian_markov_channel(channel, lambda_reg, n_iters):
    y = np.asarray(channel, dtype=np.float64)
    u = y.copy()
    h, w = u.shape
    neighbors = np.empty(4, dtype=np.float64)

    for _ in range(int(n_iters)):
        for i in range(h):
            for j in range(w):
                n_neighbors = 0
                if i > 0:
                    neighbors[n_neighbors] = u[i - 1, j]
                    n_neighbors += 1
                if i + 1 < h:
                    neighbors[n_neighbors] = u[i + 1, j]
                    n_neighbors += 1
                if j > 0:
                    neighbors[n_neighbors] = u[i, j - 1]
                    n_neighbors += 1
                if j + 1 < w:
                    neighbors[n_neighbors] = u[i, j + 1]
                    n_neighbors += 1

                u[i, j] = _quantized_abs_local_minimizer(
                    y[i, j], neighbors[:n_neighbors], lambda_reg,
                )
    return u

def my_bayesian_markov(u_noisy, lambda_reg=0.05, n_iters=10):
    img = np.asarray(u_noisy, dtype=np.float64)
    squeeze_output = False
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze_output = True

    u_hat = np.empty_like(img)
    for c in range(img.shape[2]):
        u_hat[:, :, c] = _bayesian_markov_channel(img[:, :, c], lambda_reg, n_iters)

    if squeeze_output:
        return u_hat[:, :, 0]
    return u_hat

# ''''''''''''''''''''''''''''''''''Huber_Markov''''''''''''''''''''''''''''''''''

def my_Huber_Markov(u_noisy, lambda_reg=0.8, n_iters=10, huber_delta=0.08):
    img = np.asarray(u_noisy, dtype=np.float64)
    squeeze_output = False
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze_output = True

    u = img.copy()
    y = img
    for _ in range(int(n_iters)):
        padded = np.pad(u, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        center = padded[1:-1, 1:-1, :]
        neighbors = [
            padded[:-2, 1:-1, :],
            padded[2:, 1:-1, :],
            padded[1:-1, :-2, :],
            padded[1:-1, 2:, :],
        ]

        numer = y.copy()
        denom = np.ones(center.shape[:2], dtype=np.float64)
        for neighbor in neighbors:
            diff = neighbor - center
            dist = np.sqrt(np.sum(diff * diff, axis=-1))
            weights = np.minimum(1.0, huber_delta / (dist + 1e-12))
            numer += lambda_reg * weights[:, :, np.newaxis] * neighbor
            denom += lambda_reg * weights

        u = np.clip(numer / denom[:, :, np.newaxis], 0.0, 1.0)

    if squeeze_output:
        return u[:, :, 0]
    return u

# ''''''''''''''''''''''''''''''''''Wiener''''''''''''''''''''''''''''''''''

def _wiener(img, radius=1):
    size = 2 * radius + 1
    k = np.full((size, size), 1.0 / (size * size), dtype=np.float64)
    mean = correlate(img, k[:, :, np.newaxis], mode="reflect")
    mean2 = correlate(img * img, k[:, :, np.newaxis], mode="reflect")
    var = np.maximum(mean2 - mean * mean, 0.0)
    noise = var.mean(axis=(0, 1), keepdims=True)
    gain = np.maximum(var - noise, 0.0) / (var + 1e-12)
    return mean + gain * (img - mean)

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


# ''''''''''''''''''''''''''''''''''Gaussian''''''''''''''''''''''''''''''''''

def _gaussian(img, s):
    k2d = _gauss_kernel(s)
    return correlate(img, k2d[:, :, np.newaxis], mode="reflect")


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
