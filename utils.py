import lpips
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import torch

import warnings
warnings.filterwarnings("ignore")


def barbara():
    return np.array(Image.open('data/barbara.tif')).astype(np.float64) / 255.0


def bu2010():
    return np.array(Image.open('data/bu2010.tif')).astype(np.float64) / 255.0


def imnoise(u, sigma, rng):
    return u + sigma * rng.standard_normal(u.shape)


def mse(u_clean, u_hat):
    return mean_squared_error(u_clean, u_hat)


def psnr(u_clean, u_hat):
    return peak_signal_noise_ratio(u_clean, u_hat, data_range=1)


def ssim(u_clean, u_hat):
    return structural_similarity(u_clean, u_hat, data_range=1, channel_axis=-1)


def get_result_gray(handle, u_noisy, u_clean, **kwargs):
    u_hat = handle(u_noisy, **kwargs)
    mse_val = mse(u_clean, u_hat)
    psnr_val = psnr(u_clean, u_hat)
    ssim_val = ssim(u_clean, u_hat)
    perceptual_val = perceptual(u_clean, u_hat)
    return u_hat, mse_val, psnr_val, ssim_val, perceptual_val


def get_result_rgb(handle, u_noisy, u_clean, **kwargs):
    u_hat = handle(u_noisy, **kwargs)
    mse_val = mse(u_clean, u_hat)
    psnr_val = psnr(u_clean, u_hat)
    ssim_val = ssim(u_clean, u_hat)
    perceptual_val = perceptual(u_clean, u_hat)
    return u_hat, mse_val, psnr_val, ssim_val, perceptual_val


def get_result_lab(handle, u_noisy, u_clean, **kwargs):
    u_lab = rgb2lab(u_noisy)
    u_lab_hat = handle(u_lab, **kwargs)
    u_hat = lab2rgb(u_lab_hat)
    mse_val = mse(u_clean, u_hat)
    psnr_val = psnr(u_clean, u_hat)
    ssim_val = ssim(u_clean, u_hat)
    perceptual_val = perceptual(u_clean, u_hat)
    return u_hat, mse_val, psnr_val, ssim_val, perceptual_val


def to_tensor(u):
    if u.ndim == 2:
        u = np.stack([u, u, u], axis=-1)
    return torch.tensor(u).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1


def perceptual(u_clean, u_hat):
    loss_fn = lpips.LPIPS(verbose=False)
    return loss_fn(to_tensor(u_clean), to_tensor(u_hat)).item()
