import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    barbara, bu2010, 
    imnoise, 
    get_result_gray, get_result_rgb, get_result_lab,
)
from nlm import nlm_numba as nlm
from bf import bilateral_filter_numba as bf


rng = np.random.default_rng(0)
sigma = 0.1


u_clean = barbara()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy = get_result_gray(lambda x: x, u_noisy, u_clean)

