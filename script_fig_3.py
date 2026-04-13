import matplotlib.pyplot as plt
import numpy as np

from utils import (
    barbara, bu2010, 
    imnoise, 
    get_result_gray, get_result_rgb, get_result_lab,
)
from nlm import nlm_numba as nlm


rng = np.random.default_rng(0)
sigma = 0.1


# gray
u_clean = barbara()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy = get_result_gray(lambda x: x, u_noisy, u_clean)

mse_nlm_lst, psnr_nlm_lst, ssim_nlm_lst = [], [], []
for h in [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]:
    u_nlm, mse_nlm, psnr_nlm, ssim_nlm = get_result_gray(nlm, u_noisy, u_clean, h=h)
    mse_nlm_lst.append(mse_nlm)
    psnr_nlm_lst.append(psnr_nlm)
    ssim_nlm_lst.append(ssim_nlm)


# rgb
u_clean = bu2010()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy = get_result_rgb(lambda x: x, u_noisy, u_clean)

mse_nlm_lst_rgb, psnr_nlm_lst_rgb, ssim_nlm_lst_rgb = [], [], []
for h in [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]:
    u_nlm_rgb, mse_nlm_rgb, psnr_nlm_rgb, ssim_nlm_rgb = get_result_rgb(nlm, u_noisy, u_clean, h=h)
    mse_nlm_lst_rgb.append(mse_nlm_rgb)
    psnr_nlm_lst_rgb.append(psnr_nlm_rgb)
    ssim_nlm_lst_rgb.append(ssim_nlm_rgb)

mse_nlm_lst_lab, psnr_nlm_lst_lab, ssim_nlm_lst_lab = [], [], []
for h in [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]:
    u_nlm_lab, mse_nlm_lab, psnr_nlm_lab, ssim_nlm_lab = get_result_lab(nlm, u_noisy, u_clean, h=h)
    mse_nlm_lst_lab.append(mse_nlm_lab)
    psnr_nlm_lst_lab.append(psnr_nlm_lab)
    ssim_nlm_lst_lab.append(ssim_nlm_lab)


# Figure 3
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
axes[0].plot([0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200], mse_nlm_lst, marker='o')
axes[0].set_xlabel("h")
axes[0].set_title("Grayscale")
axes[1].plot([0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200], mse_nlm_lst_rgb, marker='o')
axes[1].set_xlabel("h")
axes[1].set_title("RGB")
axes[2].plot([2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0], mse_nlm_lst_lab, marker='o')
axes[2].set_xlabel("h")
axes[2].set_title("LAB")
fig.savefig("fig3.png", dpi=300)
plt.show()

