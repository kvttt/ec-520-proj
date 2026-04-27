"""
File Name: script_fig_3.py
Author: Kaibo
Function: Sweeps NLM smoothing strength and plots its effect on denoising quality.
Reference: None.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    barbara, bu2010, 
    imnoise, 
    get_result_gray, get_result_rgb, get_result_lab,
)
from nlm import nlm_numba as nlm


rng = np.random.default_rng(0)
sigma = 0.1


# DataFrame
df = pd.DataFrame(columns=[
    "h", "MSE", "PSNR", "SSIM", "Perceptual", "Method"
])


# gray
u_clean = barbara()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_gray(lambda x: x, u_noisy, u_clean)

mse_nlm_lst, psnr_nlm_lst, ssim_nlm_lst, perceptual_nlm_lst = [], [], [], []
for h in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
    u_nlm, mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm = get_result_gray(nlm, u_noisy, u_clean, h=h)
    mse_nlm_lst.append(mse_nlm)
    psnr_nlm_lst.append(psnr_nlm)
    ssim_nlm_lst.append(ssim_nlm)
    perceptual_nlm_lst.append(perceptual_nlm)
    df = pd.concat([df, pd.DataFrame({
        "h": h,
        "MSE": mse_nlm,
        "PSNR": psnr_nlm,
        "SSIM": ssim_nlm,
        "Perceptual": perceptual_nlm,
        "Method": "Grayscale",
    }, index=[0])], ignore_index=True)
    print(df.tail(1))


# rgb
u_clean = bu2010()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_rgb(lambda x: x, u_noisy, u_clean)

mse_nlm_lst_rgb, psnr_nlm_lst_rgb, ssim_nlm_lst_rgb, perceptual_nlm_lst_rgb = [], [], [], []
for h in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
    u_nlm_rgb, mse_nlm_rgb, psnr_nlm_rgb, ssim_nlm_rgb, perceptual_nlm_rgb = get_result_rgb(nlm, u_noisy, u_clean, h=h)
    mse_nlm_lst_rgb.append(mse_nlm_rgb)
    psnr_nlm_lst_rgb.append(psnr_nlm_rgb)
    ssim_nlm_lst_rgb.append(ssim_nlm_rgb)
    perceptual_nlm_lst_rgb.append(perceptual_nlm_rgb)
    df = pd.concat([df, pd.DataFrame({
        "h": h,
        "MSE": mse_nlm_rgb,
        "PSNR": psnr_nlm_rgb,
        "SSIM": ssim_nlm_rgb,
        "Perceptual": perceptual_nlm_rgb,
        "Method": "RGB (denoised in RGB)",
    }, index=[0])], ignore_index=True)
    print(df.tail(1))

mse_nlm_lst_lab, psnr_nlm_lst_lab, ssim_nlm_lst_lab, perceptual_nlm_lst_lab = [], [], [], []
for h in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]:
    u_nlm_lab, mse_nlm_lab, psnr_nlm_lab, ssim_nlm_lab, perceptual_nlm_lab = get_result_lab(nlm, u_noisy, u_clean, h=h)
    mse_nlm_lst_lab.append(mse_nlm_lab)
    psnr_nlm_lst_lab.append(psnr_nlm_lab)
    ssim_nlm_lst_lab.append(ssim_nlm_lab)
    perceptual_nlm_lst_lab.append(perceptual_nlm_lab)
    df = pd.concat([df, pd.DataFrame({
        "h": h,
        "MSE": mse_nlm_lab,
        "PSNR": psnr_nlm_lab,
        "SSIM": ssim_nlm_lab,
        "Perceptual": perceptual_nlm_lab,
        "Method": "RGB (denoised in CIELAB)",
    }, index=[0])], ignore_index=True)
    print(df.tail(1))


# DataFrame
df.to_csv("figures/fig3/fig3.csv", index=False)


# Figure 3
df = pd.read_csv("figures/fig3/fig3.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
this_df = df[df["Method"] == "Grayscale"]
axes[0].plot(this_df["h"], this_df["PSNR"], marker='o')
axes[0].set_xlabel(r"$h$")
axes[0].set_ylabel("PSNR (dB)")
axes[0].set_title("Grayscale")
axes[0].set_ylim(19.5, 28.5)
axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
axes[0].set_xticks(this_df["h"])
axes[0].set_xticklabels([f"{h:.2f}" for h in this_df["h"]])
this_df = df[df["Method"] == "RGB (denoised in RGB)"]
axes[1].plot(this_df["h"], this_df["PSNR"], marker='o')
axes[1].set_xlabel(r"$h$")
axes[1].set_ylabel("PSNR (dB)")
axes[1].set_title("RGB (denoised in RGB)")
axes[1].set_ylim(19.5, 28.5)
axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
axes[1].set_xticks(this_df["h"])
axes[1].set_xticklabels([f"{h:.2f}" for h in this_df["h"]])
this_df = df[df["Method"] == "RGB (denoised in CIELAB)"]
axes[2].plot(this_df["h"], this_df["PSNR"], marker='o')
axes[2].set_xlabel(r"$h$")
axes[2].set_ylabel("PSNR (dB)")
axes[2].set_title("RGB (denoised in CIELAB)")
axes[2].set_ylim(19.5, 28.5)
axes[2].grid(True, which='both', linestyle='--', alpha=0.5)
axes[2].set_xticks(this_df["h"])
axes[2].set_xticklabels([f"{h:.0f}" for h in this_df["h"]])
fig.savefig("figures/fig3/fig3.png", dpi=300)
plt.show()
