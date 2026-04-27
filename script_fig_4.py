"""
File Name: script_fig_4.py
Author: Kaibo
Function: Sweeps bilateral filtering parameters and plots PSNR heatmaps.
Reference: None.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    barbara, bu2010, 
    imnoise, 
    get_result_gray, get_result_rgb, get_result_lab,
)
from bf import bilateral_filter_numba as bf


rng = np.random.default_rng(0)
sigma = 0.1


# DataFrame
df = pd.DataFrame(columns=[
    "sigma_spatial", "sigma_range", "MSE", "PSNR", "SSIM", "Perceptual", "Method"
])


# gray
u_clean = barbara()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_gray(lambda x: x, u_noisy, u_clean)

mse_bf_lst, psnr_bf_lst, ssim_bf_lst, perceptual_bf_lst = [], [], [], []
for sigma_spatial in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    for sigma_range in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        u_bf, mse_bf, psnr_bf, ssim_bf, perceptual_bf = get_result_gray(bf, u_noisy, u_clean, sigma_spatial=sigma_spatial, sigma_range=sigma_range)
        mse_bf_lst.append(mse_bf)
        psnr_bf_lst.append(psnr_bf)
        ssim_bf_lst.append(ssim_bf)
        perceptual_bf_lst.append(perceptual_bf)
        df = pd.concat([df, pd.DataFrame({
            "sigma_spatial": sigma_spatial,
            "sigma_range": sigma_range,
            "MSE": mse_bf,
            "PSNR": psnr_bf,
            "SSIM": ssim_bf,
            "Perceptual": perceptual_bf,
            "Method": "Grayscale",
        }, index=[0])], ignore_index=True)
        print(df.tail(1))


# rgb
u_clean = bu2010()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_rgb(lambda x: x, u_noisy, u_clean)

mse_bf_lst_rgb, psnr_bf_lst_rgb, ssim_bf_lst_rgb, perceptual_bf_lst_rgb = [], [], [], []
for sigma_spatial in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    for sigma_range in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        u_bf_rgb, mse_bf_rgb, psnr_bf_rgb, ssim_bf_rgb, perceptual_bf_rgb = get_result_rgb(bf, u_noisy, u_clean, sigma_spatial=sigma_spatial, sigma_range=sigma_range)
        mse_bf_lst_rgb.append(mse_bf_rgb)
        psnr_bf_lst_rgb.append(psnr_bf_rgb)
        ssim_bf_lst_rgb.append(ssim_bf_rgb)
        perceptual_bf_lst_rgb.append(perceptual_bf_rgb)
        df = pd.concat([df, pd.DataFrame({
            "sigma_spatial": sigma_spatial,
            "sigma_range": sigma_range,
            "MSE": mse_bf_rgb,
            "PSNR": psnr_bf_rgb,
            "SSIM": ssim_bf_rgb,
            "Perceptual": perceptual_bf_rgb,
            "Method": "RGB (denoised in RGB)",
        }, index=[0])], ignore_index=True)
        print(df.tail(1))

mse_bf_lst_lab, psnr_bf_lst_lab, ssim_bf_lst_lab, perceptual_bf_lst_lab = [], [], [], []
for sigma_spatial in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    for sigma_range in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        u_bf_lab, mse_bf_lab, psnr_bf_lab, ssim_bf_lab, perceptual_bf_lab = get_result_lab(bf, u_noisy, u_clean, sigma_spatial=sigma_spatial, sigma_range=sigma_range)
        mse_bf_lst_lab.append(mse_bf_lab)
        psnr_bf_lst_lab.append(psnr_bf_lab)
        ssim_bf_lst_lab.append(ssim_bf_lab)
        perceptual_bf_lst_lab.append(perceptual_bf_lab)
        df = pd.concat([df, pd.DataFrame({
            "sigma_spatial": sigma_spatial,
            "sigma_range": sigma_range,
            "MSE": mse_bf_lab,
            "PSNR": psnr_bf_lab,
            "SSIM": ssim_bf_lab,
            "Perceptual": perceptual_bf_lab,
            "Method": "RGB (denoised in CIELAB)",
        }, index=[0])], ignore_index=True)
        print(df.tail(1))


# save DataFrame
df.to_csv("figures/fig4/fig4.csv", index=False)


# Figure 4
df = pd.read_csv("figures/fig4/fig4.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
this_df = df[df["Method"] == "Grayscale"]
sns.heatmap(
    this_df.pivot(columns="sigma_spatial", index="sigma_range", values="PSNR"), 
    ax=axes[0], annot=True, fmt=".2f", cbar_kws={"label": "PSNR (dB)"},
    square=True, vmin=20, vmax=26,
)
axes[0].set_xlabel(r"$\sigma_r$")
axes[0].set_ylabel(r"$\sigma_d$")
axes[0].set_title("Grayscale")
this_df = df[df["Method"] == "RGB (denoised in RGB)"]
sns.heatmap(
    this_df.pivot(columns="sigma_spatial", index="sigma_range", values="PSNR"), 
    ax=axes[1], annot=True, fmt=".2f", cbar_kws={"label": "PSNR (dB)"},
    square=True, vmin=20, vmax=26,
)
axes[1].set_xlabel(r"$\sigma_r$")
axes[1].set_ylabel(r"$\sigma_d$")
axes[1].set_title("RGB (denoised in RGB)")
this_df = df[df["Method"] == "RGB (denoised in CIELAB)"]
sns.heatmap(
    this_df.pivot(columns="sigma_spatial", index="sigma_range", values="PSNR"), 
    ax=axes[2], annot=True, fmt=".2f", cbar_kws={"label": "PSNR (dB)"},
    square=True, vmin=20, vmax=26,
)
axes[2].set_xlabel(r"$\sigma_r$")
axes[2].set_ylabel(r"$\sigma_d$")
axes[2].set_title("RGB (denoised in CIELAB)")
fig.savefig("figures/fig4/fig4.png", dpi=300)
plt.show()
