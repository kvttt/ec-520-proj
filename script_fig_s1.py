"""
File Name: script_fig_s1.py
Author: Kaibo
Function: Sweeps NLM patch and search-window sizes for supplemental analysis.
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
from nlm import nlm_numba as nlm


rng = np.random.default_rng(0)
sigma = 0.1


# DataFrame
df = pd.DataFrame(columns=[
    "patch_size", "patch_distance", "MSE", "PSNR", "SSIM", "Perceptual", "Method"
])


# gray
u_clean = barbara()
u_noisy = imnoise(u_clean, sigma, rng)
u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_gray(lambda x: x, u_noisy, u_clean)

mse_nlm_lst, psnr_nlm_lst, ssim_nlm_lst, perceptual_nlm_lst = [], [], [], []
for patch_size in [3, 5, 7, 9, 11]:
    for patch_distance in [5, 7, 9, 11, 13]:
        u_nlm, mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm = get_result_gray(nlm, u_noisy, u_clean, h=0.1, patch_size=patch_size, patch_distance=patch_distance)
        mse_nlm_lst.append(mse_nlm)
        psnr_nlm_lst.append(psnr_nlm)
        ssim_nlm_lst.append(ssim_nlm)
        perceptual_nlm_lst.append(perceptual_nlm)
        df = pd.concat([df, pd.DataFrame({
            "patch_size": patch_size,
            "patch_distance": patch_distance,
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
for patch_size in [3, 5, 7, 9, 11]:
    for patch_distance in [5, 7, 9, 11, 13]:
        u_nlm_rgb, mse_nlm_rgb, psnr_nlm_rgb, ssim_nlm_rgb, perceptual_nlm_rgb = get_result_rgb(nlm, u_noisy, u_clean, h=0.18, patch_size=patch_size, patch_distance=patch_distance)
        mse_nlm_lst_rgb.append(mse_nlm_rgb)
        psnr_nlm_lst_rgb.append(psnr_nlm_rgb)
        ssim_nlm_lst_rgb.append(ssim_nlm_rgb)
        perceptual_nlm_lst_rgb.append(perceptual_nlm_rgb)
        df = pd.concat([df, pd.DataFrame({
            "patch_size": patch_size,
            "patch_distance": patch_distance,
            "MSE": mse_nlm_rgb,
            "PSNR": psnr_nlm_rgb,
            "SSIM": ssim_nlm_rgb,
            "Perceptual": perceptual_nlm_rgb,
            "Method": "RGB (denoised in RGB)",
        }, index=[0])], ignore_index=True)
        print(df.tail(1))

mse_nlm_lst_lab, psnr_nlm_lst_lab, ssim_nlm_lst_lab, perceptual_nlm_lst_lab = [], [], [], []
for patch_size in [3, 5, 7, 9, 11]:
    for patch_distance in [5, 7, 9, 11, 13]:
        u_nlm_lab, mse_nlm_lab, psnr_nlm_lab, ssim_nlm_lab, perceptual_nlm_lab = get_result_lab(nlm, u_noisy, u_clean, h=18, patch_size=patch_size, patch_distance=patch_distance)
        mse_nlm_lst_lab.append(mse_nlm_lab)
        psnr_nlm_lst_lab.append(psnr_nlm_lab)
        ssim_nlm_lst_lab.append(ssim_nlm_lab)
        perceptual_nlm_lst_lab.append(perceptual_nlm_lab)
        df = pd.concat([df, pd.DataFrame({
            "patch_size": patch_size,
            "patch_distance": patch_distance,
            "MSE": mse_nlm_lab,
            "PSNR": psnr_nlm_lab,
            "SSIM": ssim_nlm_lab,
            "Perceptual": perceptual_nlm_lab,
            "Method": "RGB (denoised in CIELAB)",
        }, index=[0])], ignore_index=True)
        print(df.tail(1))


# DataFrame
df.to_csv("figures/figs1/figs1.csv", index=False)


# Figure S1
df = pd.read_csv("figures/figs1/figs1.csv")
fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
this_df = df[df["Method"] == "Grayscale"]
sns.heatmap(
    this_df.pivot(columns="patch_distance", index="patch_size", values="PSNR"), 
    ax=axes[0], annot=True, fmt=".2f", cbar_kws={"label": "PSNR (dB)"},
    square=True, vmin=23, vmax=29,
)
axes[0].set_xlabel("Search window size")
axes[0].set_ylabel("Neighborhood size")
axes[0].set_title("Grayscale")
this_df = df[df["Method"] == "RGB (denoised in RGB)"]
sns.heatmap(
    this_df.pivot(columns="patch_distance", index="patch_size", values="PSNR"), 
    ax=axes[1], annot=True, fmt=".2f", cbar_kws={"label": "PSNR (dB)"},
    square=True, vmin=23, vmax=29,
)
axes[1].set_xlabel("Search window size")
axes[1].set_ylabel("Neighborhood size")
axes[1].set_title("RGB (denoised in RGB)")
this_df = df[df["Method"] == "RGB (denoised in CIELAB)"]
sns.heatmap(
    this_df.pivot(columns="patch_distance", index="patch_size", values="PSNR"), 
    ax=axes[2], annot=True, fmt=".2f", cbar_kws={"label": "PSNR (dB)"},
    square=True, vmin=23, vmax=29,
)
axes[2].set_xlabel("Search window size")
axes[2].set_ylabel("Neighborhood size")
axes[2].set_title("RGB (denoised in CIELAB)")
fig.savefig("figures/figs1/figs1.png", dpi=300)
plt.show()
