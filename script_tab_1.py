import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils import imnoise, get_result_rgb
from nlm import nlm_numba as nlm
from bf import bilateral_filter_numba as bf
from baseline import (
    my_gaussian, my_median, my_wiener, my_bayesian_markov,
)

import glob
import os


rng = np.random.default_rng(0)
sigma = 0.1


# DataFrame
df = pd.DataFrame(columns=[
    "Image ID", "MSE", "PSNR", "SSIM", "Perceptual", "Method"
])

for ds in ["BSD100", "Urban100"]:
    for fn in sorted(glob.glob(f"data/{ds}/*.png")):
        imgid = f'{ds}_{os.path.basename(fn).split(".")[0].split("_")[1]}'
        img = np.array(Image.open(fn)).astype(np.float64) / 255.0
        u_clean = img
        u_noisy = imnoise(u_clean, sigma, rng)
        u_noisy, mse_noisy, psnr_noisy, ssim_noisy, perceptual_noisy = get_result_rgb(lambda x: x, u_noisy, u_clean)
        u_gausian, mse_gausian, psnr_gausian, ssim_gausian, perceptual_gausian = get_result_rgb(my_gaussian, u_noisy, u_clean)
        u_median, mse_median, psnr_median, ssim_median, perceptual_median = get_result_rgb(my_median, u_noisy, u_clean)
        u_wiener, mse_wiener, psnr_wiener, ssim_wiener, perceptual_wiener = get_result_rgb(my_wiener, u_noisy, u_clean)
        u_bayesian_markov, mse_bayesian_markov, psnr_bayesian_markov, ssim_bayesian_markov, perceptual_bayesian_markov = get_result_rgb(my_bayesian_markov, u_noisy, u_clean)
        u_nlm, mse_nlm, psnr_nlm, ssim_nlm, perceptual_nlm = get_result_rgb(nlm, u_noisy, u_clean, h=0.18)
        u_bf, mse_bf, psnr_bf, ssim_bf, perceptual_bf = get_result_rgb(bf, u_noisy, u_clean, sigma_spatial=1.5, sigma_range=0.25)
        df = pd.concat([df, pd.DataFrame({
            "Image ID": imgid,
            "MSE": mse_noisy,
            "PSNR": psnr_noisy,
            "SSIM": ssim_noisy,
            "Perceptual": perceptual_noisy,
            "Method": "Noisy",
        }, index=[0])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({
            "Image ID": imgid,
            "MSE": mse_gausian,
            "PSNR": psnr_gausian,
            "SSIM": ssim_gausian,
            "Perceptual": perceptual_gausian,
            "Method": "Gaussian",
        }, index=[0])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({
            "Image ID": imgid,
            "MSE": mse_median,
            "PSNR": psnr_median,
            "SSIM": ssim_median,
            "Perceptual": perceptual_median,
            "Method": "Median",
        }, index=[0])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({
            "Image ID": imgid,
            "MSE": mse_wiener,
            "PSNR": psnr_wiener,
            "SSIM": ssim_wiener,
            "Perceptual": perceptual_wiener,
            "Method": "Wiener",
        }, index=[0])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({
            "Image ID": imgid,
            "MSE": mse_bayesian_markov,
            "PSNR": psnr_bayesian_markov,
            "SSIM": ssim_bayesian_markov,
            "Perceptual": perceptual_bayesian_markov,
            "Method": "Bayesian Markov",
        }, index=[0])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({
            "Image ID": imgid,
            "MSE": mse_nlm,
            "PSNR": psnr_nlm,
            "SSIM": ssim_nlm,
            "Perceptual": perceptual_nlm,
            "Method": "NLM",
        }, index=[0])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({
            "Image ID": imgid,
            "MSE": mse_bf,
            "PSNR": psnr_bf,
            "SSIM": ssim_bf,
            "Perceptual": perceptual_bf,
            "Method": "BF",
        }, index=[0])], ignore_index=True)
        print(df.tail(8))

df.to_csv("tables/tab1/tab1_raw.csv", index=False)


df = pd.read_csv("tables/tab1/tab1_raw.csv")

mse_bsd100_df = df[df["Image ID"].str.contains("BSD100")].pivot_table(index="Image ID", columns="Method", values="MSE")
psnr_bsd100_df = df[df["Image ID"].str.contains("BSD100")].pivot_table(index="Image ID", columns="Method", values="PSNR")
ssim_bsd100_df = df[df["Image ID"].str.contains("BSD100")].pivot_table(index="Image ID", columns="Method", values="SSIM")
perceptual_bsd100_df = df[df["Image ID"].str.contains("BSD100")].pivot_table(index="Image ID", columns="Method", values="Perceptual")
mse_urban100_df = df[df["Image ID"].str.contains("Urban100")].pivot_table(index="Image ID", columns="Method", values="MSE")
psnr_urban100_df = df[df["Image ID"].str.contains("Urban100")].pivot_table(index="Image ID", columns="Method", values="PSNR")
ssim_urban100_df = df[df["Image ID"].str.contains("Urban100")].pivot_table(index="Image ID", columns="Method", values="SSIM")
perceptual_urban100_df = df[df["Image ID"].str.contains("Urban100")].pivot_table(index="Image ID", columns="Method", values="Perceptual")

order = ["Noisy", "Gaussian", "Median", "Wiener", "Bayesian Markov", "NLM", "BF"]

bsd100_summary_df = pd.DataFrame({
    "Method": mse_bsd100_df.columns,
    "MSE (mean)": mse_bsd100_df.mean().values,
    "MSE (std)": mse_bsd100_df.std().values,
    "PSNR (mean)": psnr_bsd100_df.mean().values,
    "PSNR (std)": psnr_bsd100_df.std().values,
    "SSIM (mean)": ssim_bsd100_df.mean().values,
    "SSIM (std)": ssim_bsd100_df.std().values,
    "Perceptual (mean)": perceptual_bsd100_df.mean().values,
    "Perceptual (std)": perceptual_bsd100_df.std().values,
})
bsd100_summary_df = bsd100_summary_df.set_index("Method").loc[order].reset_index()
bsd100_summary_df.to_csv("tables/tab1/tab1_stats_bsd100.csv", index=False)
print(bsd100_summary_df)

urban100_summary_df = pd.DataFrame({
    "Method": mse_urban100_df.columns,
    "MSE (mean)": mse_urban100_df.mean().values,
    "MSE (std)": mse_urban100_df.std().values,
    "PSNR (mean)": psnr_urban100_df.mean().values,
    "PSNR (std)": psnr_urban100_df.std().values,
    "SSIM (mean)": ssim_urban100_df.mean().values,
    "SSIM (std)": ssim_urban100_df.std().values,
    "Perceptual (mean)": perceptual_urban100_df.mean().values,
    "Perceptual (std)": perceptual_urban100_df.std().values,
})
urban100_summary_df = urban100_summary_df.set_index("Method").loc[order].reset_index()
urban100_summary_df.to_csv("tables/tab1/tab1_stats_urban100.csv", index=False)
print(urban100_summary_df)
