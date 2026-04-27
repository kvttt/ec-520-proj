"""
File Name: script_dinov3_dataset.py
Author: Jiatong
Function: Provides the DINOv3-ViT denoising dataset with synthetic noise.
Reference: PyTorch Dataset API, https://pytorch.org/docs/stable/data.html.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _list_images(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])


def _augment_image(image: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return image
    if mode == 1:
        return np.flipud(image)
    if mode == 2:
        return np.rot90(image)
    if mode == 3:
        return np.flipud(np.rot90(image))
    if mode == 4:
        return np.rot90(image, k=2)
    if mode == 5:
        return np.flipud(np.rot90(image, k=2))
    if mode == 6:
        return np.rot90(image, k=3)
    if mode == 7:
        return np.flipud(np.rot90(image, k=3))
    raise ValueError(f"Unsupported augmentation mode: {mode}")


class DenoiseDataset(Dataset):
    def __init__(
        self,
        *,
        root: Path,
        phase: str,
        patch_size: int = 64,
        sigma: int = 25,
        noise_type: str = "gaussian",
        amount: float = 0.1,
        salt_vs_pepper: float = 0.5,
    ) -> None:
        super().__init__()
        if phase not in {"train", "test"}:
            raise ValueError("phase must be 'train' or 'test'.")
        if noise_type not in {"gaussian", "salt_pepper"}:
            raise ValueError("noise_type must be 'gaussian' or 'salt_pepper'.")

        self.root = root
        self.phase = phase
        self.patch_size = patch_size
        self.sigma = sigma
        self.noise_type = noise_type
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
        self.paths = _list_images(root)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_rgb(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def _add_train_noise(self, clean: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "gaussian":
            noisy = clean + torch.randn_like(clean) * (self.sigma / 255.0)
            return noisy

        noisy = clean.clone()
        mask = torch.rand((1, noisy.shape[1], noisy.shape[2]), dtype=noisy.dtype)
        pepper = mask < self.amount * (1.0 - self.salt_vs_pepper)
        salt = (mask >= self.amount * (1.0 - self.salt_vs_pepper)) & (mask < self.amount)
        keep = ~(pepper | salt)
        return noisy * keep.to(noisy.dtype) + salt.to(noisy.dtype)

    def _add_test_noise(self, clean: np.ndarray) -> np.ndarray:
        if self.noise_type == "gaussian":
            rng = np.random.default_rng(seed=0)
            return clean + (self.sigma / 255.0) * rng.standard_normal(clean.shape)

        rng = np.random.default_rng(seed=0)
        noisy = np.array(clean, copy=True)
        mask = rng.random(noisy.shape[:2])
        pepper = mask < self.amount * (1.0 - self.salt_vs_pepper)
        salt = (mask >= self.amount * (1.0 - self.salt_vs_pepper)) & (mask < self.amount)
        noisy[pepper, :] = 0.0
        noisy[salt, :] = 1.0
        return noisy

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        clean = self._load_rgb(path)

        if self.phase == "train":
            height, width = clean.shape[:2]
            crop_h = min(self.patch_size, height)
            crop_w = min(self.patch_size, width)
            rnd_h = random.randint(0, max(0, height - crop_h))
            rnd_w = random.randint(0, max(0, width - crop_w))
            clean = clean[rnd_h : rnd_h + crop_h, rnd_w : rnd_w + crop_w, :]
            clean = _augment_image(clean, random.randint(0, 7)).copy()
            clean_tensor = torch.from_numpy(np.transpose(clean, (2, 0, 1)))
            noisy_tensor = self._add_train_noise(clean_tensor)
        else:
            noisy = self._add_test_noise(clean)
            clean_tensor = torch.from_numpy(np.transpose(clean, (2, 0, 1)))
            noisy_tensor = torch.from_numpy(np.transpose(noisy, (2, 0, 1)))

        return {
            "L": noisy_tensor.float(),
            "H": clean_tensor.float(),
            "H_path": str(path),
            "L_path": str(path),
        }
