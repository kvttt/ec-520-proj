"""
File Name: script_dinov3_denoiser.py
Author: Jiatong
Function: Defines and loads the DINOv3-based ViT image denoiser.
Reference: DINOv3, https://github.com/facebookresearch/dinov3; timm, https://github.com/huggingface/pytorch-image-models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_TORCHVISION_COMPAT_LIB = None
_DINOV3_MODELS = {}


def _ensure_torchvision_nms_schema() -> None:
    global _TORCHVISION_COMPAT_LIB

    if _TORCHVISION_COMPAT_LIB is not None:
        return

    try:
        from torch.library import Library

        _TORCHVISION_COMPAT_LIB = Library("torchvision", "DEF")
        _TORCHVISION_COMPAT_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


def add_external_paths(project_root: Path) -> None:
    kair_root = project_root / "external" / "KAIR"
    dinov3_root = project_root / "external" / "dinov3"
    src_root = project_root / "src"
    sys.path = [p for p in sys.path if p != str(src_root)]
    for root in (kair_root, dinov3_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvNormAct(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(x)


class UpCatBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvNormAct(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class DINOv3Denoiser(nn.Module):
    def __init__(
        self,
        *,
        project_root: Path,
        backbone_backend: str = "auto",
        pretrained_backbone: bool = True,
        backbone_weights: str | None = None,
        pretrained_model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        feature_layers: Iterable[int] = (8, 9, 10, 11),
        decoder_channels: tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        add_external_paths(project_root)
        self.pretrained_model_name = pretrained_model_name
        self.feature_layers = tuple(feature_layers)

        use_timm = backbone_backend == "timm"
        if backbone_backend == "auto":
            use_timm = pretrained_backbone and backbone_weights is None

        if use_timm:
            _ensure_torchvision_nms_schema()
            import timm

            if backbone_weights is not None:
                self.backbone = timm.create_model(
                    pretrained_model_name,
                    pretrained=False,
                    num_classes=0,
                    checkpoint_path=backbone_weights,
                )
            else:
                self.backbone = timm.create_model(pretrained_model_name, pretrained=pretrained_backbone, num_classes=0)
            self.backbone_backend = "timm"
            self.patch_size = int(self.backbone.patch_embed.patch_size[0])
            self.embed_dim = int(self.backbone.embed_dim)
            self.num_register_tokens = 0
        else:
            from dinov3.hub.backbones import dinov3_vits16

            if backbone_weights is None:
                self.backbone = dinov3_vits16(pretrained=pretrained_backbone)
            else:
                self.backbone = dinov3_vits16(pretrained=pretrained_backbone, weights=backbone_weights)
            self.backbone_backend = "official"
            self.patch_size = int(self.backbone.patch_size)
            self.embed_dim = int(self.backbone.embed_dim)
            self.num_register_tokens = int(self.backbone.n_storage_tokens)

        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.stem = ConvNormAct(3, 32)
        self.enc1 = ConvNormAct(32, 64)
        self.enc2 = ConvNormAct(64, 128)
        self.enc3 = ConvNormAct(128, 192)
        self.enc4 = ConvNormAct(192, 256)

        self.vit_fuse = ConvNormAct(self.embed_dim * len(self.feature_layers), 256)
        self.bottleneck = ConvNormAct(256 + 256, decoder_channels[0])
        self.up3 = UpCatBlock(decoder_channels[0], 192, decoder_channels[1])
        self.up2 = UpCatBlock(decoder_channels[1], 128, decoder_channels[2])
        self.up1 = UpCatBlock(decoder_channels[2], 64, decoder_channels[3])
        self.up0 = UpCatBlock(decoder_channels[3], 32, decoder_channels[4])
        self.head = nn.Conv2d(decoder_channels[4], 3, kernel_size=3, padding=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.head.bias)

    def parameter_groups(self, backbone_lr: float, decoder_lr: float) -> list[dict]:
        decoder_modules = [
            self.stem,
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
            self.vit_fuse,
            self.bottleneck,
            self.up3,
            self.up2,
            self.up1,
            self.up0,
            self.head,
        ]
        decoder_params = []
        for module in decoder_modules:
            decoder_params.extend(list(module.parameters()))
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        original_h, original_w = noisy.shape[-2:]
        padded, _ = self._pad_to_patch_multiple(noisy)
        normalized = (padded - self.pixel_mean) / self.pixel_std

        stem = self.stem(padded)
        enc1 = self.enc1(F.avg_pool2d(stem, kernel_size=2))
        enc2 = self.enc2(F.avg_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.avg_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.avg_pool2d(enc3, kernel_size=2))

        if self.backbone_backend == "timm":
            features = tuple(
                self.backbone.forward_intermediates(
                    normalized,
                    indices=self.feature_layers,
                    intermediates_only=True,
                )
            )
        else:
            features = self.backbone.get_intermediate_layers(
                normalized,
                n=self.feature_layers,
                reshape=True,
                norm=True,
            )

        vit = self.vit_fuse(torch.cat(features, dim=1))
        x = self.bottleneck(torch.cat([vit, enc4], dim=1))
        x = self.up3(x, enc3)
        x = self.up2(x, enc2)
        x = self.up1(x, enc1)
        x = self.up0(x, stem)
        residual = self.head(x)
        denoised = padded + residual
        return denoised[:, :, :original_h, :original_w]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_dinov3_model(checkpoint: Path | str | None = None, device: str = "auto"):
    root = project_root()
    checkpoint_path = Path(checkpoint) if checkpoint is not None else (
        root / "artifacts" / "dinov3_vits16_sigma25" / "checkpoints" / "best.pt"
    )
    device_obj = resolve_device(device)
    cache_key = (str(checkpoint_path.resolve()), str(device_obj))

    if cache_key not in _DINOV3_MODELS:
        checkpoint_data = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        ckpt_args = checkpoint_data.get("args", {})
        model = DINOv3Denoiser(
            project_root=root,
            backbone_backend=ckpt_args.get("backbone_backend", "auto"),
            pretrained_backbone=False,
            pretrained_model_name=ckpt_args.get("pretrained_model_name", "vit_small_patch16_dinov3.lvd1689m"),
        ).to(device_obj)
        model.load_state_dict(checkpoint_data["model_state"], strict=True)
        model.eval()
        _DINOV3_MODELS[cache_key] = model

    return _DINOV3_MODELS[cache_key], device_obj


def dinov3_denoise(u_noisy: np.ndarray, checkpoint: Path | str | None = None, device: str = "auto") -> np.ndarray:
    model, device_obj = load_dinov3_model(checkpoint=checkpoint, device=device)
    is_gray = u_noisy.ndim == 2
    model_input = np.stack([u_noisy, u_noisy, u_noisy], axis=-1) if is_gray else u_noisy

    tensor = torch.from_numpy(np.transpose(model_input, (2, 0, 1))).unsqueeze(0).float().to(device_obj)
    with torch.no_grad():
        denoised = model(tensor).clamp(0.0, 1.0).cpu().numpy()[0]

    denoised = np.transpose(denoised, (1, 2, 0))
    if is_gray:
        return np.mean(denoised, axis=-1)
    return denoised
