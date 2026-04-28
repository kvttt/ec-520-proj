"""
File Name: script_dinov3_train.py
Author: Jiatong
Function: Fine-tunes the DINOv3-based ViT denoiser
Reference: DINOv3, https://github.com/facebookresearch/dinov3; PyTorch, https://pytorch.org.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from script_dinov3_dataset import DenoiseDataset
from script_dinov3_denoiser import DINOv3Denoiser, add_external_paths


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Fine-tune a DINOv3-based ViT denoiser on the project split.")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=project_root / "data" / "dncnn_split",
        help="Root directory created by script_dncnn_dataset.py.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=project_root / "artifacts" / "dinov3_vits16_sigma25",
        help="Where to save checkpoints and logs.",
    )
    parser.add_argument("--sigma", type=int, default=25, help="AWGN sigma on the 0-255 scale.")
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=("gaussian", "salt_pepper"),
        default="gaussian",
        help="Synthetic corruption model used for train/validation pairs.",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=0.1,
        help="Salt-and-pepper corruption probability when --noise-type=salt_pepper.",
    )
    parser.add_argument(
        "--salt-vs-pepper",
        type=float,
        default=0.5,
        help="Salt probability among corrupted pixels when --noise-type=salt_pepper.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Training crop size. Kept as a multiple of the ViT patch size (16).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Per-step micro-batch size.")
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps. Effective batch size = batch-size * accum-steps.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--max-steps", type=int, default=3000, help="Number of optimizer steps.")
    parser.add_argument("--val-interval", type=int, default=250, help="Validation interval in steps.")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint interval in steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Backbone learning rate.")
    parser.add_argument("--seed", type=int, default=520, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto | cuda | cpu.",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Initialize the DINOv3 backbone randomly instead of loading the official pretrained weights.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="Optional local path or URL overriding the default official DINOv3 weights.",
    )
    parser.add_argument(
        "--pretrained-model-name",
        type=str,
        default="vit_small_patch16_dinov3.lvd1689m",
        help="Public timm DINOv3 model name used when the timm backend is available.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    return np.transpose(array, (1, 2, 0))


def compute_psnr(clean: np.ndarray, denoised: np.ndarray) -> float:
    mse = np.mean((clean - denoised) ** 2)
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnr_values = []
    l1_values = []
    with torch.no_grad():
        for batch in loader:
            noisy = batch["L"].to(device, non_blocking=True)
            clean = batch["H"].to(device, non_blocking=True)
            denoised = model(noisy)
            l1_values.append(torch.mean(torch.abs(denoised - clean)).item())
            for idx in range(denoised.shape[0]):
                psnr_values.append(
                    compute_psnr(
                        tensor_to_image(clean[idx]),
                        tensor_to_image(denoised[idx]),
                    )
                )
    model.train()
    return float(np.mean(l1_values)), float(np.mean(psnr_values))


def next_batch(loader: DataLoader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    add_external_paths(project_root)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.experiment_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.experiment_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.experiment_root / "history.csv"

    train_dataset = DenoiseDataset(
        root=args.split_root / "train",
        phase="train",
        patch_size=args.patch_size,
        sigma=args.sigma,
        noise_type=args.noise_type,
        amount=args.amount,
        salt_vs_pepper=args.salt_vs_pepper,
    )
    val_dataset = DenoiseDataset(
        root=args.split_root / "val",
        phase="test",
        patch_size=args.patch_size,
        sigma=args.sigma,
        noise_type=args.noise_type,
        amount=args.amount,
        salt_vs_pepper=args.salt_vs_pepper,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0 if args.num_workers == 0 else 1,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = DINOv3Denoiser(
        project_root=project_root,
        pretrained_backbone=not args.no_pretrained_backbone,
        backbone_weights=args.backbone_weights,
        pretrained_model_name=args.pretrained_model_name,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameter_groups(args.backbone_lr, args.lr))
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    serialized_args["effective_batch_size"] = args.batch_size * args.accum_steps
    serialized_args["backbone_backend"] = model.backbone_backend
    serialized_args["pretrained_model_name"] = model.pretrained_model_name

    best_val_psnr = -float("inf")
    step = 0
    start_time = time.time()
    train_iter = iter(train_loader)

    print(
        f"Using backbone backend={model.backbone_backend} "
        f"patch_size={model.patch_size} embed_dim={model.embed_dim}"
    )

    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_l1", "val_psnr", "elapsed_sec"])

        while step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for _ in range(args.accum_steps):
                batch, train_iter = next_batch(train_loader, train_iter)
                noisy = batch["L"].to(device, non_blocking=True)
                clean = batch["H"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    denoised = model(noisy)
                    raw_loss = criterion(denoised, clean)
                    loss = raw_loss / args.accum_steps

                scaler.scale(loss).backward()
                running_loss += float(raw_loss.item())

            scaler.step(optimizer)
            scaler.update()
            step += 1
            current_loss = running_loss / args.accum_steps

            if step % args.val_interval == 0 or step == 1:
                val_l1, val_psnr = evaluate(model, val_loader, device)
                elapsed_sec = time.time() - start_time
                writer.writerow([step, current_loss, val_l1, val_psnr, elapsed_sec])
                f.flush()
                print(
                    f"step={step} train_loss={current_loss:.6f} "
                    f"val_l1={val_l1:.6f} val_psnr={val_psnr:.4f} elapsed={elapsed_sec:.1f}s"
                )

                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    torch.save(
                        {
                            "step": step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "best_val_psnr": best_val_psnr,
                            "args": serialized_args,
                        },
                        checkpoints_dir / "best.pt",
                    )

            if step % args.save_interval == 0 or step == args.max_steps:
                state = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_val_psnr": best_val_psnr,
                    "args": serialized_args,
                }
                torch.save(state, checkpoints_dir / f"step_{step:06d}.pt")
                torch.save(state, checkpoints_dir / "latest.pt")

    print(f"Training finished at step {step}.")
    print(f"Artifacts written to: {args.experiment_root}")


if __name__ == "__main__":
    main()
