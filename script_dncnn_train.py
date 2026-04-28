"""
File Name: script_dncnn_train.py
Author: Jiatong
Function: Trains DnCNN on the project denoising split using KAIR components.
Reference: DnCNN, https://doi.org/10.1109/TIP.2017.2662206; KAIR, https://github.com/cszn/KAIR.
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


def add_kair_to_path(project_root: Path) -> None:
    kair_root = project_root / "external" / "KAIR"
    src_root = project_root / "src"
    sys.path = [p for p in sys.path if p != str(src_root)]
    if str(kair_root) not in sys.path:
        sys.path.insert(0, str(kair_root))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train DnCNN model")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=project_root / "data" / "dncnn_split",
        help="Root directory",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=project_root / "artifacts" / "dncnn_color_sigma25",
        help="save checkpoints and logs.",
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
    parser.add_argument("--patch-size", type=int, default=40, help="Training crop size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--max-steps", type=int, default=3000, help="Number of optimization steps.")
    parser.add_argument("--val-interval", type=int, default=250, help="Validation interval in steps.")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint interval in steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=520, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto | cuda | cpu.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.transpose(array, (1, 2, 0))
    return array


def compute_psnr(clean: np.ndarray, denoised: np.ndarray) -> float:
    mse = np.mean((clean - denoised) ** 2)
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnr_values = []
    l1_values = []
    with torch.no_grad():
        for batch in loader:
            noisy = batch["L"].to(device)
            clean = batch["H"].to(device)
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


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    add_kair_to_path(project_root)

    from data.dataset_dncnn import DatasetDnCNN
    from models.network_dncnn import DnCNN

    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.experiment_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.experiment_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.experiment_root / "history.csv"

    train_opt = {
        "phase": "train",
        "n_channels": 3,
        "H_size": args.patch_size,
        "sigma": args.sigma,
        "sigma_test": args.sigma,
        "noise_type": args.noise_type,
        "amount": args.amount,
        "salt_vs_pepper": args.salt_vs_pepper,
        "dataroot_H": str(args.split_root / "train"),
    }
    val_opt = {
        "phase": "test",
        "n_channels": 3,
        "H_size": args.patch_size,
        "sigma": args.sigma,
        "sigma_test": args.sigma,
        "noise_type": args.noise_type,
        "amount": args.amount,
        "salt_vs_pepper": args.salt_vs_pepper,
        "dataroot_H": str(args.split_root / "val"),
    }

    train_dataset = DatasetDnCNN(train_opt)
    val_dataset = DatasetDnCNN(val_opt)
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
        num_workers=1,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=17, act_mode="BR").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_psnr = -float("inf")
    start_time = time.time()
    step = 0

    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_l1", "val_psnr", "elapsed_sec"])

        while step < args.max_steps:
            for batch in train_loader:
                step += 1
                noisy = batch["L"].to(device)
                clean = batch["H"].to(device)

                optimizer.zero_grad(set_to_none=True)
                denoised = model(noisy)
                loss = criterion(denoised, clean)
                loss.backward()
                optimizer.step()

                if step % args.val_interval == 0 or step == 1:
                    val_l1, val_psnr = evaluate(model, val_loader, device)
                    elapsed_sec = time.time() - start_time
                    writer.writerow([step, float(loss.item()), val_l1, val_psnr, elapsed_sec])
                    f.flush()
                    print(
                        f"step={step} train_loss={loss.item():.6f} "
                        f"val_l1={val_l1:.6f} val_psnr={val_psnr:.4f} elapsed={elapsed_sec:.1f}s"
                    )

                    if val_psnr > best_val_psnr:
                        best_val_psnr = val_psnr
                        torch.save(
                            {
                                "step": step,
                                "model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "best_val_psnr": best_val_psnr,
                                "args": serialized_args,
                            },
                            checkpoints_dir / "best.pt",
                        )

                if step % args.save_interval == 0 or step == args.max_steps:
                    torch.save(
                        {
                            "step": step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_val_psnr": best_val_psnr,
                            "args": serialized_args,
                        },
                        checkpoints_dir / f"step_{step:06d}.pt",
                    )
                    torch.save(
                        {
                            "step": step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_val_psnr": best_val_psnr,
                            "args": serialized_args,
                        },
                        checkpoints_dir / "latest.pt",
                    )

                if step >= args.max_steps:
                    break

    print(f"Training finished at step {step}.")
    print(f"Artifacts written to: {args.experiment_root}")


if __name__ == "__main__":
    main()
