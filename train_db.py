"""
train_db.py
-----------
Training script for image Deblurring (DB) -- paper §IV-A, IR-NSF framework.

Paper §IV-A (DB task):
  Network:   MIMO [28] (32 base feats, 8 res-blocks per enc/dec level).
  Loss:      Separable L1 in Fourier domain (IRNSFLoss, penalty='l1').
             Applied to the full-scale output; auxiliary scale outputs
             also penalised with the same loss (multi-scale supervision).
  Input:     Clean HR convolved with 1 of 8 Levin motion blur kernels [46]
             + Gaussian noise sigma=2.
  Target:    Clean HR + Gaussian noise sigma=10 (noisy target).
  Optimizer: Adam lr=1e-4, LR halved every 50 epochs (paper override for MIMO).
  Epochs:    300 (paper override for MIMO: "adjusted to 300 epochs").
  Data:      22988 DIV2K 480x480 patches (colour RGB).
  Metrics:   PSNR/SSIM on all RGB channels.

Usage:
    python train_db.py --train_dir data/DIV2K/patches_train \\
                       --val_dir   data/DIV2K/patches_val
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import build_db_dataloaders
from loss import IRNSFLoss
from model import build_mimo
from train import batch_psnr, batch_ssim   # reuse metric helpers


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch_db(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: IRNSFLoss,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device:    torch.device,
    epoch:     int,
    log_every: int = 100,
) -> dict:
    """
    One DB training epoch.

    MIMO produces three outputs (full, half, quarter scale). The IR-NSF loss
    is applied to all three against correspondingly downsampled targets,
    summed with equal weight (consistent with the MIMO paper's multi-scale loss).

    Returns:
        dict with 'train_loss' and 'train_psnr' (full-scale only) averaged over batches.
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches  = len(loader)
    t0         = time.time()

    for i, (blurry, noisy_sharp) in enumerate(loader):
        blurry      = blurry.to(device, non_blocking=True)       # (B, 3, H, W)
        noisy_sharp = noisy_sharp.to(device, non_blocking=True)  # (B, 3, H, W)

        # MIMO forward: three scale outputs
        out_full, out_half, out_qtr = model(blurry)

        # Multi-scale targets (downsample the noisy sharp target to match)
        import torch.nn.functional as F
        tgt_half = F.avg_pool2d(noisy_sharp, 2)
        tgt_qtr  = F.avg_pool2d(noisy_sharp, 4)

        # IR-NSF L1 Fourier loss at each scale, summed
        loss_full, info_full = criterion(out_full, noisy_sharp)
        loss_half, _         = criterion(out_half, tgt_half)
        loss_qtr,  _         = criterion(out_qtr,  tgt_qtr)
        loss = loss_full + loss_half + loss_qtr

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        psnr = batch_psnr(out_full.detach(), noisy_sharp)
        total_loss += info_full["loss"]
        total_psnr += psnr

        if (i + 1) % log_every == 0:
            print(f"  Epoch {epoch:03d} [{i+1:4d}/{n_batches}]  "
                  f"Loss={info_full['loss']:.5f}  PSNR~={psnr:.2f}dB  "
                  f"LR={scheduler.get_last_lr()[0]:.2e}  "
                  f"({time.time()-t0:.1f}s)")

    # Step scheduler once per epoch (LR halved every 50 epochs for MIMO)
    scheduler.step()

    return {"train_loss": total_loss / n_batches,
            "train_psnr": total_psnr / n_batches}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_db(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: IRNSFLoss,
    device:    torch.device,
) -> dict:
    """Validation: L1 Fourier loss + PSNR/SSIM on full-scale output."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches  = len(loader)

    for blurry, noisy_sharp in loader:
        blurry      = blurry.to(device, non_blocking=True)
        noisy_sharp = noisy_sharp.to(device, non_blocking=True)
        out_full, _, _ = model(blurry)
        _, info = criterion(out_full, noisy_sharp)
        total_loss += info["loss"]
        total_psnr += batch_psnr(out_full, noisy_sharp)
        total_ssim += batch_ssim(out_full, noisy_sharp)

    return {"val_loss": total_loss / n_batches,
            "val_psnr": total_psnr / n_batches,
            "val_ssim": total_ssim / n_batches}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    torch.save(state, path)
    print(f"[train_db] Checkpoint -> {path}")


def load_checkpoint(path: Path, model: nn.Module,
                    optimiser: Optional[torch.optim.Optimizer] = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimiser and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    epoch = ckpt.get("epoch", 0)
    print(f"[train_db] Loaded {path} (epoch {epoch})")
    return epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_db(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[train_db] Device: {device}")

    train_loader, val_loader = build_db_dataloaders(
        train_patch_dir=args.train_dir,
        val_patch_dir=args.val_dir or None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_sigma=args.input_sigma,
        target_sigma=args.target_sigma,
    )

    model = build_mimo(
        n_feats=args.n_feats,
        n_res=args.n_res,
        device=device,
    )

    # Paper uses L1 Fourier loss for DB
    criterion = IRNSFLoss(penalty="l1")
    print(f"[train_db] Loss: IRNSFLoss(penalty=l1, multi-scale)")

    optimiser = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Paper override for MIMO: "LR halved every 50 epochs" (epoch-based, not step-based)
    milestones = list(range(50, args.epochs + 1, 50))
    scheduler  = MultiStepLR(optimiser, milestones=milestones, gamma=0.5)
    print(f"[train_db] LR milestones (epoch-based, every 50): {milestones[:5]}...")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_psnr   = 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch = load_checkpoint(Path(args.resume), model, optimiser)

    log_path   = out_dir / "training_log_db.csv"
    log_fields = ["epoch", "lr", "train_loss", "train_psnr",
                  "val_loss", "val_psnr", "val_ssim"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    print(f"[train_db] Training {args.epochs} epochs | log -> {log_path}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} | LR={scheduler.get_last_lr()[0]:.2e} ===")

        t_metrics = train_one_epoch_db(
            model, train_loader, criterion, optimiser, scheduler,
            device, epoch, log_every=args.log_every,
        )

        v_metrics: dict = {}
        if val_loader:
            v_metrics = validate_db(model, val_loader, criterion, device)
            print(f"  Val  | Loss={v_metrics['val_loss']:.5f}  "
                  f"PSNR~={v_metrics['val_psnr']:.2f}dB  SSIM~={v_metrics['val_ssim']:.4f}")

        print(f"  Train| Loss={t_metrics['train_loss']:.5f}  PSNR~={t_metrics['train_psnr']:.2f}dB")

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        row.update(t_metrics)
        row.update(v_metrics)
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        if epoch % args.save_every == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(), "optimiser": optimiser.state_dict()},
                out_dir / f"db_checkpoint_epoch{epoch:04d}.pt",
            )

        cur_psnr = v_metrics.get("val_psnr", t_metrics["train_psnr"])
        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimiser": optimiser.state_dict(), "best_psnr": best_psnr},
                out_dir / "db_best_model.pt",
            )
            print(f"  ** Best PSNR: {best_psnr:.2f} dB **")

    print(f"\n[train_db] Done. Best PSNR: {best_psnr:.2f} dB | {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IR-NSF Deblurring Training (MIMO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Data")
    g.add_argument("--train_dir",     required=True)
    g.add_argument("--val_dir",       default=None)
    g.add_argument("--batch_size",    type=int,   default=16)
    g.add_argument("--num_workers",   type=int,   default=4)
    g.add_argument("--input_sigma",   type=float, default=2.0,
                   help="Gaussian sigma added to blurry input (paper: 2)")
    g.add_argument("--target_sigma",  type=float, default=10.0,
                   help="Gaussian sigma for noisy sharp target (paper: 10)")

    g = p.add_argument_group("Model (MIMO)")
    g.add_argument("--n_feats",       type=int,   default=32,
                   help="Base feature width (paper: 32)")
    g.add_argument("--n_res",         type=int,   default=8,
                   help="Residual blocks per enc/dec block (paper: 8)")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs",        type=int,   default=300,
                   help="Paper default for MIMO: 300")
    g.add_argument("--lr",            type=float, default=1e-4)
    g.add_argument("--gpu",           type=int,   default=0)
    g.add_argument("--resume",        default=None)
    g.add_argument("--out_dir",       default="checkpoints_db")
    g.add_argument("--save_every",    type=int,   default=10)
    g.add_argument("--log_every",     type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    train_db(parse_args())
