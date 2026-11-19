"""
train.py
--------
Training loop for replicating the IR-NSF denoising baseline
(Liu et al., "Image Restoration Learning via Noisy Supervision in Fourier
Domain", IEEE TIP 2026).

Paper training settings (Section IV-A):
  - Network:     ResNet (modified EDSR, n_feats=64, n_resblocks=16)
  - Loss:        Fourier-domain separable Huber (delta=1) -- NO spatial loss
  - Optimizer:   Adam (default EDSR settings: lr=1e-4)
  - LR schedule: Halved every 200k iterations (EDSR default)
  - Data:        800 DIV2K images pre-cropped to 22988 patches of 480x480
  - Noise:       i.i.d. Gaussian sigma=5 + Poisson + Periodic stripe (on the fly)
  - Grayscale:   Yes (DN task)
  - Metrics:     PSNR and SSIM (all color channels; for grayscale: single channel)

Usage:
    # 1. Download DIV2K (run once):
    python -c "from dataset import download_div2k; download_div2k()"

    # 2. Pre-crop patches (run once):
    python preprocess.py --src_dir data/DIV2K/DIV2K_train_HR \
                         --out_dir data/DIV2K/patches_480

    # 3. Train:
    python train.py --train_dir data/DIV2K/patches_480 \
                    --val_dir   data/DIV2K/val_patches_480
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import build_dataloaders
from loss import IRNSFLoss, RobustImpulseIRNSFLoss
from model import build_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def batch_psnr(pred: torch.Tensor, target: torch.Tensor,
               max_val: float = 1.0) -> float:
    """
    Mean PSNR over a batch.

    PSNR = 10 * log10(max_val^2 / MSE)

    The paper computes PSNR for DN using all channels (here: 1 channel,
    grayscale). For SR it uses Y channel of YCbCr; that is handled separately.

    Args:
        pred   (torch.Tensor): Network output (B, C, H, W) in [0, 1].
        target (torch.Tensor): Reference image (B, C, H, W) in [0, 1].
        max_val (float): Maximum pixel value.

    Returns:
        float: Mean PSNR in dB.
    """
    with torch.no_grad():
        mse  = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        psnr = 10.0 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr.mean().item()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean SSIM over a batch using a simplified single-scale estimate.

    The paper reports SSIM alongside PSNR for all tasks.
    Full SSIM (with 11x11 Gaussian window) is computed here.

    Args:
        pred   (torch.Tensor): Network output  (B, C, H, W) in [0, 1].
        target (torch.Tensor): Reference image (B, C, H, W) in [0, 1].

    Returns:
        float: Mean SSIM.
    """
    with torch.no_grad():
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        # Gaussian kernel for local statistics
        k   = 11
        sig = 1.5
        coords = torch.arange(k, dtype=torch.float32, device=pred.device) - k // 2
        g  = torch.exp(-0.5 * (coords / sig) ** 2)
        g  = g / g.sum()
        kernel = torch.outer(g, g).unsqueeze(0).unsqueeze(0)   # (1,1,11,11)

        B, C, H, W = pred.shape
        ssim_vals = []
        for c in range(C):
            p = pred[:, c:c+1]      # (B,1,H,W)
            t = target[:, c:c+1]

            # Expand kernel for this batch
            ker = kernel.expand(1, 1, k, k)
            pad = k // 2

            mu1  = nn.functional.conv2d(p, ker, padding=pad)
            mu2  = nn.functional.conv2d(t, ker, padding=pad)
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2= mu1 * mu2

            sig1_sq = nn.functional.conv2d(p * p, ker, padding=pad) - mu1_sq
            sig2_sq = nn.functional.conv2d(t * t, ker, padding=pad) - mu2_sq
            sig12   = nn.functional.conv2d(p * t, ker, padding=pad) - mu1_mu2

            numer = (2 * mu1_mu2 + C1) * (2 * sig12   + C2)
            denom = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
            ssim_map = numer / (denom + 1e-8)
            ssim_vals.append(ssim_map.mean().item())

        return sum(ssim_vals) / len(ssim_vals)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
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
    Run one full training epoch.

    During training, the noisy input is fed to the network, and the loss is
    computed between the network output and the independently-drawn noisy
    target -- this is the IR-NSF training protocol (Eq. 12).

    Returns:
        dict: {"train_loss", "train_psnr"} averaged over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches  = len(loader)
    t0         = time.time()

    for i, (inp, tgt) in enumerate(loader):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # Forward: network denoises the input observation
        pred = model(inp)

        # Fourier-domain loss against the *noisy* target (IR-NSF Eq. 12)
        loss, info = criterion(pred, tgt)

        optimiser.zero_grad()
        loss.backward()
        # Gradient clipping (EDSR trains without explicit clipping, but it
        # helps stability when targets are noisy)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        # Paper uses step-based LR decay; step scheduler every iteration
        scheduler.step()

        # For PSNR logging we compare pred against the INPUT (not the clean
        # image, which we don't have), so this is a rough training metric.
        psnr = batch_psnr(pred.detach().clamp(0.0, 1.0), inp)
        total_loss += info["loss"]
        total_psnr += psnr

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:03d} [{i+1:4d}/{n_batches}] "
                  f"Loss={info['loss']:.5f}  PSNR~={psnr:.2f}dB  "
                  f"LR={lr:.2e}  ({elapsed:.1f}s)")

    return {
        "train_loss": total_loss / n_batches,
        "train_psnr": total_psnr / n_batches,
    }


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: IRNSFLoss,
    device:    torch.device,
) -> dict:
    """
    Evaluate the model.

    At validation time we measure PSNR/SSIM between the model's output and
    the *input* noisy image (since we have no clean reference in the
    noisy-target training setup). For true evaluation you would use a
    held-out clean test set (Set5, Set14, etc.) as the paper does.

    Returns:
        dict: {"val_loss", "val_psnr", "val_ssim"} averaged over the loader.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches  = len(loader)

    for inp, tgt in loader:
        inp  = inp.to(device, non_blocking=True)
        tgt  = tgt.to(device, non_blocking=True)
        pred = model(inp)

        _, info = criterion(pred, tgt)
        total_loss += info["loss"]
        total_psnr += batch_psnr(pred, inp)
        total_ssim += batch_ssim(pred, inp)

    return {
        "val_loss": total_loss / n_batches,
        "val_psnr": total_psnr / n_batches,
        "val_ssim": total_ssim / n_batches,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    torch.save(state, path)
    print(f"[train] Checkpoint -> {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimiser: Optional[torch.optim.Optimizer] = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimiser and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    epoch = ckpt.get("epoch", 0)
    print(f"[train] Loaded checkpoint {path} (epoch {epoch})")
    return epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    use_gpu = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_gpu else "cpu")
    print(f"[train] Device: {device}")

    # -- Data --
    train_loader, val_loader = build_dataloaders(
        train_patch_dir=args.train_dir,
        val_patch_dir=args.val_dir or None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gaussian_sigma=args.gaussian_sigma,
        stripe_amplitude=args.stripe_amplitude,
        stripe_num_freq=args.stripe_num_freq,
        noise_type=args.noise_type,
        impulse_ratio=args.impulse_ratio,
    )

    # -- Model (paper DN backbone: modified EDSR, grayscale) --
    model = build_model(
        n_feats=args.n_feats,
        n_resblocks=args.n_resblocks,
        in_channels=1,   # grayscale
        device=device,
    )

    # -- Loss --
    if args.loss_type == "robust_impulse":
        criterion = RobustImpulseIRNSFLoss(
            penalty="huber",
            delta=args.delta,
            lambda_spatial=args.lambda_spatial,
            impulse_threshold=args.impulse_threshold,
        )
        print(f"[train] Loss: RobustImpulseIRNSFLoss(penalty=huber, delta={args.delta}, "
              f"lambda_spatial={args.lambda_spatial}, threshold={args.impulse_threshold})")
    else:
        criterion = IRNSFLoss(penalty="huber", delta=args.delta)
        print(f"[train] Loss: IRNSFLoss(penalty=huber, delta={args.delta})")

    # -- Optimiser (EDSR default) --
    optimiser = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # -- LR scheduler: halve every `decay_steps` steps (EDSR default: 200k) --
    # MultiStepLR milestones in units of epochs estimated from steps:
    # milestones_epochs = [decay_steps / steps_per_epoch * k for k in 1,2,3,...]
    steps_per_epoch  = len(train_loader)
    decay_epochs     = max(1, args.decay_steps // steps_per_epoch)
    milestones       = [decay_epochs * (k + 1) for k in range(10)]
    scheduler = MultiStepLR(optimiser, milestones=milestones, gamma=0.5)
    print(f"[train] LR halves every {decay_epochs} epochs "
          f"(= {args.decay_steps} steps)")

    # -- Output dir --
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Best checkpoint tracking --
    start_epoch = 0
    best_loss = float("inf")
    best_psnr = 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch = load_checkpoint(Path(args.resume), model, optimiser)

    # -- CSV log --
    log_path = out_dir / "training_log.csv"
    log_fields = ["epoch", "lr", "train_loss", "train_psnr",
                  "val_loss", "val_psnr", "val_ssim"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()
    print(f"[train] Log -> {log_path}")

    # ==========================================================================
    # Training loop
    # ==========================================================================
    print(f"[train] Training for {args.epochs} epochs ...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} | "
              f"LR={scheduler.get_last_lr()[0]:.2e} ===")

        t_metrics = train_one_epoch(
            model, train_loader, criterion, optimiser,
            scheduler, device, epoch, log_every=args.log_every,
        )

        v_metrics: dict = {}
        if val_loader:
            v_metrics = validate(model, val_loader, criterion, device)
            print(f"  Val | Loss={v_metrics['val_loss']:.5f} "
                  f"PSNR~={v_metrics['val_psnr']:.2f}dB "
                  f"SSIM~={v_metrics['val_ssim']:.4f}")

        print(f"  Trn | Loss={t_metrics['train_loss']:.5f} "
              f"PSNR~={t_metrics['train_psnr']:.2f}dB")

        # Log
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        row.update(t_metrics)
        row.update(v_metrics)
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        # Checkpoint every N epochs
        if epoch % args.save_every == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimiser": optimiser.state_dict()},
                out_dir / f"checkpoint_epoch{epoch:04d}.pt",
            )

        # Always save latest model
        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(),
             "optimiser": optimiser.state_dict()},
            out_dir / "latest_model.pt",
        )

        # Best loss checkpoint
        cur_loss = v_metrics.get("val_loss", t_metrics["train_loss"])
        if cur_loss < best_loss:
            best_loss = cur_loss
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimiser": optimiser.state_dict(), "loss": best_loss},
                out_dir / "best_model.pt",
            )
            print(f"  ** Saved Best Model (Loss: {best_loss:.5f}) **")

    print(f"\n[train] Done. Best Loss: {best_loss:.5f} | Checkpoints: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IR-NSF Denoising Baseline Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--train_dir",      required=True,
                   help="Dir of pre-cropped 480x480 grayscale training patches")
    g.add_argument("--val_dir",        default=None,
                   help="Dir of pre-cropped validation patches (optional)")
    g.add_argument("--batch_size",     type=int,   default=16)
    g.add_argument("--num_workers",    type=int,   default=4)
    # Noise
    g = p.add_argument_group("Noise (paper DN mixture or Impulse)")
    g.add_argument("--noise_type",       choices=["mixed", "impulse", "combined"], default="mixed",
                   help="Degradation type: 'mixed' (paper DN), 'impulse' (salt & pepper), 'combined' (both)")
    g.add_argument("--impulse_ratio",    type=float, default=0.20,
                   help="Ratio of pixels corrupted with impulse noise (default: 0.20)")
    g.add_argument("--gaussian_sigma",   type=float, default=5.0,
                   help="Gaussian noise std in [0,255] scale (paper: 5)")
    g.add_argument("--stripe_amplitude", type=float, default=0.05)
    g.add_argument("--stripe_num_freq",  type=int,   default=3)
    # Model
    g = p.add_argument_group("Model (modified EDSR)")
    g.add_argument("--n_feats",      type=int,   default=32)
    g.add_argument("--n_resblocks",  type=int,   default=9)
    # Loss
    g = p.add_argument_group("Loss")
    g.add_argument("--loss_type",    choices=["irnsf", "robust_impulse"], default="irnsf",
                   help="Loss type: 'irnsf' (base paper) or 'robust_impulse' (proposed improvement)")
    g.add_argument("--delta",        type=float, default=1.0,
                   help="Huber threshold delta (paper: 1.0)")
    g.add_argument("--lambda_spatial", type=float, default=0.5,
                   help="Spatial loss weight for robust impulse loss (default: 0.5)")
    g.add_argument("--impulse_threshold", type=float, default=0.15,
                   help="Detection threshold for local median deviation (default: 0.15)")
    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       type=int,   default=30)
    g.add_argument("--lr",           type=float, default=1e-4)
    g.add_argument("--decay_steps",  type=int,   default=200_000,
                   help="LR halved every N iterations (EDSR default: 200k)")
    g.add_argument("--gpu",          type=int,   default=0)
    g.add_argument("--resume",       default=None)
    g.add_argument("--out_dir",      default="checkpoints_irnf")
    g.add_argument("--save_every",   type=int,   default=10)
    g.add_argument("--log_every",    type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
