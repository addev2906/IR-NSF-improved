"""
train_usr.py
------------
Training script for Unsupervised Stripe Removal (USR) -- Path B of the
IR-NSF paper (Liu et al., "Image Restoration Learning via Noisy Supervision
in Fourier Domain", IEEE TIP 2026), Section V.

USR training procedure (paper Section V, Fig. 14):
  1. Load a batch of stripe-noisy images {y_0, ..., y_{B-1}}.
  2. Split the batch into "i" half and "j" half.
  3. Run the CURRENT network (no gradient) on both halves:
         z_hat_i = f_theta(y_i)   [torch.no_grad()]
         z_hat_j = f_theta(y_j)   [torch.no_grad()]
  4. Estimate stripe noise from the j-half:
         n_hat_j = y_j - z_hat_j
  5. Synthesise a pseudo-degraded input (amplification factor eps > 1):
         x_tilde_i = z_hat_i + eps * n_hat_j     (clamped to [0, 1])
  6. Forward pass WITH gradient on the pseudo-input:
         pred = f_theta(x_tilde_i)
  7. Compute the Masked Fourier Loss over Omega (stripe-corrupted DFT positions):
         L_phi^USR = sum_{(k,l) in Omega} l_phi( F(pred)[k,l] - F(y_i)[k,l] )
  8. Backpropagate and update.

Inference destriping (paper Section V):
  Replace DFT coefficients of the original noisy image with those from
  f_theta(noisy) ONLY at Omega positions; keep all others unchanged.
  This preserves pixel-wise noise and image structure while removing stripes.

Usage:
    # 1. Download DIV2K (run once):
    python -c "from dataset import download_div2k; download_div2k()"

    # 2. Pre-crop patches (run once):
    python preprocess.py --src_dir data/DIV2K/DIV2K_train_HR \\
                         --out_dir data/DIV2K/patches_480

    # 3. Train USR:
    python train_usr.py --train_dir data/DIV2K/patches_480 \\
                        --val_dir   data/DIV2K/val_patches_480
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

from dataset import build_usr_dataloaders
from loss import USRLoss, masked_fourier_reconstruct
from model import build_model
from train import batch_psnr, batch_ssim     # reuse metric helpers


# ---------------------------------------------------------------------------
# USR training step (Path B, paper Section V)
# ---------------------------------------------------------------------------

def train_one_epoch_usr(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: USRLoss,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device:    torch.device,
    epoch:     int,
    eps:       float = 2.0,
    log_every: int   = 100,
) -> dict:
    """
    Run one full USR training epoch (Path B noise-swapping loop).

    Each batch of B images is split into two halves (i, j). The network
    is first run with no gradient on both halves to produce clean estimates
    and a residual stripe noise estimate from the j-half.  A pseudo-degraded
    input is synthesised and the network is trained with gradient to minimise
    the Masked Fourier Loss over the stripe-corrupted Fourier positions Omega.

    Args:
        model:     ResNet model under training.
        loader:    USR DataLoader (batch_size must be even).
        criterion: USRLoss instance.
        optimiser: Adam optimiser.
        scheduler: MultiStepLR scheduler (stepped every iteration).
        device:    Training device.
        epoch:     Current epoch number (for logging).
        eps:       Amplification factor epsilon > 1 (paper default: 2.0).
                   Compensates for potential underestimation of stripe noise.
        log_every: Print frequency (in batches).

    Returns:
        dict: {"train_loss", "train_psnr"} averaged over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches  = len(loader)
    t0         = time.time()
    half       = loader.batch_size // 2   # number of i-images per step

    for i, batch in enumerate(loader):
        # batch shape: (B, C, H, W) -- B must be even
        batch = batch.to(device, non_blocking=True)
        y_i = batch[:half]   # first half  -> "i" images
        y_j = batch[half:]   # second half -> "j" images

        # ------------------------------------------------------------------
        # Step 3: Frozen forward passes (no gradient)
        # Paper: "To stabilise training, gradient backpropagation is stopped
        #         during this step."
        # ------------------------------------------------------------------
        with torch.no_grad():
            z_hat_i = model(y_i)          # estimated clean from y_i
            z_hat_j = model(y_j)          # estimated clean from y_j

        # ------------------------------------------------------------------
        # Step 4: Estimate stripe noise from j-half
        # n_hat_j = y_j - z_hat_j
        # ------------------------------------------------------------------
        n_hat_j = y_j - z_hat_j          # (half, C, H, W)

        # ------------------------------------------------------------------
        # Step 5: Synthesise pseudo-degraded input
        # x_tilde_i = z_hat_i + eps * n_hat_j
        # eps > 1 compensates for underestimation of stripe noise amplitude.
        # ------------------------------------------------------------------
        x_tilde_i = (z_hat_i + eps * n_hat_j).clamp(0.0, 1.0)
        x_tilde_i = x_tilde_i.detach()   # stop any lingering grad paths

        # ------------------------------------------------------------------
        # Step 6: Gradient forward pass on pseudo-input
        # ------------------------------------------------------------------
        pred = model(x_tilde_i)           # (half, C, H, W)

        # ------------------------------------------------------------------
        # Step 7: Masked Fourier Loss over Omega
        # Target is y_i (original noisy image), NOT a clean reference.
        # ------------------------------------------------------------------
        loss, info = criterion(pred, y_i)

        # ------------------------------------------------------------------
        # Step 8: Backprop + update
        # ------------------------------------------------------------------
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()

        # PSNR proxy: compare network output on x_tilde_i against y_i
        psnr = batch_psnr(pred.detach(), y_i)
        total_loss += info["loss"]
        total_psnr += psnr

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:03d} [{i+1:4d}/{n_batches}] "
                  f"Loss={info['loss']:.5f}  PSNR~={psnr:.2f}dB  "
                  f"LR={lr:.2e}  Omega={info['omega_size']}  ({elapsed:.1f}s)")

    return {
        "train_loss": total_loss / n_batches,
        "train_psnr": total_psnr / n_batches,
    }


# ---------------------------------------------------------------------------
# Validation step for USR
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_usr(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   USRLoss,
    device:      torch.device,
    stripe_type: str = "vertical",
) -> dict:
    """
    Evaluate USR model.

    At validation time we apply inference-time destriping via
    masked_fourier_reconstruct and compute PSNR/SSIM of the destriped output
    against the original noisy input (proxy metric; true eval uses clean refs).

    Returns:
        dict: {"val_loss", "val_psnr", "val_ssim"}.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches  = len(loader)
    half       = loader.batch_size // 2

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        y_i   = batch[:half]
        y_j   = batch[half:]

        # Frozen passes for pseudo-pair synthesis (same as training, no grad)
        z_hat_i = model(y_i)
        z_hat_j = model(y_j)
        n_hat_j = y_j - z_hat_j
        x_tilde_i = (z_hat_i + 2.0 * n_hat_j).clamp(0.0, 1.0)

        pred = model(x_tilde_i)

        _, info = criterion(pred, y_i)
        total_loss += info["loss"]

        # Inference destriping: replace Omega coefficients of y_i with pred's
        destriped = masked_fourier_reconstruct(y_i, pred, stripe_type=stripe_type)
        total_psnr += batch_psnr(destriped, y_i)
        total_ssim += batch_ssim(destriped, y_i)

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
    print(f"[train_usr] Checkpoint -> {path}")


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
    print(f"[train_usr] Loaded checkpoint {path} (epoch {epoch})")
    return epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_usr(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[train_usr] Device: {device}")
    print(f"[train_usr] Stripe type: {args.stripe_type}  |  eps={args.eps}")

    # -- Data --
    train_loader, val_loader = build_usr_dataloaders(
        train_patch_dir=args.train_dir,
        val_patch_dir=args.val_dir or None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stripe_amplitude=args.stripe_amplitude,
        stripe_num_freq=args.stripe_num_freq,
    )

    # -- Model (same modified EDSR backbone as Path A, grayscale) --
    model = build_model(
        n_feats=args.n_feats,
        n_resblocks=args.n_resblocks,
        in_channels=1,   # grayscale
        device=device,
    )

    # -- Loss: Masked Fourier Loss over Omega (paper Eq. 13) --
    criterion = USRLoss(
        penalty=args.penalty,
        delta=args.delta,
        stripe_type=args.stripe_type,
    )
    print(f"[train_usr] Loss: USRLoss(penalty={args.penalty}, delta={args.delta}, "
          f"stripe_type={args.stripe_type})")

    # -- Optimiser: same EDSR defaults as Path A --
    optimiser = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # -- LR schedule: halved every decay_steps iterations --
    steps_per_epoch = len(train_loader)
    decay_epochs    = max(1, args.decay_steps // steps_per_epoch)
    milestones      = [decay_epochs * (k + 1) for k in range(10)]
    scheduler = MultiStepLR(optimiser, milestones=milestones, gamma=0.5)
    print(f"[train_usr] LR halves every {decay_epochs} epochs "
          f"(= {args.decay_steps} steps)")

    # -- Output dir --
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Optional resume --
    start_epoch = 0
    best_psnr   = 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch = load_checkpoint(Path(args.resume), model, optimiser)

    # -- CSV log --
    log_path   = out_dir / "training_log_usr.csv"
    log_fields = ["epoch", "lr", "train_loss", "train_psnr",
                  "val_loss", "val_psnr", "val_ssim"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()
    print(f"[train_usr] Log -> {log_path}")

    # =========================================================================
    # Training loop
    # =========================================================================
    print(f"[train_usr] Training for {args.epochs} epochs ...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} | "
              f"LR={scheduler.get_last_lr()[0]:.2e} ===")

        t_metrics = train_one_epoch_usr(
            model, train_loader, criterion, optimiser,
            scheduler, device, epoch,
            eps=args.eps,
            log_every=args.log_every,
        )

        v_metrics: dict = {}
        if val_loader:
            v_metrics = validate_usr(
                model, val_loader, criterion, device,
                stripe_type=args.stripe_type,
            )
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

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimiser": optimiser.state_dict()},
                out_dir / f"usr_checkpoint_epoch{epoch:04d}.pt",
            )

        # Best PSNR checkpoint
        cur_psnr = v_metrics.get("val_psnr", t_metrics["train_psnr"])
        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimiser": optimiser.state_dict(), "best_psnr": best_psnr},
                out_dir / "usr_best_model.pt",
            )
            print(f"  ** Best PSNR: {best_psnr:.2f} dB **")

    print(f"\n[train_usr] Done. Best PSNR: {best_psnr:.2f} dB | "
          f"Checkpoints: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IR-NSF Unsupervised Stripe Removal (USR) Training -- Path B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--train_dir",      required=True,
                   help="Dir of pre-cropped 480x480 grayscale training patches")
    g.add_argument("--val_dir",        default=None,
                   help="Dir of pre-cropped validation patches (optional)")
    g.add_argument("--batch_size",     type=int,   default=16,
                   help="Must be even (split into i/j halves per step)")
    g.add_argument("--num_workers",    type=int,   default=4)
    # Noise (stripe only for USR)
    g = p.add_argument_group("Stripe Noise")
    g.add_argument("--stripe_amplitude", type=float, default=0.05)
    g.add_argument("--stripe_num_freq",  type=int,   default=3)
    g.add_argument("--stripe_type",      default="vertical",
                   choices=["vertical", "both"],
                   help="Omega mask: 'vertical' -> k=0 row; 'both' -> k=0 or l=0")
    # USR-specific
    g = p.add_argument_group("USR")
    g.add_argument("--eps",  type=float, default=2.0,
                   help="Amplification factor eps > 1 for pseudo-pair synthesis "
                        "(paper uses eps > 1; default 2.0 from paper's GitHub)")
    # Model
    g = p.add_argument_group("Model (modified EDSR)")
    g.add_argument("--n_feats",     type=int,   default=64)
    g.add_argument("--n_resblocks", type=int,   default=16)
    # Loss
    g = p.add_argument_group("Loss")
    g.add_argument("--penalty", default="huber", choices=["huber", "l1"],
                   help="Separable penalty for USRLoss")
    g.add_argument("--delta",   type=float, default=1.0,
                   help="Huber threshold delta (paper: 1.0)")
    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",      type=int,   default=300)
    g.add_argument("--lr",          type=float, default=1e-4)
    g.add_argument("--decay_steps", type=int,   default=200_000,
                   help="LR halved every N iterations (EDSR default: 200k)")
    g.add_argument("--gpu",         type=int,   default=0)
    g.add_argument("--resume",      default=None)
    g.add_argument("--out_dir",     default="checkpoints_usr")
    g.add_argument("--save_every",  type=int,   default=10)
    g.add_argument("--log_every",   type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    train_usr(parse_args())
