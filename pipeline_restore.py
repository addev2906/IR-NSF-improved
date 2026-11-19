"""
pipeline_restore.py
-------------------
End-to-End Image Restoration Pipeline chaining:
1. Denoising (DN)       - Upgraded ResNet (GELU + SE Attention + Non-Local Block)
2. Deblurring (DB)      - Upgraded MIMO-UNet (Multi-Scale Coarse-to-Fine)
3. Super-Resolution (SR)- Upgraded EDSR (SE Attention + Sub-Pixel PixelShuffle Upsampling)

Supports executing individual stages or the full chained pipeline on compound degraded images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from model import build_model, build_mimo, build_edsr
from dataset import get_motion_kernels
from evaluate import psnr_tensor, ssim_tensor


def resolve_path(candidates):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(candidates[0])


class RestorationPipeline:
    """
    End-to-end composite image restoration pipeline.
    Chains Denoising -> Deblurring -> Super-Resolution.
    """

    def __init__(
        self,
        checkpoint_dn: Optional[str] = None,
        checkpoint_db: Optional[str] = None,
        checkpoint_sr: Optional[str] = None,
        sr_scale: int = 2,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if checkpoint_dn is None:
            checkpoint_dn = str(resolve_path(["checkpoints/checkpoints_improved/best_model.pt", "checkpoints_improved/best_model.pt"]))
        if checkpoint_db is None:
            checkpoint_db = str(resolve_path(["checkpoints/checkpoints_db/db_best_model.pt", "checkpoints_db/db_best_model.pt"]))
        if checkpoint_sr is None:
            checkpoint_sr = str(resolve_path(["checkpoints/checkpoints_sr/sr_best_model.pt", "checkpoints_sr/sr_best_model.pt"]))

        print(f"[pipeline] Initializing restoration pipeline on {self.device}...")

        # 1. Denoising model (ResNet: 32 feats, 9 resblocks, improved SE + NonLocal)
        self.model_dn = None
        if checkpoint_dn and Path(checkpoint_dn).exists():
            print(f"[pipeline] Loading Denoising model from {checkpoint_dn}...")
            self.model_dn = build_model(n_feats=32, n_resblocks=9, legacy=False, device=self.device)
            ckpt = torch.load(checkpoint_dn, map_location="cpu")
            self.model_dn.load_state_dict(ckpt.get("model", ckpt))
            self.model_dn.eval()
        else:
            print(f"[pipeline] NOTICE: No DN checkpoint found at {checkpoint_dn}")

        # 2. Deblurring model (MIMO-UNet: 16 feats, 4 resblocks)
        self.model_db = None
        if checkpoint_db and Path(checkpoint_db).exists():
            print(f"[pipeline] Loading Deblurring model from {checkpoint_db}...")
            self.model_db = build_mimo(n_feats=16, n_res=4, device=self.device)
            ckpt = torch.load(checkpoint_db, map_location="cpu")
            self.model_db.load_state_dict(ckpt.get("model", ckpt))
            self.model_db.eval()
        else:
            print(f"[pipeline] NOTICE: No DB checkpoint found at {checkpoint_db}")

        # 3. Super-Resolution model (EDSR: 32 feats, 8 resblocks, scale=sr_scale)
        self.model_sr = None
        self.sr_scale = sr_scale
        if checkpoint_sr and Path(checkpoint_sr).exists():
            print(f"[pipeline] Loading Super-Resolution (x{sr_scale}) model from {checkpoint_sr}...")
            self.model_sr = build_edsr(scale=sr_scale, n_feats=32, n_resblocks=8, device=self.device)
            ckpt = torch.load(checkpoint_sr, map_location="cpu")
            self.model_sr.load_state_dict(ckpt.get("model", ckpt))
            self.model_sr.eval()
        else:
            print(f"[pipeline] NOTICE: No SR checkpoint found at {checkpoint_sr}")

    @torch.no_grad()
    def denoise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise an input tensor in [0, 1]. Supports (1, H, W) or (3, H, W).
        For RGB, processes channels sequentially through the 1-channel trained model.
        """
        if self.model_dn is None:
            return x

        was_3d = False
        if x.dim() == 3:
            was_3d = True
            x = x.unsqueeze(0)  # (1, C, H, W)

        B, C, H, W = x.shape
        x_dev = x.to(self.device)

        if C == 1:
            out = self.model_dn(x_dev)
        else:
            # Process each color channel through the denoising network
            out_channels = []
            for c in range(C):
                ch_in = x_dev[:, c:c+1, :, :]
                ch_out = self.model_dn(ch_in)
                out_channels.append(ch_out)
            out = torch.cat(out_channels, dim=1)

        out = out.clamp(0.0, 1.0).cpu()
        return out.squeeze(0) if was_3d else out

    @torch.no_grad()
    def deblur_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deblur an input tensor in [0, 1] using MIMO-UNet. Supports (1, H, W) or (3, H, W).
        """
        if self.model_db is None:
            return x

        was_3d = False
        if x.dim() == 3:
            was_3d = True
            x = x.unsqueeze(0)  # (1, C, H, W)

        B, C, H, W = x.shape
        is_gray = (C == 1)
        if is_gray:
            x_in = x.repeat(1, 3, 1, 1).to(self.device)
        else:
            x_in = x.to(self.device)

        out_full, _, _ = self.model_db(x_in)
        out = out_full.clamp(0.0, 1.0)

        if is_gray:
            out = TF.rgb_to_grayscale(out)

        out = out.cpu()
        return out.squeeze(0) if was_3d else out

    @torch.no_grad()
    def super_resolve_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Super-resolve an input tensor in [0, 1] using EDSR. Supports (1, H, W) or (3, H, W).
        """
        if self.model_sr is None:
            return x

        was_3d = False
        if x.dim() == 3:
            was_3d = True
            x = x.unsqueeze(0)  # (1, C, H, W)

        B, C, H, W = x.shape
        is_gray = (C == 1)
        if is_gray:
            x_in = x.repeat(1, 3, 1, 1).to(self.device)
        else:
            x_in = x.to(self.device)

        out = self.model_sr(x_in).clamp(0.0, 1.0)

        if is_gray:
            out = TF.rgb_to_grayscale(out)

        out = out.cpu()
        return out.squeeze(0) if was_3d else out

    def restore_full_pipeline(
        self,
        img: torch.Tensor,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the restoration pipeline step-by-step.
        stages: list containing subset or all of ['dn', 'db', 'sr'] in order.
        """
        if stages is None:
            stages = ["dn", "db", "sr"]

        results = {"input": img}
        current = img

        for stage in stages:
            stage_lower = stage.lower()
            if stage_lower == "dn":
                t0 = time.time()
                current = self.denoise_tensor(current)
                dt = time.time() - t0
                results["dn"] = current
                print(f"  [stage: DN] Completed in {dt*1000:.1f}ms (Shape: {list(current.shape)})")
            elif stage_lower == "db":
                t0 = time.time()
                current = self.deblur_tensor(current)
                dt = time.time() - t0
                results["db"] = current
                print(f"  [stage: DB] Completed in {dt*1000:.1f}ms (Shape: {list(current.shape)})")
            elif stage_lower == "sr":
                t0 = time.time()
                current = self.super_resolve_tensor(current)
                dt = time.time() - t0
                results["sr"] = current
                print(f"  [stage: SR] Completed in {dt*1000:.1f}ms (Shape: {list(current.shape)})")

        results["final"] = current
        return results


def synthesize_compound_degradation(
    clean: torch.Tensor,
    blur_kernel_idx: int = 2,
    downscale_factor: int = 2,
    noise_ratio: float = 0.15,
) -> Tuple[torch.Tensor, str]:
    """
    Synthesize realistic compound degradation:
    Scene -> Motion Blur -> Downsampling -> Impulse/Sensor Noise
    """
    # 1. Blur using motion kernel
    kernels = get_motion_kernels()
    k_idx = blur_kernel_idx % len(kernels)
    kernel = kernels[k_idx]
    kh, kw = kernel.shape
    k_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)

    C, H, W = clean.shape
    blurry_channels = []
    pad_h, pad_w = kh // 2, kw // 2
    for c in range(C):
        ch = clean[c:c+1].unsqueeze(0)
        ch_pad = F.pad(ch, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        blurred = F.conv2d(ch_pad, k_tensor)
        blurry_channels.append(blurred.squeeze(0))
    blurry = torch.cat(blurry_channels, dim=0).clamp(0.0, 1.0)

    # 2. Downsample
    h_lr = H // downscale_factor
    w_lr = W // downscale_factor
    lr_blurry = TF.resize(
        blurry,
        size=[h_lr, w_lr],
        interpolation=TF.InterpolationMode.BICUBIC,
        antialias=True,
    ).clamp(0.0, 1.0)

    # 3. Add salt & pepper impulse noise
    corrupted = lr_blurry.clone()
    mask = torch.rand_like(corrupted[0:1]) < noise_ratio
    val = (torch.rand_like(corrupted[0:1]) < 0.5).float()
    for c in range(C):
        corrupted[c:c+1][mask] = val[mask]

    desc = f"Motion Blur (Kernel #{k_idx+1}) + x{downscale_factor} Downscale + {int(noise_ratio*100)}% Impulse Noise"
    return corrupted, desc


def run_pipeline_demo(
    image_paths: List[Path],
    out_dir: Path,
    artifact_dir: Optional[Path] = None,
    scale: int = 2,
) -> None:
    """
    Run pipeline demonstration on test images and save visualization.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = RestorationPipeline(
        checkpoint_dn="checkpoints_improved/best_model.pt",
        checkpoint_db="checkpoints_db/db_best_model.pt",
        checkpoint_sr="checkpoints_sr/sr_best_model.pt",
        sr_scale=scale,
    )

    metrics_log = []

    for img_path in image_paths:
        if not img_path.exists():
            continue

        fname = img_path.stem
        print(f"\n=======================================================")
        print(f"Processing Image: {fname} ({img_path.name})")
        print(f"=======================================================")

        # Load clean reference
        img_pil = Image.open(img_path).convert("RGB")
        clean_hr = TF.to_tensor(img_pil) # (3, H, W)
        H, W = clean_hr.shape[1], clean_hr.shape[2]
        # Crop dimensions so they divide cleanly by scale
        H_crop = (H // scale) * scale
        W_crop = (W // scale) * scale
        clean_hr = clean_hr[:, :H_crop, :W_crop]

        # 1. Synthesize compound degradation
        degraded, deg_desc = synthesize_compound_degradation(
            clean_hr,
            blur_kernel_idx=2, # diagonal 15x15 blur
            downscale_factor=scale,
            noise_ratio=0.15,
        )

        # 2. Run pipeline
        res = pipeline.restore_full_pipeline(degraded, stages=["dn", "db", "sr"])

        # Compute metrics
        # For intermediate stages (at LR scale), compare with clean downsampled
        clean_lr = TF.resize(clean_hr, size=[H_crop // scale, W_crop // scale], interpolation=TF.InterpolationMode.BICUBIC).clamp(0, 1)

        psnr_deg = psnr_tensor(degraded, clean_lr)
        ssim_deg = ssim_tensor(degraded, clean_lr)

        psnr_dn = psnr_tensor(res["dn"], clean_lr)
        ssim_dn = ssim_tensor(res["dn"], clean_lr)

        psnr_db = psnr_tensor(res["db"], clean_lr)
        ssim_db = ssim_tensor(res["db"], clean_lr)

        # For final SR, compare with full HR clean
        psnr_final = psnr_tensor(res["final"], clean_hr)
        ssim_final = ssim_tensor(res["final"], clean_hr)

        print(f"  [Metrics] Degraded Input (LR):   PSNR={psnr_deg:.2f} dB, SSIM={ssim_deg:.4f}")
        print(f"  [Metrics] After Stage 1 (DN):    PSNR={psnr_dn:.2f} dB, SSIM={ssim_dn:.4f} (+{psnr_dn - psnr_deg:.2f} dB)")
        print(f"  [Metrics] After Stage 2 (DB):    PSNR={psnr_db:.2f} dB, SSIM={ssim_db:.4f}")
        print(f"  [Metrics] After Stage 3 (SR-HR): PSNR={psnr_final:.2f} dB, SSIM={ssim_final:.4f}")

        metrics_log.append({
            "image": fname,
            "hr_size": f"{W_crop}x{H_crop}",
            "lr_size": f"{W_crop//scale}x{H_crop//scale}",
            "psnr_degraded_lr": round(psnr_deg, 2),
            "ssim_degraded_lr": round(ssim_deg, 4),
            "psnr_dn": round(psnr_dn, 2),
            "ssim_dn": round(ssim_dn, 4),
            "psnr_db": round(psnr_db, 2),
            "ssim_db": round(ssim_db, 4),
            "psnr_final_hr": round(psnr_final, 2),
            "ssim_final_hr": round(ssim_final, 4),
        })

        # --- Plot 5-Panel Full Progression ---
        # [Clean Reference | Compound Degraded | Stage 1: Denoised | Stage 2: Deblurred | Stage 3: Super-Resolved (Final)]
        fig, axes = plt.subplots(1, 5, figsize=(22, 5.0), dpi=150)
        plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.86, bottom=0.04)

        panels = [
            (clean_hr, f"1. Clean Ground Truth\nReference ({W_crop}x{H_crop})", "black"),
            (degraded, f"2. Compound Degraded\nBlur+Noise+x{scale} LR\n{psnr_deg:.2f} dB | {ssim_deg:.4f}", "#d9534f"),
            (res["dn"], f"3. Stage 1: Denoised\nResNet (SE+NonLocal)\n{psnr_dn:.2f} dB | {ssim_dn:.4f}", "#f0ad4e"),
            (res["db"], f"4. Stage 2: Deblurred\nMIMO-UNet\n{psnr_db:.2f} dB | {ssim_db:.4f}", "#0275d8"),
            (res["final"], f"5. Stage 3: Super-Resolved\nEDSR (x{scale} Upscaled HR)\n{psnr_final:.2f} dB | {ssim_final:.4f}", "#5cb85c"),
        ]

        for ax, (tensor_img, title, col) in zip(axes, panels):
            np_img = tensor_img.permute(1, 2, 0).cpu().numpy()
            ax.imshow(np_img)
            ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8, color=col)
            ax.axis("off")

        fig.suptitle(f"{fname.upper()} — Complete End-to-End Restoration Pipeline Progression", fontsize=14, fontweight="bold", y=0.98)

        out_img_name = f"pipeline_{fname}_progression.png"
        out_img_path = out_dir / out_img_name
        plt.savefig(out_img_path, bbox_inches="tight")
        plt.close()

        if artifact_dir and artifact_dir.exists():
            shutil.copy(out_img_path, artifact_dir / out_img_name)

    # Save metrics JSON
    with open(out_dir / "pipeline_metrics_summary.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\n[pipeline] Done! Visual outputs saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="End-to-End Restoration Pipeline")
    p.add_argument("--scale", type=int, default=2, help="Super-resolution scaling factor (default: 2)")
    p.add_argument("--out_dir", default="results/composite_pipeline")
    args = p.parse_args()

    artifact_dir = Path(r"C:\Users\arjun\.gemini\antigravity-ide\brain\740563c7-2734-436f-8b71-762673fd0eec")
    sample_images = [
        resolve_path(["sample_images/bird.png", "data/testsets/Set5/Set5_HR/bird.png"]),
        resolve_path(["sample_images/butterfly.png", "data/testsets/Set5/Set5_HR/butterfly.png"]),
        resolve_path(["sample_images/kodim03.png", "data/testsets/Kodak24/kodim03.png"]),
        resolve_path(["sample_images/kodim01.png", "data/testsets/Kodak24/kodim01.png"]),
    ]

    run_pipeline_demo(sample_images, Path(args.out_dir), artifact_dir=artifact_dir, scale=args.scale)
