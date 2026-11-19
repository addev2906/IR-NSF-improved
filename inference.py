"""
inference.py
------------
Executable inference script for Image Denoising using the Proposed Robust
Fourier-Domain Denoising Model (ResNet with SE Attention, GELU, and Non-Local block).

Key Features:
  - Supports single image inference or batch directory inference.
  - Supports synthetic noise addition (20% impulse, mixed degradation) or direct denoising.
  - Automatically reports PSNR and SSIM if a clean ground-truth reference is available.
  - Saves restored output image and optional side-by-side comparison figure.

Usage Examples:
    # 1. Run inference on default sample image (bird.png) with 20% impulse noise:
    python inference.py --input sample_images/bird.png --noise_type impulse --noise_ratio 0.20

    # 2. Denoise an already-corrupted image without adding extra noise:
    python inference.py --input my_noisy_image.png --noise_type none

    # 3. Run on butterfly with mixed noise (Gaussian + Poisson + Stripe + Impulse):
    python inference.py --input sample_images/butterfly.png --noise_type mixed

    # 4. Batch run over all sample images:
    python inference.py --input sample_images/ --output_dir outputs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from model import build_model
from dataset import apply_degradation_impulse, apply_degradation_dn
from evaluate import psnr_tensor, ssim_tensor


def find_checkpoint(user_path: Optional[str] = None) -> Path:
    """Find the best available checkpoint file."""
    candidates = [
        user_path,
        "checkpoints/checkpoints_improved/best_model.pt",
        "checkpoints_improved/best_model.pt",
        "checkpoints/best_model.pt",
        "../checkpoints_improved/best_model.pt",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        "Could not find model checkpoint. Please specify --checkpoint <path_to_best_model.pt>."
    )


def load_denoiser(
    checkpoint_path: Path,
    device: torch.device,
    n_feats: int = 32,
    n_resblocks: int = 9,
    legacy: bool = False,
) -> torch.nn.Module:
    """Instantiate and load weights for the denoising model."""
    print(f"[inference] Loading model from: {checkpoint_path}")
    model = build_model(
        n_feats=n_feats,
        n_resblocks=n_resblocks,
        in_channels=1,
        legacy=legacy,
        device=device,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def denoise_image_tensor(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Denoise a tensor in [0, 1].
    Supports (1, H, W) grayscale or (3, H, W) RGB (channel-wise inference).
    """
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)

    C, H, W = img_tensor.shape
    if C == 1:
        inp = img_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)
        out = model(inp).squeeze(0).cpu().clamp(0.0, 1.0)
        return out
    elif C == 3:
        # Process each RGB channel sequentially
        out_channels = []
        for c in range(3):
            inp = img_tensor[c:c+1].unsqueeze(0).to(device)  # (1, 1, H, W)
            out_c = model(inp).squeeze(0).cpu().clamp(0.0, 1.0)
            out_channels.append(out_c)
        return torch.cat(out_channels, dim=0)
    else:
        raise ValueError(f"Unsupported channel dimension: {C}")


def process_single_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    noise_type: str = "impulse",
    noise_ratio: float = 0.20,
    save_comparison: bool = True,
    color_mode: str = "gray",
) -> dict:
    """Process one input image through noise generation and model restoration."""
    pil_raw = Image.open(image_path)
    if color_mode == "gray":
        pil_img = pil_raw.convert("L")
    else:
        pil_img = pil_raw.convert("RGB")

    clean_tensor = TF.to_tensor(pil_img)  # [C, H, W] in [0, 1]

    # Synthesize noise if requested
    has_ground_truth = True
    if noise_type == "impulse":
        torch.manual_seed(42)
        if clean_tensor.shape[0] == 1:
            noisy_tensor, _ = apply_degradation_impulse(clean_tensor, impulse_ratio=noise_ratio)
        else:
            # Multi-channel impulse noise
            noisy_channels = []
            for c in range(clean_tensor.shape[0]):
                ch_noisy, _ = apply_degradation_impulse(clean_tensor[c:c+1], impulse_ratio=noise_ratio)
                noisy_channels.append(ch_noisy)
            noisy_tensor = torch.cat(noisy_channels, dim=0)
    elif noise_type == "mixed":
        torch.manual_seed(42)
        if clean_tensor.shape[0] == 1:
            noisy_tensor, _ = apply_degradation_dn(clean_tensor)
            # Add impulse component to make paper mixed degradation robust
            noisy_tensor, _ = apply_degradation_impulse(noisy_tensor, impulse_ratio=min(0.15, noise_ratio))
        else:
            noisy_channels = []
            for c in range(clean_tensor.shape[0]):
                ch_noisy, _ = apply_degradation_dn(clean_tensor[c:c+1])
                ch_noisy, _ = apply_degradation_impulse(ch_noisy, impulse_ratio=min(0.15, noise_ratio))
                noisy_channels.append(ch_noisy)
            noisy_tensor = torch.cat(noisy_channels, dim=0)
    elif noise_type == "none":
        noisy_tensor = clean_tensor
        has_ground_truth = False
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Run Model Inference
    t0 = time.time()
    restored_tensor = denoise_image_tensor(model, noisy_tensor, device)
    elapsed = time.time() - t0

    # Compute Metrics if ground truth is available
    psnr_noisy, ssim_noisy = 0.0, 0.0
    psnr_clean, ssim_clean = 0.0, 0.0
    if has_ground_truth:
        psnr_noisy = psnr_tensor(noisy_tensor, clean_tensor)
        ssim_noisy = ssim_tensor(noisy_tensor, clean_tensor)
        psnr_clean = psnr_tensor(restored_tensor, clean_tensor)
        ssim_clean = ssim_tensor(restored_tensor, clean_tensor)
        psnr_gain = psnr_clean - psnr_noisy
        ssim_gain = ssim_clean - ssim_noisy

    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save output restored image
    out_img_path = output_dir / f"{stem}_restored.png"
    TF.to_pil_image(restored_tensor).save(out_img_path)

    if has_ground_truth:
        noisy_img_path = output_dir / f"{stem}_noisy.png"
        TF.to_pil_image(noisy_tensor).save(noisy_img_path)

    # Save side-by-side comparison figure
    if save_comparison and has_ground_truth:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=150)
        plt.subplots_adjust(wspace=0.03, left=0.02, right=0.98, top=0.86, bottom=0.04)

        if clean_tensor.shape[0] == 1:
            cmap = "gray"
            clean_disp = clean_tensor.squeeze(0).numpy()
            noisy_disp = noisy_tensor.squeeze(0).numpy()
            rest_disp  = restored_tensor.squeeze(0).numpy()
        else:
            cmap = None
            clean_disp = clean_tensor.permute(1, 2, 0).numpy()
            noisy_disp = noisy_tensor.permute(1, 2, 0).numpy()
            rest_disp  = restored_tensor.permute(1, 2, 0).numpy()

        axes[0].imshow(clean_disp, cmap=cmap)
        axes[0].set_title("1. Clean Reference", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        axes[1].imshow(noisy_disp, cmap=cmap)
        axes[1].set_title(
            f"2. Corrupted Input ({noise_type})\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}",
            fontsize=11, fontweight="bold", pad=8, color="#d9534f"
        )
        axes[1].axis("off")

        axes[2].imshow(rest_disp, cmap=cmap)
        axes[2].set_title(
            f"3. Proposed Model Restoration\nPSNR: {psnr_clean:.2f} dB | SSIM: {ssim_clean:.4f} (+{psnr_gain:.2f} dB)",
            fontsize=11, fontweight="bold", pad=8, color="#5cb85c"
        )
        axes[2].axis("off")

        fig.suptitle(f"Denoising Evaluation: {image_path.name}", fontsize=13, fontweight="bold", y=0.98)
        comp_path = output_dir / f"{stem}_comparison.png"
        plt.savefig(comp_path, bbox_inches="tight")
        plt.close()

    print(f"\n[Result] Processed '{image_path.name}' in {elapsed:.3f}s:")
    if has_ground_truth:
        print(f"  - Corrupted Input:  PSNR = {psnr_noisy:6.2f} dB, SSIM = {ssim_noisy:.4f}")
        print(f"  - Restored Output:  PSNR = {psnr_clean:6.2f} dB, SSIM = {ssim_clean:.4f}")
        print(f"  - Absolute Gain:    +{psnr_gain:.2f} dB PSNR, +{ssim_gain:.4f} SSIM")
    print(f"  - Saved Restored:   {out_img_path}")
    if save_comparison and has_ground_truth:
        print(f"  - Saved Comparison: {comp_path}")

    return {
        "file": image_path.name,
        "psnr_noisy": round(psnr_noisy, 2),
        "ssim_noisy": round(ssim_noisy, 4),
        "psnr_clean": round(psnr_clean, 2),
        "ssim_clean": round(ssim_clean, 4),
        "time_sec": round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Inference for Proposed Image Denoising Model")
    parser.add_argument("--input", default="sample_images/bird.png",
                        help="Path to an image file or a directory containing images.")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory where restored results will be saved.")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit path to model checkpoint (.pt).")
    parser.add_argument("--noise_type", choices=["impulse", "mixed", "none"], default="impulse",
                        help="Type of noise to evaluate: 'impulse' (default), 'mixed', or 'none'.")
    parser.add_argument("--noise_ratio", type=float, default=0.20,
                        help="Impulse noise ratio (default: 0.20 for 20%% salt-and-pepper).")
    parser.add_argument("--color_mode", choices=["gray", "rgb"], default="gray",
                        help="Processing mode: 'gray' (default, native model domain) or 'rgb'.")
    parser.add_argument("--n_feats", type=int, default=32,
                        help="Channel width (default: 32 for improved model).")
    parser.add_argument("--n_resblocks", type=int, default=9,
                        help="Number of residual blocks (default: 9 for improved model).")
    parser.add_argument("--legacy_arch", action="store_true", default=False,
                        help="Set if loading baseline model (64 feats, 16 blocks, ReLU).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (-1 for CPU, 0 for cuda:0).")
    parser.add_argument("--no_comparison", action="store_true", default=False,
                        help="Disable generating side-by-side comparison image.")

    args = parser.parse_args()

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"[inference] Running on device: {device}")

    # Determine checkpoint
    ckpt_path = find_checkpoint(args.checkpoint)

    # Load model
    model = load_denoiser(
        checkpoint_path=ckpt_path,
        device=device,
        n_feats=args.n_feats,
        n_resblocks=args.n_resblocks,
        legacy=args.legacy_arch,
    )

    # Collect images
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted([
            p for p in input_path.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ])
        if not image_files:
            print(f"[inference] Error: No supported image files found in directory {input_path}")
            sys.exit(1)
    else:
        # Check in sample_images
        fallback = Path("sample_images") / input_path.name
        if fallback.is_file():
            image_files = [fallback]
        else:
            print(f"[inference] Error: Input path '{args.input}' does not exist.")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    print(f"[inference] Processing {len(image_files)} image(s)...")

    for img_file in image_files:
        process_single_image(
            image_path=img_file,
            model=model,
            device=device,
            output_dir=output_dir,
            noise_type=args.noise_type,
            noise_ratio=args.noise_ratio,
            save_comparison=not args.no_comparison,
            color_mode=args.color_mode,
        )

    print(f"\n[inference] All done! Restored images and comparisons saved to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
