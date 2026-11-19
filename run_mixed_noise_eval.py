"""
run_mixed_noise_eval.py
-----------------------
Generates qualitative side-by-side visual comparisons and zoomed detail crops
for the mixed degradation:
Gaussian Noise (sigma=5) + Poisson Noise + Periodic Stripe Noise + Salt-and-Pepper Impulse Noise (15%)

Compares:
[1. Clean Reference | 2. Mixed Corrupted Input | 3. Baseline IR-NSF | 4. Proposed Improved Model]
"""

from pathlib import Path
import shutil
import json
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from model import build_model
from dataset import apply_degradation_impulse
from evaluate import psnr_tensor, ssim_tensor


def resolve_path(candidates):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(candidates[0])


def run_mixed_eval():
    artifact_dir = Path(r"C:\Users\arjun\.gemini\antigravity-ide\brain\740563c7-2734-436f-8b71-762673fd0eec")
    out_dir = Path("results/mixed_noise")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[mixed_eval] Running on {device}...")

    # 1. Load Baseline Model (64 feats, 16 blocks, legacy)
    baseline_model = build_model(n_feats=64, n_resblocks=16, legacy=True, device=device)
    ckpt_base_path = resolve_path(["checkpoints/checkpoints_baseline/best_model.pt", "checkpoints_baseline/best_model.pt"])
    ckpt_base = torch.load(ckpt_base_path, map_location="cpu")
    baseline_model.load_state_dict(ckpt_base.get("model", ckpt_base))
    baseline_model.eval()

    # 2. Load Improved Model (32 feats, 9 blocks, SE + NonLocal + GELU)
    improved_model = build_model(n_feats=32, n_resblocks=9, legacy=False, device=device)
    ckpt_imp_path = resolve_path(["checkpoints/checkpoints_improved/best_model.pt", "checkpoints_improved/best_model.pt"])
    ckpt_imp = torch.load(ckpt_imp_path, map_location="cpu")
    improved_model.load_state_dict(ckpt_imp.get("model", ckpt_imp))
    improved_model.eval()

    test_specs = [
        {
            "name": "bird",
            "path": resolve_path(["sample_images/bird.png", "data/testsets/Set5/Set5_HR/bird.png"]),
            "crop_center": (150, 150),
            "crop_size": 90,
            "desc": "Set5 Bird: Eye Contour & Feather Textures",
        },
        {
            "name": "butterfly",
            "path": resolve_path(["sample_images/butterfly.png", "data/testsets/Set5/Set5_HR/butterfly.png"]),
            "crop_center": (130, 130),
            "crop_size": 90,
            "desc": "Set5 Butterfly: High-Frequency Wing Patterns",
        },
        {
            "name": "kodim03",
            "path": resolve_path(["sample_images/kodim03.png", "data/testsets/Kodak24/kodim03.png"]),
            "crop_center": (260, 260),
            "crop_size": 130,
            "desc": "Kodak #3: Landscape, Cap Text & Stone Wall",
        },
        {
            "name": "kodim01",
            "path": resolve_path(["sample_images/kodim01.png", "data/testsets/Kodak24/kodim01.png"]),
            "crop_center": (260, 390),
            "crop_size": 130,
            "desc": "Kodak #1: Cottage Timber Walls & Windows",
        },
        {
            "name": "kodim05",
            "path": resolve_path(["sample_images/kodim05.png", "data/testsets/Kodak24/kodim05.png"]),
            "crop_center": (280, 220),
            "crop_size": 130,
            "desc": "Kodak #5: Motorcycle Engine & Wire Spokes",
        },
    ]

    metrics_list = []

    for spec in test_specs:
        fname = spec["name"]
        p = spec["path"]
        if not p.exists():
            print(f"Skipping {fname}, not found: {p}")
            continue

        pil_img = Image.open(p).convert("L")
        clean = TF.to_tensor(pil_img)  # (1, H, W)
        H, W = clean.shape[1], clean.shape[2]

        # Mixed degradation: Gaussian sigma=5 + Poisson + Periodic Stripe + 15% Salt & Pepper
        torch.manual_seed(42)
        noisy, _ = apply_degradation_impulse(
            clean,
            impulse_ratio=0.15,
            add_base_noise=True,
            gaussian_sigma=5.0,
            stripe_amplitude=0.05,
            stripe_num_freq=3,
        )

        clean_dev = clean.to(device)
        noisy_dev = noisy.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_base = baseline_model(noisy_dev).squeeze(0).clamp(0.0, 1.0)
            pred_imp = improved_model(noisy_dev).squeeze(0).clamp(0.0, 1.0)

        psnr_corrupt = psnr_tensor(noisy_dev.squeeze(0), clean_dev)
        ssim_corrupt = ssim_tensor(noisy_dev.squeeze(0), clean_dev)

        psnr_base = psnr_tensor(pred_base, clean_dev)
        ssim_base = ssim_tensor(pred_base, clean_dev)

        psnr_imp = psnr_tensor(pred_imp, clean_dev)
        ssim_imp = ssim_tensor(pred_imp, clean_dev)

        delta_psnr = psnr_imp - psnr_base
        delta_ssim = ssim_imp - ssim_base

        print(f"\n[{fname.upper()}] Mixed Degradation (Gaussian + Poisson + Stripe + Impulse):")
        print(f"  Corrupted Input:  PSNR = {psnr_corrupt:.2f} dB | SSIM = {ssim_corrupt:.4f}")
        print(f"  Baseline IR-NSF:  PSNR = {psnr_base:.2f} dB | SSIM = {ssim_base:.4f}")
        print(f"  Proposed Model:   PSNR = {psnr_imp:.2f} dB | SSIM = {ssim_imp:.4f} (+{delta_psnr:.2f} dB, +{delta_ssim:.4f} SSIM)")

        metrics_list.append({
            "image": fname,
            "desc": spec["desc"],
            "psnr_corrupt": round(psnr_corrupt, 2),
            "ssim_corrupt": round(ssim_corrupt, 4),
            "psnr_base": round(psnr_base, 2),
            "ssim_base": round(ssim_base, 4),
            "psnr_improved": round(psnr_imp, 2),
            "ssim_improved": round(ssim_imp, 4),
            "delta_psnr": round(delta_psnr, 2),
            "delta_ssim": round(delta_ssim, 4),
        })

        # --- 1. Full 4-Panel Comparison Figure ---
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), dpi=150)
        plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.03)

        panels = [
            (clean.squeeze(0).cpu().numpy(), "1. Clean Ground Truth\n(Reference Image)", "black"),
            (noisy.squeeze(0).cpu().numpy(), f"2. Mixed Corrupted Input\nGaussian+Poisson+Stripe+Impulse\n{psnr_corrupt:.2f} dB | {ssim_corrupt:.4f}", "#d9534f"),
            (pred_base.squeeze(0).cpu().numpy(), f"3. Baseline IR-NSF (Paper)\n16 Blocks, 64 Feats (1.22M params)\n{psnr_base:.2f} dB | {ssim_base:.4f}", "#0275d8"),
            (pred_imp.squeeze(0).cpu().numpy(), f"4. Proposed Improved Model\n9 Blocks, 32 Feats + SE + NonLocal\n{psnr_imp:.2f} dB | {ssim_imp:.4f} (+{delta_psnr:.2f} dB)", "#5cb85c"),
        ]

        for ax, (img_arr, title, col) in zip(axes, panels):
            ax.imshow(img_arr, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8, color=col)
            ax.axis("off")

        fig.suptitle(f"{fname.upper()} — Mixed Degradation Denoising Comparison (Gaussian + Poisson + Stripe + Impulse)",
                     fontsize=13, fontweight="bold", y=0.98)

        full_img_name = f"mixed_eval_{fname}_comparison.png"
        full_img_path = out_dir / full_img_name
        plt.savefig(full_img_path, bbox_inches="tight")
        plt.close()

        # --- 2. Zoom Crop Comparison Figure ---
        cy, cx = spec["crop_center"]
        cs = spec["crop_size"]
        y1, y2 = max(0, cy - cs // 2), min(H, cy + cs // 2)
        x1, x2 = max(0, cx - cs // 2), min(W, cx + cs // 2)

        crop_clean = clean.squeeze(0)[y1:y2, x1:x2].cpu().numpy()
        crop_noisy = noisy.squeeze(0)[y1:y2, x1:x2].cpu().numpy()
        crop_base = pred_base.squeeze(0)[y1:y2, x1:x2].cpu().numpy()
        crop_imp = pred_imp.squeeze(0)[y1:y2, x1:x2].cpu().numpy()

        fig_z, axes_z = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
        plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.86, bottom=0.04)

        z_panels = [
            (crop_clean, "1. Clean Reference (Zoom)", "black"),
            (crop_noisy, "2. Mixed Corrupted (Zoom)", "#d9534f"),
            (crop_base, f"3. Baseline IR-NSF (Zoom)\n{psnr_base:.2f} dB", "#0275d8"),
            (crop_imp, f"4. Proposed Model (Zoom)\n{psnr_imp:.2f} dB (+{delta_psnr:.2f} dB)", "#5cb85c"),
        ]

        for ax, (c_arr, title, col) in zip(axes_z, z_panels):
            ax.imshow(c_arr, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6, color=col)
            ax.axis("off")

        fig_z.suptitle(f"{fname.upper()} — Zoom Detail ({spec['desc']})", fontsize=13, fontweight="bold", y=0.98)

        zoom_img_name = f"mixed_eval_{fname}_zoom.png"
        zoom_img_path = out_dir / zoom_img_name
        plt.savefig(zoom_img_path, bbox_inches="tight")
        plt.close()

        # Copy to artifact dir for UI embedding
        if artifact_dir.exists():
            shutil.copy(full_img_path, artifact_dir / full_img_name)
            shutil.copy(zoom_img_path, artifact_dir / zoom_img_name)

    # Save summary JSON
    with open(out_dir / "mixed_eval_metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2)

    print(f"\n[mixed_eval] Done! Outputs saved to {out_dir} and copied to artifact directory.")


if __name__ == "__main__":
    run_mixed_eval()
