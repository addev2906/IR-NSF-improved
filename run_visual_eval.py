import os
import shutil
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import build_model
from dataset import apply_degradation_impulse, apply_degradation_dn
from evaluate import psnr_tensor, ssim_tensor

# Helper to find first existing path
def resolve_path(candidates):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(candidates[0])

# Setup paths
artifact_dir = Path(r"C:\Users\arjun\.gemini\antigravity-ide\brain\740563c7-2734-436f-8b71-762673fd0eec")
output_dir = Path("results/impulse_noise")
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[visual_eval] Using device: {device}")

# 1. Load Baseline Model
print("[visual_eval] Loading Baseline model...")
baseline_model = build_model(n_feats=64, n_resblocks=16, legacy=True, device=device)
ckpt_base_path = resolve_path(["checkpoints/checkpoints_baseline/best_model.pt", "checkpoints_baseline/best_model.pt"])
ckpt_base = torch.load(ckpt_base_path, map_location="cpu")
baseline_model.load_state_dict(ckpt_base.get("model", ckpt_base))
baseline_model.eval()

# 2. Load Improved Model
print("[visual_eval] Loading Improved model...")
improved_model = build_model(n_feats=32, n_resblocks=9, legacy=False, device=device)
ckpt_imp_path = resolve_path(["checkpoints/checkpoints_improved/best_model.pt", "checkpoints_improved/best_model.pt"])
ckpt_imp = torch.load(ckpt_imp_path, map_location="cpu")
improved_model.load_state_dict(ckpt_imp.get("model", ckpt_imp))
improved_model.eval()

# Selected test images
test_specs = [
    {
        "name": "kodim01",
        "path": resolve_path(["sample_images/kodim01.png", "data/testsets/Kodak24/kodim01.png"]),
        "crop_center": (260, 390), # (y, x)
        "crop_size": 130,
        "desc": "Kodak #1: Cottage with Timber Walls & Stream",
    },
    {
        "name": "kodim03",
        "path": resolve_path(["sample_images/kodim03.png", "data/testsets/Kodak24/kodim03.png"]),
        "crop_center": (260, 260),
        "crop_size": 130,
        "desc": "Kodak #3: Island & Stone Wall Landscape",
    },
    {
        "name": "kodim05",
        "path": resolve_path(["sample_images/kodim05.png", "data/testsets/Kodak24/kodim05.png"]),
        "crop_center": (280, 220),
        "crop_size": 130,
        "desc": "Kodak #5: Motorcycle Engine & Wire Spokes",
    },
    {
        "name": "butterfly",
        "path": resolve_path(["sample_images/butterfly.png", "data/testsets/Set5/Set5_HR/butterfly.png"]),
        "crop_center": (130, 130),
        "crop_size": 100,
        "desc": "Set5 Butterfly: High-Frequency Wing Patterns",
    },
    {
        "name": "bird",
        "path": resolve_path(["sample_images/bird.png", "data/testsets/Set5/Set5_HR/bird.png"]),
        "crop_center": (150, 150),
        "crop_size": 100,
        "desc": "Set5 Bird: Fine Feather Texture & Eye",
    },
]

results = []

for spec in test_specs:
    fname = spec["name"]
    img_path = spec["path"]
    if not img_path.exists():
        print(f"Skipping {fname}, file not found: {img_path}")
        continue

    # Load image and convert to grayscale tensor
    pil_img = Image.open(img_path).convert("L")
    clean = TF.to_tensor(pil_img) # (1, H, W)
    H, W = clean.shape[1], clean.shape[2]

    # Deterministic noise for fair comparison
    torch.manual_seed(42)
    noisy, mask = apply_degradation_impulse(clean, impulse_ratio=0.20, add_base_noise=False)

    clean_dev = clean.to(device)
    noisy_dev = noisy.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_base_dev = baseline_model(noisy_dev).squeeze(0)
        pred_imp_dev = improved_model(noisy_dev).squeeze(0)

    # Compute metrics
    psnr_noisy = psnr_tensor(noisy_dev.squeeze(0), clean_dev)
    ssim_noisy = ssim_tensor(noisy_dev.squeeze(0), clean_dev)

    psnr_base = psnr_tensor(pred_base_dev, clean_dev)
    ssim_base = ssim_tensor(pred_base_dev, clean_dev)

    psnr_imp = psnr_tensor(pred_imp_dev, clean_dev)
    ssim_imp = ssim_tensor(pred_imp_dev, clean_dev)

    delta_psnr = psnr_imp - psnr_base
    delta_ssim = ssim_imp - ssim_base

    print(f"[{fname}] Noisy: {psnr_noisy:.2f}dB | Base: {psnr_base:.2f}dB / {ssim_base:.4f} | Imp: {psnr_imp:.2f}dB / {ssim_imp:.4f} | Gain: +{delta_psnr:.2f}dB")

    res_item = {
        "name": fname,
        "description": spec["desc"],
        "resolution": f"{W}x{H}",
        "psnr_noisy": round(psnr_noisy, 2),
        "ssim_noisy": round(ssim_noisy, 4),
        "psnr_base": round(psnr_base, 2),
        "ssim_base": round(ssim_base, 4),
        "psnr_imp": round(psnr_imp, 2),
        "ssim_imp": round(ssim_imp, 4),
        "delta_psnr": round(delta_psnr, 2),
        "delta_ssim": round(delta_ssim, 4),
    }
    results.append(res_item)

    # Convert to CPU numpy for visualization
    clean_np = clean.squeeze(0).cpu().numpy()
    noisy_np = noisy.squeeze(0).cpu().numpy()
    pred_base_np = pred_base_dev.squeeze(0).cpu().numpy()
    pred_imp_np = pred_imp_dev.squeeze(0).cpu().numpy()

    # --- Plot 1: Full-Frame 4-Panel Comparison ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=150)
    plt.subplots_adjust(wspace=0.03, left=0.02, right=0.98, top=0.88, bottom=0.04)

    panels = [
        (clean_np, "Clean Ground Truth\nReference", "black"),
        (noisy_np, f"Corrupted Input (20% Impulse)\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}", "#d9534f"),
        (pred_base_np, f"Baseline IR-NSF (Paper)\nPSNR: {psnr_base:.2f} dB | SSIM: {ssim_base:.4f}", "#f0ad4e"),
        (pred_imp_np, f"Proposed Improved Model (Ours)\nPSNR: {psnr_imp:.2f} dB | SSIM: {ssim_imp:.4f} (+{delta_psnr:.2f} dB)", "#5cb85c"),
    ]

    for ax, (img_data, title, col) in zip(axes, panels):
        ax.imshow(img_data, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=col)
        ax.axis("off")

    fig.suptitle(f"{spec['desc']} — Full Image Denoising Comparison", fontsize=14, fontweight="bold", y=0.98)

    comp_filename = f"eval_{fname}_comparison.png"
    comp_path = output_dir / comp_filename
    plt.savefig(comp_path, bbox_inches="tight")
    plt.close()

    # Copy to artifact dir if present
    if artifact_dir and artifact_dir.exists():
        shutil.copy(comp_path, artifact_dir / comp_filename)

    # --- Plot 2: Zoomed Detail Crop Comparison ---
    cy, cx = spec["crop_center"]
    cs = spec["crop_size"] // 2
    y1, y2 = max(0, cy - cs), min(H, cy + cs)
    x1, x2 = max(0, cx - cs), min(W, cx + cs)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    plt.subplots_adjust(wspace=0.03, left=0.02, right=0.98, top=0.85, bottom=0.04)

    zoom_panels = [
        (clean_np[y1:y2, x1:x2], "Clean Reference (Crop)", "black"),
        (noisy_np[y1:y2, x1:x2], "Corrupted (20% Noise)", "#d9534f"),
        (pred_base_np[y1:y2, x1:x2], "Baseline IR-NSF (Smudged / Residuals)", "#f0ad4e"),
        (pred_imp_np[y1:y2, x1:x2], "Proposed Model (Clean Edges & Structure)", "#5cb85c"),
    ]

    for ax, (img_data, title, col) in zip(axes, zoom_panels):
        ax.imshow(img_data, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8, color=col)
        ax.axis("off")

    fig.suptitle(f"{spec['desc']} — Zoomed Detail Crop ({y2-y1}x{x2-x1})", fontsize=13, fontweight="bold", y=0.98)

    zoom_filename = f"eval_{fname}_zoom.png"
    zoom_path = output_dir / zoom_filename
    plt.savefig(zoom_path, bbox_inches="tight")
    plt.close()

    if artifact_dir and artifact_dir.exists():
        shutil.copy(zoom_path, artifact_dir / zoom_filename)

# Save JSON metrics summary
metrics_file = output_dir / "eval_metrics_summary.json"
with open(metrics_file, "w") as f:
    json.dump(results, f, indent=2)

print("\n[visual_eval] Finished! Generated comparison images and saved to artifact dir and results_experiments.")
