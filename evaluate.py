"""
evaluate.py
-----------
Evaluation script matching paper §IV-A metrics exactly.

Evaluates a trained IR-NSF model on the standard test sets:
    Set5, Set14, Kodak24, McMaster, Urban100, Manga109

Metric conventions per paper §IV-A:
    DN:  PSNR/SSIM on all colour channels (here: single grayscale channel).
    SR:  PSNR/SSIM on Y channel of YCbCr; border pixels cropped by scale factor.
    DB:  PSNR/SSIM on all RGB channels.

Usage:
    # Evaluate DN model (ResNet):
    python evaluate.py --task dn --checkpoint checkpoints_irnf/best_model.pt

    # Evaluate SR model (EDSR x4):
    python evaluate.py --task sr --checkpoint checkpoints_sr/sr_best_model.pt --scale 4

    # Evaluate DB model (MIMO):
    python evaluate.py --task db --checkpoint checkpoints_db/db_best_model.pt

    # List available test sets without running:
    python evaluate.py --list_sets
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
import tarfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

from model import build_model, build_edsr, build_mimo
from dataset import (
    apply_degradation_dn,
    apply_degradation_impulse,
    apply_degradation_sr,
    apply_degradation_db,
    get_motion_kernels,
)
try:
    from train_sr import rgb_to_y
except ImportError:
    def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
        """Convert RGB image tensor to Y channel of YCbCr in [0, 255]."""
        weights = torch.tensor([65.481, 128.553, 24.966], dtype=img.dtype, device=img.device).view(-1, 1, 1)
        if img.dim() == 4:
            weights = weights.unsqueeze(0)
        return (img * weights).sum(dim=-3, keepdim=True) + 16.0


# ---------------------------------------------------------------------------
# Test set registry (name -> download URL -> local folder)
# ---------------------------------------------------------------------------

# All datasets land in data/testsets/<name>/
TEST_SET_REGISTRY: Dict[str, dict] = {
    "Set5": {
        "url": "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz",
        "subdir": "Set5",
        "ext": ".png",
    },
    "Set14": {
        "url": "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz",
        "subdir": "Set14",
        "ext": ".png",
    },
    "Kodak24": {
        "url": "http://r0k.us/graphics/kodak/",   # individual images — handled specially
        "subdir": "Kodak24",
        "ext": ".png",
        "special": "kodak",
    },
    "McMaster": {
        "url": "https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm",
        "subdir": "McMaster",
        "ext": ".tif",
        "special": "manual",   # must be downloaded manually
    },
    "Urban100": {
        "url": "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz",
        "subdir": "Urban100",
        "ext": ".png",
    },
    "Manga109": {
        "url": "http://www.manga109.org/en/download.html",
        "subdir": "Manga109",
        "ext": ".png",
        "special": "manual",   # requires registration
    },
}

# Mirror URLs for datasets that are reliably available.
# SelfExSR GitHub zips and the old Illinois Box links now 404; Hugging Face
# hosts the HR images as tar.gz archives.
_MIRRORS = {
    "Set5": [
        "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set5.zip",
    ],
    "Set14": [
        "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14.zip",
    ],
    "Urban100": [
        "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Urban100.zip",
    ],
}


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    d = Path("data/testsets")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_archive(url: str, dest_dir: Path, timeout: int = 180) -> bool:
    """Download a zip or tar.gz and extract into dest_dir. Returns True on success."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  Downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "image-denoising-eval"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        lower = url.lower().split("?", 1)[0]
        if lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(dest_dir)
        elif lower.endswith(".tar.gz") or lower.endswith(".tgz") or lower.endswith(".tar"):
            mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as t:
                try:
                    t.extractall(dest_dir, filter="data")
                except TypeError:
                    t.extractall(dest_dir)
        else:
            raise ValueError(f"Unsupported archive type: {url}")
        return True
    except Exception as e:
        print(f"  [WARN] Failed: {e}")
        return False


def _download_kodak(dest_dir: Path) -> None:
    """Download 24 Kodak images individually from r0k.us."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = "http://r0k.us/graphics/kodak/kodak/"
    for i in range(1, 25):
        fname = f"kodim{i:02d}.png"
        fpath = dest_dir / fname
        if fpath.exists():
            continue
        try:
            print(f"  Downloading Kodak {fname} ...")
            urllib.request.urlretrieve(base + fname, fpath)
        except Exception as e:
            print(f"  [WARN] Could not download {fname}: {e}")


def download_testsets(root: Optional[str] = None) -> None:
    """
    Download all automatically-available test sets.

    Skips McMaster and Manga109 (require manual download / registration).
    Kodak24 is downloaded image by image from r0k.us.
    Set5, Set14, Urban100 are downloaded from GitHub mirrors.
    """
    base = Path(root) if root else _data_dir()
    base.mkdir(parents=True, exist_ok=True)

    for name, meta in TEST_SET_REGISTRY.items():
        dest = base / meta["subdir"]
        if dest.exists() and any(dest.iterdir()):
            print(f"  [{name}] already exists, skipping.")
            continue

        special = meta.get("special")
        if special == "manual":
            print(f"  [{name}] requires manual download. See: {meta['url']}")
            continue
        if special == "kodak":
            print(f"  [{name}] downloading individually from r0k.us ...")
            _download_kodak(dest)
            continue

        # Try Hugging Face / other mirrors first, then the registry URL
        mirrors = _MIRRORS.get(name, [])
        if isinstance(mirrors, str):
            mirrors = [mirrors]
        urls = list(mirrors) + [meta["url"]]
        for url in urls:
            if _download_archive(url, dest):
                print(f"  [{name}] downloaded OK.")
                break
        else:
            print(f"  [{name}] FAILED. Download manually from {meta['url']}")


def load_testset(name: str, root: Optional[str] = None) -> List[torch.Tensor]:
    """
    Load all images from a test set as float tensors in [0, 1].

    Args:
        name: Test set name (key in TEST_SET_REGISTRY).
        root: Root directory for test sets. Defaults to data/testsets/.

    Returns:
        List of (C, H, W) tensors.
    """
    base = Path(root) if root else _data_dir()
    meta = TEST_SET_REGISTRY[name]
    folder = base / meta["subdir"]

    if not folder.exists():
        raise FileNotFoundError(
            f"Test set '{name}' not found at {folder}. "
            f"Run: python evaluate.py --download"
        )

    ext = meta["ext"]
    paths = sorted(folder.rglob(f"*{ext}"))
    if not paths:
        # Also check .bmp, .jpg
        for alt in [".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]:
            paths = sorted(folder.rglob(f"*{alt}"))
            if paths:
                break
    # Hugging Face SR archives also ship bicubic LR folders; keep HR only.
    paths = [
        p for p in paths
        if not any(token in str(p).lower() for token in ("_lr_", "/lr/", "\\lr\\", "lr_x"))
    ]

    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    images = []
    for p in paths:
        img = Image.open(p)
        images.append(TF.to_tensor(img))
    return images


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two (C, H, W) tensors in [0, 1], [0, 255] MSE scale."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim_tensor(pred: torch.Tensor, target: torch.Tensor,
                window_size: int = 11) -> float:
    """
    SSIM between two (C, H, W) tensors.
    Implements the standard Wang et al. SSIM formula.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pred   = pred.unsqueeze(0)    # (1, C, H, W)
    target = target.unsqueeze(0)

    # Gaussian window
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g)

    win   = gaussian_window(window_size).to(pred.device)
    C_ch  = pred.shape[1]
    win4d = win.unsqueeze(0).unsqueeze(0).expand(C_ch, 1, -1, -1)
    pad   = window_size // 2

    mu1  = F.conv2d(pred,   win4d, padding=pad, groups=C_ch)
    mu2  = F.conv2d(target, win4d, padding=pad, groups=C_ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1   = F.conv2d(pred   * pred,   win4d, padding=pad, groups=C_ch) - mu1_sq
    s2   = F.conv2d(target * target, win4d, padding=pad, groups=C_ch) - mu2_sq
    s12  = F.conv2d(pred   * target, win4d, padding=pad, groups=C_ch) - mu1_mu2

    num  = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den  = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return (num / den).mean().item()


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dn(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    n_noise_realizations: int = 5,
    noise_type: str = "mixed",
    impulse_ratio: float = 0.20,
) -> Tuple[float, float]:
    """
    Evaluate DN model. Runs n_noise_realizations of the noise model and
    averages PSNR/SSIM.

    Args:
        model:  Trained ResNet for DN.
        images: List of clean (1 or 3, H, W) tensors.
        device: Evaluation device.
        noise_type: 'mixed', 'impulse', or 'combined'.
        impulse_ratio: ratio of impulse corrupted pixels.

    Returns:
        (mean_psnr, mean_ssim).
    """
    model.eval()
    all_psnr, all_ssim = [], []

    for img in images:
        # Convert to grayscale if needed
        if img.shape[0] == 3:
            img = TF.rgb_to_grayscale(img)  # (1, H, W)

        img_gpu = img.to(device)
        psnr_runs, ssim_runs = [], []
        for _ in range(n_noise_realizations):
            if noise_type == "impulse":
                noisy_in, _ = apply_degradation_impulse(img, impulse_ratio=impulse_ratio, add_base_noise=False)
            elif noise_type == "combined":
                noisy_in, _ = apply_degradation_impulse(img, impulse_ratio=impulse_ratio, add_base_noise=True)
            else:
                noisy_in, _ = apply_degradation_dn(img)

            noisy_in    = noisy_in.unsqueeze(0).to(device)  # (1, 1, H, W)
            pred        = model(noisy_in).squeeze(0)        # (1, H, W)
            psnr_runs.append(psnr_tensor(pred, img_gpu))
            ssim_runs.append(ssim_tensor(pred, img_gpu))

        cur_p = sum(psnr_runs) / len(psnr_runs)
        cur_s = sum(ssim_runs) / len(ssim_runs)
        all_psnr.append(cur_p)
        all_ssim.append(cur_s)
        print(f"    img {len(all_psnr)}/{len(images)}: {cur_p:.2f} dB, {cur_s:.4f}", flush=True)

    return (sum(all_psnr) / len(all_psnr),
            sum(all_ssim) / len(all_ssim))


@torch.no_grad()
def evaluate_sr(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    scale: int,
) -> Tuple[float, float]:
    """
    Evaluate SR model. Metric: PSNR/SSIM on Y channel of YCbCr with border crop.
    Paper §IV-A: "an equal number of border pixels is ignored corresponding
    to the scaling factor."
    """
    model.eval()
    all_psnr, all_ssim = [], []

    for img in images:
        # Convert to RGB if grayscale
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # Ensure dimensions are divisible by scale
        C, H, W = img.shape
        H_crop  = (H // scale) * scale
        W_crop  = (W // scale) * scale
        hr_clean = img[:, :H_crop, :W_crop]

        lr, _ = apply_degradation_sr(hr_clean, scale=scale)
        lr    = lr.unsqueeze(0).to(device)

        sr_pred = model(lr).squeeze(0).cpu()

        # Crop borders
        b = scale
        pred_crop = sr_pred  [:, b:-b, b:-b]
        gt_crop   = hr_clean[:, b:-b, b:-b]

        # Y channel [0, 255]
        pred_y = rgb_to_y(pred_crop.unsqueeze(0)).squeeze(0).clamp(0, 255)
        gt_y   = rgb_to_y(gt_crop.unsqueeze(0)).squeeze(0).clamp(0, 255)

        # Normalise to [0,1] for psnr_tensor which uses [0,1] scale
        all_psnr.append(psnr_tensor(pred_y / 255.0, gt_y / 255.0))
        all_ssim.append(ssim_tensor(pred_y / 255.0, gt_y / 255.0))

    return (sum(all_psnr) / len(all_psnr),
            sum(all_ssim) / len(all_ssim))


@torch.no_grad()
def evaluate_db(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    n_kernels: int = 8,
) -> Tuple[float, float]:
    """
    Evaluate DB model. Metric: PSNR/SSIM on all RGB channels.
    Averages over all 8 motion blur kernels (consistent with paper's
    evaluation over all 8 Levin kernels).
    """
    model.eval()
    kernels  = get_motion_kernels()
    all_psnr, all_ssim = [], []

    for img in images:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # Ensure divisible by 4 (MIMO requirement)
        C, H, W   = img.shape
        H_crop    = (H // 4) * 4
        W_crop    = (W // 4) * 4
        img_crop  = img[:, :H_crop, :W_crop]

        psnr_k, ssim_k = [], []
        for kernel in kernels[:n_kernels]:
            blurry, _ = apply_degradation_db(img_crop, kernel=kernel)
            blurry    = blurry.unsqueeze(0).to(device)

            out_full, _, _ = model(blurry)
            pred = out_full.squeeze(0).cpu()

            psnr_k.append(psnr_tensor(pred, img_crop))
            ssim_k.append(ssim_tensor(pred, img_crop))

        all_psnr.append(sum(psnr_k) / len(psnr_k))
        all_ssim.append(sum(ssim_k) / len(ssim_k))

    return (sum(all_psnr) / len(all_psnr),
            sum(all_ssim) / len(all_ssim))


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(
    task: str,
    results: Dict[str, Tuple[float, float]],
    scale: Optional[int] = None,
) -> None:
    """Print results in a format matching paper Tables I-III."""
    task_label = task.upper()
    if task == "sr" and scale:
        task_label += f" x{scale}"

    print(f"\n{'='*62}")
    print(f"  IR-NSF {task_label} Evaluation Results")
    print(f"{'='*62}")
    print(f"  {'Dataset':<14}  {'PSNR (dB)':>10}  {'SSIM':>8}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*8}")

    all_psnr, all_ssim = [], []
    for name, (psnr, ssim) in results.items():
        print(f"  {name:<14}  {psnr:>10.2f}  {ssim:>8.4f}")
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    print(f"  {'-'*14}  {'-'*10}  {'-'*8}")
    print(f"  {'Average':<14}  {sum(all_psnr)/len(all_psnr):>10.2f}  "
          f"{sum(all_ssim)/len(all_ssim):>8.4f}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    if args.download:
        print("Downloading test sets ...")
        download_testsets(args.testset_root)
        return

    if args.list_sets:
        print("Available test sets:")
        for name, meta in TEST_SET_REGISTRY.items():
            note = f" [{meta.get('special', '')}]" if meta.get("special") else ""
            print(f"  {name:<12}  {meta['url']}{note}")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Task: {args.task.upper()}  Device: {device}")

    # Load model
    if args.task == "dn":
        model = build_model(n_feats=args.n_feats, n_resblocks=args.n_resblocks,
                            in_channels=1, legacy=args.legacy_arch, device=device)
    elif args.task == "sr":
        model = build_edsr(scale=args.scale, n_feats=args.n_feats,
                           n_resblocks=args.n_resblocks, device=device)
    elif args.task == "db":
        model = build_mimo(n_feats=args.n_feats_mimo, n_res=args.n_res, device=device)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"[evaluate] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[evaluate] WARNING: No checkpoint provided, using random weights.")

    model.eval()

    # Determine which test sets to evaluate
    names = args.datasets if args.datasets else list(TEST_SET_REGISTRY.keys())

    results = {}
    for name in names:
        try:
            images = load_testset(name, root=args.testset_root)
            print(f"  [{name}] {len(images)} images ...")

            if args.task == "dn":
                psnr, ssim = evaluate_dn(
                    model,
                    images,
                    device,
                    n_noise_realizations=getattr(args, "n_runs", 1),
                    noise_type=getattr(args, "noise_type", "mixed"),
                    impulse_ratio=getattr(args, "impulse_ratio", 0.20),
                )
            elif args.task == "sr":
                psnr, ssim = evaluate_sr(model, images, device, scale=args.scale)
            elif args.task == "db":
                psnr, ssim = evaluate_db(model, images, device)

            results[name] = (psnr, ssim)
            print(f"  [{name}] PSNR={psnr:.2f} dB  SSIM={ssim:.4f}")

        except FileNotFoundError as e:
            print(f"  [{name}] SKIPPED -- {e}")

    if results:
        print_results_table(args.task, results, scale=getattr(args, "scale", None))
    else:
        print("\n[evaluate] No test sets found. Run: python evaluate.py --download")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IR-NSF Evaluation on Standard Test Sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task",         choices=["dn", "sr", "db"], default="dn")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--datasets",     nargs="+", default=None,
                   help="Test sets to evaluate (default: all available). "
                        "Options: Set5 Set14 Kodak24 McMaster Urban100 Manga109")
    p.add_argument("--testset_root", default="data/testsets",
                   help="Root directory where test sets are stored")
    p.add_argument("--download",     action="store_true",
                   help="Download test sets and exit")
    p.add_argument("--list_sets",    action="store_true",
                   help="List available test sets and exit")
    p.add_argument("--gpu",          type=int, default=0)

    g = p.add_argument_group("DN options")
    g.add_argument("--noise_type",   choices=["mixed", "impulse", "combined"], default="mixed",
                   help="Noise type to evaluate: 'mixed' (paper DN), 'impulse' (salt & pepper), 'combined'")
    g.add_argument("--impulse_ratio", type=float, default=0.20,
                   help="Ratio of pixels corrupted with impulse noise (default: 0.20)")
    g.add_argument("--n_runs",        type=int, default=1,
                   help="Number of noise realizations per image (default: 1)")

    g = p.add_argument_group("SR options")
    g.add_argument("--scale",        type=int, default=4, choices=[2, 3, 4])

    g = p.add_argument_group("Model")
    g.add_argument("--n_feats",      type=int, default=64,
                   help="EDSR/ResNet feature width")
    g.add_argument("--n_resblocks",  type=int, default=16,
                   help="EDSR/ResNet residual blocks")
    g.add_argument("--legacy_arch",   action="store_true", default=False,
                   help="Use original base paper EDSR architecture without SE/GELU/NonLocal")
    g.add_argument("--n_feats_mimo", type=int, default=32,
                   help="MIMO base feature width")
    g.add_argument("--n_res",        type=int, default=8,
                   help="MIMO residual blocks per enc/dec block")

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
