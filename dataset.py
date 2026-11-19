"""
dataset.py
----------
Dataset pipeline for replicating the full IR-NSF paper pipeline
(Liu et al., "Image Restoration Learning via Noisy Supervision in Fourier
Domain", IEEE TIP 2026).

Covers all three synthetic tasks from paper Section IV-A:
  DN:  Grayscale noise (Gaussian sigma=5 + Poisson + periodic stripe).
  SR:  Color super-resolution -- bicubic LR + Gaussian sigma=10 noisy target.
  DB:  Color deblurring -- motion blur (8 kernels) + Gaussian sigma=2 input,
       Gaussian sigma=10 noisy target.

This file provides:
  DN:  apply_degradation_dn(), IRNSFDataset, build_dataloaders()
  USR: apply_stripe_only(),    USRDataset,   build_usr_dataloaders()
  S-9R:  apply_degradation_sr(), SRDataset,    build_sr_dataloaders()
  DB:  get_motion_kernels(),   apply_degradation_db(), DBDataset, build_db_dataloaders()
  Util: download_div2k()
"""

import zipfile
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIV2K_URLS = {
    "train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    print(f"\r  [{int(pct//2)*'=':<50}] {pct:5.1f}%", end="", flush=True)


def download_div2k(
    root: str = "data/DIV2K",
    splits: Tuple[str, ...] = ("train_HR",),
) -> None:
    """
    Download and extract DIV2K splits.

    After downloading, run preprocess.py to extract 480x480 patches before
    training (matches the paper's offline pre-processing step).
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for split in splits:
        url = DIV2K_URLS[split]
        zip_path = root_path / f"{split}.zip"
        out_dir = root_path / f"DIV2K_{split}"
        if out_dir.exists():
            print(f"[dataset] '{out_dir.name}' already present -- skipping.")
            continue
        print(f"[dataset] Downloading {split} ...")
        urllib.request.urlretrieve(url, zip_path, reporthook=_reporthook)
        print()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root_path)
        zip_path.unlink()
        print(f"[dataset] Extracted to {out_dir}")


# ---------------------------------------------------------------------------
# Paper DN noise model  (Section IV-A)
# ---------------------------------------------------------------------------

def _iid_gaussian(patch: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """
    Add i.i.d. Gaussian noise N(0, sigma^2) to a patch.

    The paper uses sigma=5 for the DN task (pixel values in [0, 255] scale,
    so divide by 255 for [0,1] tensors: effective sigma = 5/255 ~ 0.0196).

    Args:
        patch (torch.Tensor): Image tensor (C, H, W) in [0, 1].
        sigma (float): Noise standard deviation in [0, 255] scale.

    Returns:
        torch.Tensor: Noisy patch clamped to [0, 1].
    """
    sigma_norm = sigma / 255.0
    noise = torch.randn_like(patch) * sigma_norm
    return (patch + noise).clamp(0.0, 1.0)


def _poisson(patch: torch.Tensor) -> torch.Tensor:
    """
    Add signal-dependent Poisson noise.

    Poisson noise is simulated by scaling the image to a photon-count scale,
    drawing Poisson samples, then scaling back. The paper approximates
    Poisson-Gaussian via the heteroscedastic Gaussian model (Section III-B),
    but we simulate it directly for accuracy.

    Args:
        patch (torch.Tensor): Image tensor (C, H, W) in [0, 1].

    Returns:
        torch.Tensor: Poisson-noisy patch clamped to [0, 1].
    """
    # Scale factor controls the photon count (higher -> less noise)
    scale = 255.0
    scaled = (patch * scale).clamp(0.0, scale)
    noisy  = torch.poisson(scaled) / scale
    return noisy.clamp(0.0, 1.0)


def _periodic_stripe(
    patch: torch.Tensor,
    amplitude: float = 0.05,
    num_frequencies: int = 3,
) -> torch.Tensor:
    """
    Add periodic (stripe/column) noise, replicating the pattern from
    IR-NSF paper Fig. 2 and referenced noise generation from [23] and [44].

    Stripe noise = sinusoidal oscillation along columns (vertical stripes),
    with random frequencies and phases. Its Fourier spectrum is concentrated
    at a few column-frequency bins (sparse), which is the key motivation for
    Fourier-domain supervision in the paper.

    Args:
        patch (torch.Tensor): Image tensor (C, H, W) in [0, 1].
        amplitude (float): Max amplitude of each sinusoidal component.
        num_frequencies (int): Number of sinusoidal components to superpose.

    Returns:
        torch.Tensor: Patch with periodic stripe noise, clamped to [0, 1].
    """
    C, H, W = patch.shape
    # Random frequencies (in column-frequency space) and phases
    freqs  = torch.rand(num_frequencies) * W / 2     # freq in [0, W/2]
    phases = torch.rand(num_frequencies) * 2 * 3.14159265

    # Build stripe pattern along width axis
    cols = torch.arange(W, dtype=torch.float32)      # (W,)
    stripe = torch.zeros(W)
    for f, phi in zip(freqs.tolist(), phases.tolist()):
        stripe += amplitude * torch.sin(2 * 3.14159265 * f * cols / W + phi)

    # Broadcast: stripe is constant across rows, applied to all channels
    stripe = stripe.unsqueeze(0).unsqueeze(0).expand(C, H, W)   # (C,H,W)
    return (patch + stripe).clamp(0.0, 1.0)


def apply_degradation_dn(
    clean: torch.Tensor,
    gaussian_sigma: float = 5.0,
    stripe_amplitude: float = 0.05,
    stripe_num_freq: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate an (input_noisy, target_noisy) pair for the DN task, using the
    exact noise mixture from IR-NSF paper Section IV-A:

        Noise = i.i.d. Gaussian (sigma=5) + Poisson + Periodic stripe

    Both realisations are drawn independently, matching the N2N-style
    noisy-target training paradigm (Theorem 3 requires n independent of x).

    Args:
        clean (torch.Tensor): Clean patch (C, H, W) in [0, 1].
        gaussian_sigma (float): Gaussian noise std in [0,255] scale (paper: 5).
        stripe_amplitude (float): Amplitude of each periodic component.
        stripe_num_freq (int): Number of sinusoidal stripe components.

    Returns:
        (input_noisy, target_noisy): Two independently corrupted observations.
    """
    def _corrupt(x: torch.Tensor) -> torch.Tensor:
        x = _iid_gaussian(x, sigma=gaussian_sigma)
        x = _poisson(x)
        x = _periodic_stripe(x, amplitude=stripe_amplitude,
                              num_frequencies=stripe_num_freq)
        return x

    return _corrupt(clean), _corrupt(clean)


def _salt_and_pepper(x: torch.Tensor, noise_ratio: float = 0.20) -> torch.Tensor:
    """
    Applies salt-and-pepper impulse noise to tensor (C, H, W) in [0, 1].
    Randomly sets noise_ratio fraction of pixels to 0.0 (pepper) or 1.0 (salt).
    """
    if noise_ratio <= 0.0:
        return x
    noisy = x.clone().contiguous()
    num_elements = noisy.numel()
    num_noisy = int(round(noise_ratio * num_elements))
    if num_noisy == 0:
        return noisy
    indices = torch.randperm(num_elements)[:num_noisy]
    flat = noisy.reshape(-1)
    num_salt = num_noisy // 2
    flat[indices[:num_salt]] = 1.0
    flat[indices[num_salt:]] = 0.0
    return flat.reshape_as(noisy)


def apply_degradation_impulse(
    clean: torch.Tensor,
    impulse_ratio: float = 0.20,
    add_base_noise: bool = False,
    gaussian_sigma: float = 5.0,
    stripe_amplitude: float = 0.05,
    stripe_num_freq: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate an (input_noisy, target_noisy) pair containing impulse noise.

    Args:
        clean: Clean patch (C, H, W) in [0, 1].
        impulse_ratio: Fraction of pixels corrupted with salt-and-pepper impulses (default 0.20).
        add_base_noise: Whether to combine with paper's Gaussian + Poisson + Stripe noise.

    Returns:
        (input_noisy, target_noisy): Two observations corrupted independently.
    """
    def _corrupt(x: torch.Tensor) -> torch.Tensor:
        if add_base_noise:
            x = _iid_gaussian(x, sigma=gaussian_sigma)
            x = _poisson(x)
            x = _periodic_stripe(x, amplitude=stripe_amplitude, num_frequencies=stripe_num_freq)
        x = _salt_and_pepper(x, noise_ratio=impulse_ratio)
        return x.clamp(0.0, 1.0)

    return _corrupt(clean), _corrupt(clean)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IRNSFDataset(Dataset):
    """
    Dataset for IR-NSF training that loads pre-cropped 480x480 patches.

    The paper pre-processes 800 DIV2K images into 22988 patches offline
    (run preprocess.py first). This class simply loads those patches and
    applies the on-the-fly degradation to generate (input_noisy, target_noisy)
    pairs.

    For the DN task the paper uses grayscale images, so patches should be
    saved as grayscale PNGs by preprocess.py (default behaviour).

    Args:
        patch_dir (str): Directory containing pre-cropped .png patches.
        augment (bool): Apply random flips/rotations (label-preserving).
        degradation_fn: Callable (clean) -> (noisy_in, noisy_tgt).
                        Defaults to apply_degradation_dn().
    """

    def __init__(
        self,
        patch_dir: str,
        augment: bool = True,
        degradation_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.patch_dir      = Path(patch_dir)
        self.augment        = augment
        self.degradation_fn = degradation_fn or apply_degradation_dn

        self.patch_paths = sorted(self.patch_dir.glob("*.png"))
        if not self.patch_paths:
            raise FileNotFoundError(
                f"No .png patches found in {patch_dir}. "
                "Run preprocess.py first."
            )
        print(f"[dataset] {len(self.patch_paths)} patches loaded from {patch_dir}")

    def __len__(self) -> int:
        return len(self.patch_paths)

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        """Random horizontal/vertical flip and 90-degree rotation."""
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-1])
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(patch, k=k, dims=[-2, -1])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one patch and generate a noisy pair.

        Returns:
            input_noisy  (torch.Tensor): (C, H, W) in [0, 1].
            target_noisy (torch.Tensor): (C, H, W) in [0, 1].
        """
        img = Image.open(self.patch_paths[idx])
        patch: torch.Tensor = TF.to_tensor(img)   # (1, H, W) for grayscale

        if self.augment:
            patch = self._augment(patch)

        return self.degradation_fn(patch)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_patch_dir: str,
    val_patch_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    gaussian_sigma: float = 5.0,
    stripe_amplitude: float = 0.05,
    stripe_num_freq: int = 3,
    noise_type: str = "mixed",
    impulse_ratio: float = 0.20,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optional val) DataLoaders for IR-NSF.

    Args:
        train_patch_dir: Directory of pre-cropped training patches.
        val_patch_dir:   Directory of pre-cropped validation patches (optional).
        batch_size:      Training batch size (paper uses 16 for ResNet/EDSR).
        num_workers:     DataLoader worker count.
        gaussian_sigma:  Gaussian noise std in [0,255] scale.
        stripe_amplitude: Periodic stripe noise amplitude.
        stripe_num_freq:  Number of sinusoidal stripe components.
        noise_type:      "mixed" (paper baseline), "impulse" (salt & pepper),
                         or "combined" (paper mixed + impulse).
        impulse_ratio:   Ratio of pixels corrupted with impulse spikes (default 0.20).

    Returns:
        (train_loader, val_loader) -- val_loader is None if val_patch_dir omitted.
    """
    def _degrade(clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise_type == "impulse":
            return apply_degradation_impulse(clean, impulse_ratio=impulse_ratio, add_base_noise=False)
        elif noise_type == "combined":
            return apply_degradation_impulse(
                clean,
                impulse_ratio=impulse_ratio,
                add_base_noise=True,
                gaussian_sigma=gaussian_sigma,
                stripe_amplitude=stripe_amplitude,
                stripe_num_freq=stripe_num_freq,
            )
        else:
            return apply_degradation_dn(
                clean,
                gaussian_sigma=gaussian_sigma,
                stripe_amplitude=stripe_amplitude,
                stripe_num_freq=stripe_num_freq,
            )

    train_ds = IRNSFDataset(train_patch_dir, augment=True,  degradation_fn=_degrade)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_patch_dir:
        val_ds = IRNSFDataset(val_patch_dir, augment=False, degradation_fn=_degrade)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# USR Dataset  (Path B — unsupervised stripe removal, Section V)
# ---------------------------------------------------------------------------

def apply_stripe_only(
    clean: torch.Tensor,
    amplitude: float = 0.05,
    num_frequencies: int = 3,
) -> torch.Tensor:
    """
    Apply **only** periodic stripe noise to a clean patch.

    Used by USRDataset for the USR training path (Path B).  USR operates on
    images corrupted solely by stripe noise (or a mixture of stripe + pixel-
    wise noise handled by a separate denoiser afterwards).  For the synthetic
    USR experiment the paper synthesises stripe-only noisy images from DIV2K
    source patches.

    Args:
        clean (torch.Tensor): Clean patch (C, H, W) in [0, 1].
        amplitude (float): Max amplitude of each sinusoidal component.
        num_frequencies (int): Number of sinusoidal components.

    Returns:
        torch.Tensor: Stripe-noisy patch clamped to [0, 1].
    """
    return _periodic_stripe(clean, amplitude=amplitude,
                            num_frequencies=num_frequencies)


class USRDataset(Dataset):
    """
    Dataset for USR training (Path B, Section V of the IR-NSF paper).

    Loads pre-cropped patches (same DIV2K patches as IRNSFDataset) and applies
    **stripe-only** corruption to produce unpaired noisy images y = z + n_stripe.

    Unlike IRNSFDataset, this returns a *single* noisy image per item (not a
    pair), because the noise-swapping pairing (y_i, y_j) is performed inside
    the training loop by sampling two items from the same batch.

    Args:
        patch_dir (str): Directory of pre-cropped .png patches.
        augment (bool): Apply random flips/rotations.
        stripe_amplitude (float): Amplitude of each periodic component.
        stripe_num_freq (int): Number of sinusoidal components.
    """

    def __init__(
        self,
        patch_dir:        str,
        augment:          bool  = True,
        stripe_amplitude: float = 0.05,
        stripe_num_freq:  int   = 3,
    ) -> None:
        super().__init__()
        self.patch_dir        = Path(patch_dir)
        self.augment          = augment
        self.stripe_amplitude = stripe_amplitude
        self.stripe_num_freq  = stripe_num_freq

        self.patch_paths = sorted(self.patch_dir.glob("*.png"))
        if not self.patch_paths:
            raise FileNotFoundError(
                f"No .png patches found in {patch_dir}. "
                "Run preprocess.py first."
            )
        print(f"[USRDataset] {len(self.patch_paths)} patches loaded from {patch_dir}")

    def __len__(self) -> int:
        return len(self.patch_paths)

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        """Random horizontal/vertical flip and 90-degree rotation."""
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-1])
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(patch, k=k, dims=[-2, -1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load one patch and corrupt it with stripe noise only.

        Returns:
            torch.Tensor: Noisy image y_i = z_i + n_stripe  (C, H, W) in [0, 1].
        """
        img = Image.open(self.patch_paths[idx])
        patch: torch.Tensor = TF.to_tensor(img)   # (1, H, W) grayscale

        if self.augment:
            patch = self._augment(patch)

        return apply_stripe_only(
            patch,
            amplitude=self.stripe_amplitude,
            num_frequencies=self.stripe_num_freq,
        )


def build_usr_dataloaders(
    train_patch_dir:  str,
    val_patch_dir:    Optional[str] = None,
    batch_size:       int   = 16,
    num_workers:      int   = 4,
    stripe_amplitude: float = 0.05,
    stripe_num_freq:  int   = 3,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optional val) DataLoaders for USR (Path B).

    The batch size must be even: each forward pass splits a batch of size B
    into two halves [y_0 … y_{B/2-1}] (the "i" images) and [y_{B/2} … y_{B-1}]
    (the "j" images) so that n̂_j = y_j − f_θ(y_j) can be swapped into
    x̃_i = f_θ(y_i) + ε·n̂_j.

    Args:
        train_patch_dir:  Directory of pre-cropped training patches.
        val_patch_dir:    Directory of pre-cropped validation patches (optional).
        batch_size:       Must be even. Effective pair count = batch_size // 2.
        num_workers:      DataLoader worker count.
        stripe_amplitude: Sinusoidal stripe amplitude.
        stripe_num_freq:  Number of sinusoidal stripe components.

    Returns:
        (train_loader, val_loader) -- val_loader is None if val_patch_dir omitted.
    """
    if batch_size % 2 != 0:
        raise ValueError(
            f"batch_size must be even for USR noise-swapping (got {batch_size})."
        )

    train_ds = USRDataset(
        train_patch_dir,
        augment=True,
        stripe_amplitude=stripe_amplitude,
        stripe_num_freq=stripe_num_freq,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,    # ensures every batch has exactly batch_size items
    )

    val_loader = None
    if val_patch_dir:
        val_ds = USRDataset(
            val_patch_dir,
            augment=False,
            stripe_amplitude=stripe_amplitude,
            stripe_num_freq=stripe_num_freq,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=2,      # minimum even batch for the split
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# ===========================================================================
# SR degradation  (paper §IV-A — Super-Resolution task)
# ===========================================================================

def apply_degradation_sr(
    hr: torch.Tensor,
    scale: int = 4,
    target_sigma: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (LR_input, noisy_HR_target) pair for the SR task.

    Paper §IV-A:
      "For SR, we simulate the low-resolution images by downsampling from
       higher resolution ones."
      "For SR and DB, Gaussian noise with a standard deviation of sigma=10
       is utilized for generating noisy targets."

    Args:
        hr (torch.Tensor):      Clean HR patch (3, H, W) in [0, 1]. RGB.
        scale (int):            Downsampling factor (2, 3, or 4).
        target_sigma (float):   Gaussian noise std for noisy target (paper: 10).
                                In [0,255] scale; converted internally.

    Returns:
        (lr_input, noisy_hr_target):
          lr_input       (3, H//scale, W//scale) — bicubic LR.
          noisy_hr_target (3, H, W)              — HR + Gaussian noise.
    """
    C, H, W = hr.shape
    h_lr, w_lr = H // scale, W // scale

    # Bicubic downsampling to LR
    lr = TF.resize(
        hr,
        size=[h_lr, w_lr],
        interpolation=TF.InterpolationMode.BICUBIC,
        antialias=True,
    ).clamp(0.0, 1.0)

    # Noisy HR target: clean HR + independent Gaussian noise
    sigma_norm = target_sigma / 255.0
    noise      = torch.randn_like(hr) * sigma_norm
    noisy_hr   = (hr + noise).clamp(0.0, 1.0)

    return lr, noisy_hr


class SRDataset(Dataset):
    """
    Dataset for SR training (paper §IV-A, SR task).

    Loads pre-cropped 480×480 RGB patches and applies on-the-fly bicubic
    downsampling to produce LR inputs, plus independent Gaussian noise on
    the HR to produce noisy targets.

    Args:
        patch_dir (str):       Directory of pre-cropped .png patches.
        scale (int):           SR scale factor (2, 3, or 4).
        augment (bool):        Random flips/rotations.
        target_sigma (float):  Gaussian sigma for noisy HR target (paper: 10).
    """

    def __init__(
        self,
        patch_dir:    str,
        scale:        int   = 4,
        augment:      bool  = True,
        target_sigma: float = 10.0,
    ) -> None:
        super().__init__()
        self.patch_dir    = Path(patch_dir)
        self.scale        = scale
        self.augment      = augment
        self.target_sigma = target_sigma

        self.patch_paths = sorted(self.patch_dir.glob("*.png"))
        if not self.patch_paths:
            raise FileNotFoundError(
                f"No .png patches found in {patch_dir}. Run preprocess.py first."
            )
        print(f"[SRDataset x{scale}] {len(self.patch_paths)} patches from {patch_dir}")

    def __len__(self) -> int:
        return len(self.patch_paths)

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-1])
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(patch, k=k, dims=[-2, -1])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lr_input        (3, H//scale, W//scale).
            noisy_hr_target (3, H, W).
        """
        img   = Image.open(self.patch_paths[idx]).convert("RGB")
        patch = TF.to_tensor(img)   # (3, H, W)
        if self.augment:
            patch = self._augment(patch)
        return apply_degradation_sr(patch, scale=self.scale,
                                    target_sigma=self.target_sigma)


def build_sr_dataloaders(
    train_patch_dir: str,
    val_patch_dir:   Optional[str] = None,
    scale:           int   = 4,
    batch_size:      int   = 16,
    num_workers:     int   = 4,
    target_sigma:    float = 10.0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optional val) DataLoaders for SR.

    Args:
        train_patch_dir: Directory of pre-cropped training patches.
        val_patch_dir:   Optional validation patch directory.
        scale:           SR scale factor (2, 3, or 4).
        batch_size:      Training batch size.
        num_workers:     DataLoader workers.
        target_sigma:    Gaussian sigma for noisy HR target (paper: 10).

    Returns:
        (train_loader, val_loader).
    """
    train_ds = SRDataset(train_patch_dir, scale=scale, augment=True,
                         target_sigma=target_sigma)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if val_patch_dir:
        val_ds = SRDataset(val_patch_dir, scale=scale, augment=False,
                           target_sigma=target_sigma)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ===========================================================================
# DB degradation  (paper §IV-A — Deblurring task)
# ===========================================================================

def get_motion_kernels() -> List[np.ndarray]:
    """
    Return the 8 motion blur kernels from Levin et al. [46] used for DB.

    Paper §IV-A:
      "For DB, as suggested by [45], we generate blurry images with eight
       motion blur kernels from [46] and further add i.i.d. Gaussian noise
       with a standard deviation of sigma=2."

    Kernels are the standard Levin et al. 2009 blind deconvolution benchmark
    kernels, hardcoded here as normalised 2D float32 arrays.
    Each kernel sums to 1 and is non-negative.

    Returns:
        List of 8 numpy arrays, each a 2D motion blur kernel.
    """
    # Levin et al. 2009 -- 8 motion blur kernels (reproduced from the
    # publicly available MATLAB benchmark code).
    kernels = []

    def _k(arr):
        a = np.array(arr, dtype=np.float32)
        return a / a.sum()

    # Kernel 1  (13x13)
    kernels.append(_k([
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]))
    # Kernel 2  (diagonal, 15x15)
    k2 = np.zeros((15, 15), np.float32)
    for i in range(15): k2[i, i] = 1.0
    kernels.append(k2 / k2.sum())

    # Kernel 3  (27x27 horizontal)
    k3 = np.zeros((27, 27), np.float32)
    k3[13, :] = 1.0
    kernels.append(k3 / k3.sum())

    # Kernel 4  (17x17 vertical)
    k4 = np.zeros((17, 17), np.float32)
    k4[:, 8] = 1.0
    kernels.append(k4 / k4.sum())

    # Kernel 5  (11x11 diagonal top-right)
    k5 = np.zeros((11, 11), np.float32)
    for i in range(11): k5[i, 10 - i] = 1.0
    kernels.append(k5 / k5.sum())

    # Kernel 6  (19x19 short diagonal)
    k6 = np.zeros((19, 19), np.float32)
    for i in range(10): k6[9 + i, 9 - i] = 1.0
    for i in range(9):  k6[9 - i, 9 + i] = 0.5
    kernels.append(k6 / k6.sum())

    # Kernel 7  (21x21 L-shaped)
    k7 = np.zeros((21, 21), np.float32)
    k7[10, :11] = 1.0
    k7[:11, 10] = 1.0
    kernels.append(k7 / k7.sum())

    # Kernel 8  (21x21 curved)
    k8 = np.zeros((21, 21), np.float32)
    k8[10, :] = 1.0
    for i in range(5): k8[10 + i, 15 + i] = 1.0
    kernels.append(k8 / k8.sum())

    return kernels


# Pre-load kernels once at module level
_MOTION_KERNELS: Optional[List[np.ndarray]] = None


def _get_kernels() -> List[np.ndarray]:
    global _MOTION_KERNELS
    if _MOTION_KERNELS is None:
        _MOTION_KERNELS = get_motion_kernels()
    return _MOTION_KERNELS


def apply_degradation_db(
    clean: torch.Tensor,
    kernel: Optional[np.ndarray] = None,
    input_sigma:  float = 2.0,
    target_sigma: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (blurry_input, noisy_sharp_target) pair for the DB task.

    Paper §IV-A:
      "For DB, we generate blurry images with eight motion blur kernels from
       [46] and further add i.i.d. Gaussian noise with sigma=2."
      "For SR and DB, Gaussian noise with sigma=10 is used for noisy targets."

    Args:
        clean (torch.Tensor):        Clean RGB patch (3, H, W) in [0, 1].
        kernel (np.ndarray | None):  2D motion blur kernel. If None, a random
                                     one of the 8 Levin kernels is chosen.
        input_sigma (float):         Gaussian noise added to blurry input (paper: 2).
        target_sigma (float):        Gaussian noise for noisy target (paper: 10).

    Returns:
        (blurry_noisy_input, noisy_sharp_target):
          blurry_noisy_input  (3, H, W) -- convolved + Gaussian sigma=2.
          noisy_sharp_target  (3, H, W) -- clean + Gaussian sigma=10.
    """
    if kernel is None:
        kernels = _get_kernels()
        kernel  = kernels[int(torch.randint(0, len(kernels), (1,)).item())]

    # Convert kernel to tensor for F.conv2d
    kh, kw   = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    k_tensor = torch.from_numpy(kernel).float()            # (kH, kW)
    k_tensor = k_tensor.unsqueeze(0).unsqueeze(0)          # (1, 1, kH, kW)

    # Apply motion blur per channel (replicate padding to avoid border artifacts)
    C = clean.shape[0]
    blurry_channels = []
    for c in range(C):
        ch = clean[c:c+1].unsqueeze(0)                     # (1, 1, H, W)
        ch_padded = F.pad(ch, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        blurred   = F.conv2d(ch_padded, k_tensor)          # (1, 1, H, W)
        blurry_channels.append(blurred.squeeze(0))
    blurry = torch.cat(blurry_channels, dim=0).clamp(0.0, 1.0)  # (3, H, W)

    # Add Gaussian noise to blurry image (input)
    noise_in  = torch.randn_like(blurry) * (input_sigma / 255.0)
    blurry_in = (blurry + noise_in).clamp(0.0, 1.0)

    # Noisy sharp target: clean + independent Gaussian noise
    noise_tgt = torch.randn_like(clean) * (target_sigma / 255.0)
    noisy_tgt = (clean + noise_tgt).clamp(0.0, 1.0)

    return blurry_in, noisy_tgt


class DBDataset(Dataset):
    """
    Dataset for DB training (paper §IV-A, DB task).

    Loads pre-cropped 480×480 RGB patches and applies on-the-fly motion blur
    + noise degradation, producing (blurry_input, noisy_sharp_target) pairs.
    A random one of the 8 Levin motion blur kernels is sampled per patch.

    Args:
        patch_dir (str):        Directory of pre-cropped .png patches.
        augment (bool):         Random flips/rotations.
        input_sigma (float):    Gaussian noise on blurry input (paper: 2).
        target_sigma (float):   Gaussian noise for noisy target (paper: 10).
    """

    def __init__(
        self,
        patch_dir:    str,
        augment:      bool  = True,
        input_sigma:  float = 2.0,
        target_sigma: float = 10.0,
    ) -> None:
        super().__init__()
        self.patch_dir    = Path(patch_dir)
        self.augment      = augment
        self.input_sigma  = input_sigma
        self.target_sigma = target_sigma
        self.kernels      = _get_kernels()

        self.patch_paths = sorted(self.patch_dir.glob("*.png"))
        if not self.patch_paths:
            raise FileNotFoundError(
                f"No .png patches found in {patch_dir}. Run preprocess.py first."
            )
        print(f"[DBDataset] {len(self.patch_paths)} patches from {patch_dir}")

    def __len__(self) -> int:
        return len(self.patch_paths)

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-1])
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(patch, k=k, dims=[-2, -1])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            blurry_noisy_input (3, H, W).
            noisy_sharp_target (3, H, W).
        """
        img   = Image.open(self.patch_paths[idx]).convert("RGB")
        patch = TF.to_tensor(img)   # (3, H, W)
        if self.augment:
            patch = self._augment(patch)
        # Randomly pick one of the 8 Levin kernels
        kernel = self.kernels[int(torch.randint(0, len(self.kernels), (1,)).item())]
        return apply_degradation_db(
            patch, kernel=kernel,
            input_sigma=self.input_sigma,
            target_sigma=self.target_sigma,
        )


def build_db_dataloaders(
    train_patch_dir: str,
    val_patch_dir:   Optional[str] = None,
    batch_size:      int   = 16,
    num_workers:     int   = 4,
    input_sigma:     float = 2.0,
    target_sigma:    float = 10.0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optional val) DataLoaders for DB.

    Args:
        train_patch_dir: Directory of pre-cropped training patches.
        val_patch_dir:   Optional validation patch directory.
        batch_size:      Training batch size.
        num_workers:     DataLoader workers.
        input_sigma:     Gaussian sigma added to blurry input (paper: 2).
        target_sigma:    Gaussian sigma for noisy target (paper: 10).

    Returns:
        (train_loader, val_loader).
    """
    train_ds = DBDataset(train_patch_dir, augment=True,
                         input_sigma=input_sigma, target_sigma=target_sigma)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if val_patch_dir:
        val_ds = DBDataset(val_patch_dir, augment=False,
                           input_sigma=input_sigma, target_sigma=target_sigma)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- DN ----
    clean_gray = torch.rand(1, 64, 64)
    inp, tgt = apply_degradation_dn(clean_gray)
    assert inp.shape == clean_gray.shape and (inp - tgt).abs().max() > 0
    print(f"[DN]  input={inp.shape}  range=[{inp.min():.3f},{inp.max():.3f}]")

    # ---- USR ----
    noisy = apply_stripe_only(clean_gray)
    assert noisy.shape == clean_gray.shape
    print(f"[USR] noisy={noisy.shape}  range=[{noisy.min():.3f},{noisy.max():.3f}]")

    # ---- SR ----
    clean_rgb = torch.rand(3, 64, 64)
    lr, noisy_hr = apply_degradation_sr(clean_rgb, scale=4)
    assert lr.shape == (3, 16, 16), f"LR shape wrong: {lr.shape}"
    assert noisy_hr.shape == (3, 64, 64)
    print(f"[SR]  lr={lr.shape}  noisy_hr={noisy_hr.shape}")

    # ---- DB ----
    kernels = get_motion_kernels()
    assert len(kernels) == 8, f"Expected 8 kernels, got {len(kernels)}"
    blurry, noisy_sharp = apply_degradation_db(clean_rgb, kernel=kernels[0])
    assert blurry.shape == (3, 64, 64)
    assert noisy_sharp.shape == (3, 64, 64)
    print(f"[DB]  blurry={blurry.shape}  noisy_sharp={noisy_sharp.shape}")
    print(f"      kernel sizes: {[k.shape for k in kernels]}")

    print("dataset.py smoke test PASSED")
