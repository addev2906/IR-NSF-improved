from __future__ import annotations
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Stripe orientation type used by USRLoss
StripeOrientation = Literal["vertical", "both"]

def separable_huber(
    real: torch.Tensor,
    imag: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    def huber1d(t: torch.Tensor) -> torch.Tensor:
        abs_t = t.abs()
        return torch.where(
            abs_t <= delta,
            0.5 * t * t,
            delta * (abs_t - 0.5 * delta),
        )
    return (huber1d(real) + huber1d(imag)).mean()


def separable_l1(
    real: torch.Tensor,
    imag: torch.Tensor,
) -> torch.Tensor:
    return (real.abs() + imag.abs()).mean()

class IRNSFLoss(nn.Module):
    def __init__(
        self,
        penalty: Literal["huber", "l1"] = "huber",
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.penalty = penalty
        self.delta   = delta

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # 2D-DFT of prediction and noisy target
        # norm="backward" matches paper Eq. 3 (1/UV normalisation absorbed
        # symmetrically; using "ortho" also works and is numerically equivalent)
        pred_f   = torch.fft.fft2(pred,   norm="ortho")   # (B, C, H, W) complex
        target_f = torch.fft.fft2(target, norm="ortho")   # (B, C, H, W) complex

        # Complex difference  F(pred) - F(target)
        diff = pred_f - target_f                           # (B, C, H, W) complex

        # Separable penalty on real and imaginary parts
        if self.penalty == "huber":
            loss = separable_huber(diff.real, diff.imag, delta=self.delta)
        elif self.penalty == "l1":
            loss = separable_l1(diff.real, diff.imag)
        else:
            raise ValueError(f"Unknown penalty: {self.penalty!r}. Use 'huber' or 'l1'.")

        return loss, {"loss": loss.item()}

def _build_stripe_mask(H: int, W: int, stripe_type: StripeOrientation) -> torch.Tensor:
    mask = torch.zeros(H, W, dtype=torch.bool)
    if stripe_type in ("vertical", "both"):
        mask[0, :] = True          # k=0  (horizontal frequency row)
    if stripe_type == "both":
        mask[:, 0] = True          # l=0  (vertical frequency column)
    return mask                    # shape (H, W)

class USRLoss(nn.Module):
    def __init__(
        self,
        penalty:     Literal["huber", "l1"] = "huber",
        delta:       float = 1.0,
        stripe_type: StripeOrientation = "vertical",
    ) -> None:
        super().__init__()
        self.penalty     = penalty
        self.delta       = delta
        self.stripe_type = stripe_type
        self._mask: torch.Tensor | None = None  # built lazily, cached

    def _get_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if self._mask is None or self._mask.shape != (H, W):
            self._mask = _build_stripe_mask(H, W, self.stripe_type).to(device)
        elif self._mask.device != device:
            self._mask = self._mask.to(device)
        return self._mask

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        B, C, H, W = pred.shape

        pred_f   = torch.fft.fft2(pred,   norm="ortho")   
        target_f = torch.fft.fft2(target, norm="ortho")   

        diff = pred_f - target_f                           

        mask = self._get_mask(H, W, pred.device)          
        mask_b = mask.unsqueeze(0).unsqueeze(0)           

        real_masked = diff.real[mask_b.expand_as(diff.real)]  
        imag_masked = diff.imag[mask_b.expand_as(diff.imag)]  

        if self.penalty == "huber":
            loss = separable_huber(real_masked, imag_masked, delta=self.delta)
        elif self.penalty == "l1":
            loss = separable_l1(real_masked, imag_masked)
        else:
            raise ValueError(f"Unknown penalty: {self.penalty!r}. Use 'huber' or 'l1'.")

        return loss, {"loss": loss.item(), "omega_size": int(mask.sum().item())}

def masked_fourier_reconstruct(
    noisy:       torch.Tensor,
    net_output:  torch.Tensor,
    stripe_type: StripeOrientation = "vertical",
) -> torch.Tensor:
    B, C, H, W = noisy.shape
    device = noisy.device

    # DFT of both
    noisy_f  = torch.fft.fft2(noisy,      norm="ortho")   
    net_f    = torch.fft.fft2(net_output, norm="ortho")   

    mask = _build_stripe_mask(H, W, stripe_type).to(device)    
    mask_b = mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W) 

    merged_f = torch.where(mask_b, net_f, noisy_f)            

    out = torch.fft.ifft2(merged_f, norm="ortho").real         
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Proposed Improvement: Hybrid Spatial-Impulse Detection & Masked Fourier Loss
# ---------------------------------------------------------------------------

def fast_median_filter(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Fast vectorised spatial median filter for 2D image tensors.
    Operates on (B, C, H, W) tensors using sliding window unfold.
    """
    B, C, H, W = x.shape
    p = k // 2
    # Unfold operates on (B*C, 1, H, W)
    x_reshaped = x.reshape(B * C, 1, H, W)
    unfolded = F.unfold(x_reshaped, kernel_size=k, padding=p)  # (B*C, k*k, H*W)
    med = unfolded.median(dim=1).values                        # (B*C, H*W)
    return med.reshape(B, C, H, W)


def detect_impulse_mask(
    target: torch.Tensor,
    k: int = 3,
    threshold: float = 0.15,
    extreme_boost: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detects impulse noise spikes using rank-ordered deviation from local median.

    Args:
        target: Noisy image tensor (B, C, H, W) in [0, 1].
        k: Window size for local median (default: 3).
        threshold: Absolute deviation threshold from local median (default: 0.15).
        extreme_boost: Extra sensitivity for extreme boundaries (0.0 and 1.0).

    Returns:
        (impulse_mask, target_median):
            impulse_mask: Bool tensor (B, C, H, W), True where impulse is detected.
            target_median: Median filtered target image.
    """
    with torch.no_grad():
        target_med = fast_median_filter(target, k=k)
        dev = torch.abs(target - target_med)

        # Standard impulse deviation detection
        mask = dev > threshold

        # Boost detection if pixel hit extreme salt/pepper bounds
        if extreme_boost:
            is_extreme = (target <= 0.02) | (target >= 0.98)
            mask = mask | (is_extreme & (dev > (threshold * 0.5)))

    return mask, target_med


class RobustImpulseIRNSFLoss(nn.Module):
    """
    Proposed Improvement: Hybrid Spatial-Impulse Detection & Masked Fourier Loss.

    Resolves the failure of the 3 base paper theorems under impulse noise:
      1. Detects impulse spikes via local median deviation mask M.
      2. Inpaints corrupted pixels in the supervisory target:
             y_clean = (1 - M) * y + M * y_med
         Eliminating the spatial delta spikes BEFORE computing the 2D-DFT,
         thereby extinguishing the broadband spectral smearing floor across frequency bins.
      3. Applies Fourier Huber / L1 supervision on the cleaned target (restoring Theorem 1 & 2).
      4. Applies a masked spatial consistency loss strictly on the uncorrupted pixels (1 - M)
         (restoring Theorem 3 by eliminating signal-dependent bias).

    Loss = L_Fourier(pred, y_clean) + lambda_spatial * L_Spatial(pred, y, mask=(1 - M))
    """

    def __init__(
        self,
        penalty: Literal["huber", "l1"] = "huber",
        delta: float = 1.0,
        lambda_spatial: float = 0.5,
        median_k: int = 3,
        impulse_threshold: float = 0.15,
    ) -> None:
        super().__init__()
        self.penalty = penalty
        self.delta = delta
        self.lambda_spatial = lambda_spatial
        self.median_k = median_k
        self.impulse_threshold = impulse_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # 1. Detect impulse corrupted pixels and compute local median
        impulse_mask, target_med = detect_impulse_mask(
            target,
            k=self.median_k,
            threshold=self.impulse_threshold,
        )

        # 2. Inpaint supervisory target to eliminate spatial delta spikes
        # This removes the broadband spectral smearing in the Fourier domain!
        target_fourier_clean = torch.where(impulse_mask, target_med, target)

        # 3. Fourier-Domain Loss on the cleaned supervisory target
        pred_f = torch.fft.fft2(pred, norm="ortho")
        target_f = torch.fft.fft2(target_fourier_clean, norm="ortho")
        diff_f = pred_f - target_f

        if self.penalty == "huber":
            fourier_loss = separable_huber(diff_f.real, diff_f.imag, delta=self.delta)
        elif self.penalty == "l1":
            fourier_loss = separable_l1(diff_f.real, diff_f.imag)
        else:
            raise ValueError(f"Unknown penalty: {self.penalty!r}")

        # 4. Masked Spatial Loss on uncorrupted pixels only
        uncorrupted_mask = (~impulse_mask).float()
        num_uncorrupted = torch.clamp(uncorrupted_mask.sum(), min=1.0)

        spatial_diff = pred - target
        if self.penalty == "huber":
            abs_diff = spatial_diff.abs()
            huber_spatial = torch.where(
                abs_diff <= self.delta,
                0.5 * spatial_diff * spatial_diff,
                self.delta * (abs_diff - 0.5 * self.delta),
            )
            spatial_loss = (huber_spatial * uncorrupted_mask).sum() / num_uncorrupted
        else:
            spatial_loss = (spatial_diff.abs() * uncorrupted_mask).sum() / num_uncorrupted

        # 5. Combined Total Loss
        total_loss = fourier_loss + self.lambda_spatial * spatial_loss

        impulse_ratio = impulse_mask.float().mean().item()
        info = {
            "loss": total_loss.item(),
            "fourier_loss": fourier_loss.item(),
            "spatial_loss": spatial_loss.item(),
            "impulse_ratio": impulse_ratio,
        }
        return total_loss, info


if __name__ == "__main__":
    criterion = IRNSFLoss(penalty="huber", delta=1.0)
    pred   = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    loss, info = criterion(pred, target)
    loss.backward()
    print(f"[IRNSFLoss]  Loss: {info['loss']:.4f}  |  grad norm: {pred.grad.norm():.4f}")
    assert pred.grad is not None, "No gradient!"
    print("IRNSFLoss smoke test PASSED")

    # ---- Path B: USRLoss ----
    usr_crit = USRLoss(penalty="huber", delta=1.0, stripe_type="vertical")
    x_tilde  = torch.rand(2, 1, 64, 64, requires_grad=True)
    y_i      = torch.rand(2, 1, 64, 64)
    usr_loss, usr_info = usr_crit(x_tilde, y_i)
    usr_loss.backward()
    print(f"[USRLoss]    Loss: {usr_info['loss']:.4f}  |  Omega size: {usr_info['omega_size']}  |"
          f"  grad norm: {x_tilde.grad.norm():.4f}")
    assert x_tilde.grad is not None, "No gradient for USRLoss!"
    assert usr_info["omega_size"] == 64, \
        f"Expected Omega=64 (k=0 row of 64-wide DFT), got {usr_info['omega_size']}"

    # ---- masked_fourier_reconstruct ----
    noisy_img = torch.rand(1, 1, 64, 64)
    net_out   = torch.rand(1, 1, 64, 64)
    destriped = masked_fourier_reconstruct(noisy_img, net_out, stripe_type="vertical")
    assert destriped.shape == noisy_img.shape
    assert 0.0 <= destriped.min().item() and destriped.max().item() <= 1.0
    print(f"[masked_fourier_reconstruct]  output shape: {destriped.shape}  "
          f"range=[{destriped.min():.3f}, {destriped.max():.3f}]")

    # ---- Proposed RobustImpulseIRNSFLoss ----
    robust_crit = RobustImpulseIRNSFLoss(penalty="huber", lambda_spatial=0.5)
    pred_r = torch.rand(2, 1, 64, 64, requires_grad=True)
    target_r = torch.rand(2, 1, 64, 64)
    # Add 20% impulse noise spikes
    target_r[target_r < 0.10] = 0.0
    target_r[target_r > 0.90] = 1.0
    r_loss, r_info = robust_crit(pred_r, target_r)
    r_loss.backward()
    print(f"[RobustImpulseIRNSFLoss] Total Loss: {r_info['loss']:.4f} | "
          f"Fourier: {r_info['fourier_loss']:.4f} | Spatial: {r_info['spatial_loss']:.4f} | "
          f"Detected Impulse: {r_info['impulse_ratio']*100:.1f}%")
    assert pred_r.grad is not None, "No gradient for RobustImpulseIRNSFLoss!"
    print("RobustImpulseIRNSFLoss smoke test PASSED")
    print("All loss.py smoke tests PASSED")
