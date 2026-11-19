"""
model.py
--------
All network backbones for IR-NSF (Liu et al., IEEE TIP 2026).

Paper Section IV-A (Network Architectures):
  - DN task:  ResNet  -- modified EDSR, no upsample, global skip, grayscale.
  - SR task:  EDSR    -- standard EDSR with sub-pixel upsample block, RGB.
  - DB task:  MIMO    -- multi-input multi-output U-Net, RGB (Cho et al. [28]).

All backbones share the EDSR-style ResidualBlock (no BatchNorm, res_scale=0.1).
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Shared building block
# ===========================================================================

class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention with a GELU bottleneck."""

    def __init__(self, n_feats: int = 64, reduction_ratio: float = 16.0) -> None:
        super().__init__()
        if n_feats < 2 or not math.isclose(n_feats / reduction_ratio, round(n_feats / reduction_ratio)):
            raise ValueError("n_feats must be divisible by reduction_ratio")
        reduced_feats = max(1, n_feats // int(reduction_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(n_feats, reduced_feats),
            nn.GELU(),
            nn.Linear(reduced_feats, n_feats),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.squeeze(x).flatten(1)
        weights = self.excite(weights)
        return x * weights.unsqueeze(-1).unsqueeze(-1)


class SubsampledNonLocalBlock(nn.Module):
    """Channel-wise attention with constant-size spatial activations."""

    def __init__(self, n_feats: int = 32, reduced_feats: int = 16) -> None:
        super().__init__()
        if n_feats <= reduced_feats:
            raise ValueError("n_feats must be greater than reduced_feats")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(n_feats, reduced_feats, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(reduced_feats, n_feats, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(n_feats, n_feats, 1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.excite(self.pool(x)).expand_as(x)
        return x + self.gamma * self.out_proj(x * weights)


class LegacyResidualBlock(nn.Module):
    """
    EDSR-style residual block: Conv -> ReLU -> Conv -> scale + skip.
    No BatchNorm (BN is detrimental for blind denoising).
    Used for loading baseline (pre-improvement) checkpoints.
    """

    def __init__(self, n_feats: int = 64, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualBlock(nn.Module):
    """Residual block with GELU, channel attention, and scaled branch output."""

    def __init__(self, n_feats: int = 64, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
        )
        self.se = SEBlock(n_feats)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se(self.body(x)) * self.res_scale


class ResNet(nn.Module):
    """
    Modified EDSR for image denoising -- the "ResNet" backbone used in IR-NSF.

    Compared to standard EDSR:
      - No upsample block (spatial resolution preserved).
      - Global long skip connection from input to output.
      - 1-channel grayscale input/output (for DN task).

    Data flow:
        x  -->  head_conv  -->  [ResBlock x N]  -->  refine_conv
           -->  tail_conv  -->  + x (global skip)  -->  clamp[0,1]
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        out_channels: int   = 1,
        n_feats:      int   = 64,
        n_resblocks:  int   = 16,
        res_scale:    float = 0.1,
        clamp_output: bool  = True,
        legacy:       bool  = False,
    ) -> None:
        super().__init__()
        self.clamp_output = clamp_output
        self.legacy = legacy
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1, bias=True)

        if legacy:
            # Original EDSR-style architecture (ReLU, no SE, no NonLocal)
            self.body = nn.Sequential(
                *[LegacyResidualBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)],
                nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            )
        else:
            # Improved architecture (GELU + SE + NonLocal at midpoint)
            if n_resblocks < 9:
                raise ValueError("ResNet requires at least 9 residual blocks for midpoint attention")
            residual_layers = [
                ResidualBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)
            ]
            residual_layers.insert(8, SubsampledNonLocalBlock(n_feats))
            self.body = nn.Sequential(
                *residual_layers,
                nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            )
            self.nonlocal_block = self.body[8]

        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Tail layer for residual network: initialize with small weights
        # so initial output is close to identity (out = x + tail(res) ~ x).
        # This prevents early gradient collapse into flat/zero regions.
        if hasattr(self, "tail") and isinstance(self.tail, nn.Conv2d):
            self.tail.weight.data.mul_(0.001)
            if self.tail.bias is not None:
                self.tail.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res  = self.body(feat) + feat
        out  = self.tail(res) + x
        if self.clamp_output and not self.training:
            out = out.clamp(0.0, 1.0)
        return out

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    n_feats:     int   = 64,
    n_resblocks: int   = 16,
    res_scale:   float = 0.1,
    in_channels: int   = 1,
    legacy:      bool  = False,
    device: Optional[torch.device] = None,
) -> ResNet:
    """Build the IR-NSF ResNet backbone (modified EDSR for DN)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(
        in_channels=in_channels, out_channels=in_channels,
        n_feats=n_feats, n_resblocks=n_resblocks, res_scale=res_scale,
        legacy=legacy,
    ).to(device)
    arch_tag = "legacy" if legacy else "improved (SE+GELU+NonLocal)"
    print(f"[model] ResNet (DN) [{arch_tag}] | n_feats={n_feats} | "
          f"n_resblocks={n_resblocks} | params={model.num_parameters:,} | device={device}")
    return model


# ===========================================================================
# EDSR -- Super-Resolution backbone  (paper §IV-A, for SR task)
# ===========================================================================

class Upsampler(nn.Sequential):
    """
    Sub-pixel convolution upsample block (EDSR paper [53]).
    Stacks log2(scale) PixelShuffle x2 stages for scale 2 or 4;
    single PixelShuffle x3 for scale 3.
    """

    def __init__(self, scale: int, n_feats: int) -> None:
        modules: list = []
        if scale in (2, 4):
            for _ in range(int(math.log2(scale))):
                modules += [
                    nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=True),
                    nn.PixelShuffle(2),
                ]
        elif scale == 3:
            modules += [
                nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=True),
                nn.PixelShuffle(3),
            ]
        else:
            raise ValueError(f"Unsupported scale {scale}. Use 2, 3, or 4.")
        super().__init__(*modules)


class EDSR(nn.Module):
    """
    Standard EDSR for super-resolution -- the SR backbone in IR-NSF.

    Data flow:
        x (LR)  -->  head  -->  [ResBlock x N]  -->  refine
               -->  Upsampler(scale)  -->  tail  -->  clamp[0,1]
    """

    def __init__(
        self,
        scale:        int   = 4,
        in_channels:  int   = 3,
        out_channels: int   = 3,
        n_feats:      int   = 64,
        n_resblocks:  int   = 16,
        res_scale:    float = 0.1,
        clamp_output: bool  = True,
    ) -> None:
        super().__init__()
        self.scale        = scale
        self.clamp_output = clamp_output
        self.head     = nn.Conv2d(in_channels, n_feats, 3, padding=1, bias=True)
        self.body     = nn.Sequential(
            *[ResidualBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
        )
        self.upsample = Upsampler(scale, n_feats)
        self.tail     = nn.Conv2d(n_feats, out_channels, 3, padding=1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.tail.weight.data.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.body(feat) + feat
        feat = self.upsample(feat)
        out  = self.tail(feat)
        x_up = F.interpolate(x, scale_factor=float(self.scale), mode="bicubic", align_corners=False)
        out  = out + x_up
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_edsr(
    scale:       int   = 4,
    n_feats:     int   = 64,
    n_resblocks: int   = 16,
    device: Optional[torch.device] = None,
) -> EDSR:
    """Build the EDSR SR backbone (RGB, with sub-pixel upsample)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDSR(
        scale=scale, in_channels=3, out_channels=3,
        n_feats=n_feats, n_resblocks=n_resblocks,
    ).to(device)
    print(f"[model] EDSR (SR x{scale}) | n_feats={n_feats} | "
          f"n_resblocks={n_resblocks} | params={model.num_parameters:,} | device={device}")
    return model


# ===========================================================================
# MIMO -- Deblurring backbone  (paper §IV-A, for DB task)
# ===========================================================================
# Faithful implementation of MIMO-UNet from:
#   Cho et al., "Rethinking Coarse-to-Fine Approach in Single Image Deblurring"
#   ICCV 2021.  (Paper reference [28] in IR-NSF.)
#
# 3-scale encoder/decoder with multi-scale inputs at each encoder level.
# Multi-scale outputs; only the full-scale output is used at inference.
# ===========================================================================

class _ConvReLU(nn.Sequential):
    """Conv2d + ReLU helper."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    """Lightweight residual block with res_scale used inside MIMO EBlock/DBlock."""
    def __init__(self, n_feats: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class _EBlock(nn.Module):
    """MIMO encoder block: projects input -> features -> skip + downsampled."""
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 8) -> None:
        super().__init__()
        self.proj = _ConvReLU(in_ch, out_ch)
        self.body = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor):
        feat = self.body(self.proj(x))
        return feat, self.down(feat)


class _DBlock(nn.Module):
    """MIMO decoder block: fuses skip + upsampled coarser output -> features + aux image."""
    def __init__(self, in_ch: int, out_ch: int, img_ch: int = 3, n_res: int = 8) -> None:
        super().__init__()
        self.proj   = _ConvReLU(in_ch, out_ch)
        self.body   = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])
        self.up     = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2, bias=True)
        self.to_img = nn.Conv2d(out_ch, img_ch, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None):
        inp  = torch.cat([x, skip], dim=1) if skip is not None else x
        feat = self.body(self.proj(inp))
        return self.up(feat), self.to_img(feat)


class MIMO(nn.Module):
    """
    MIMO-UNet for image deblurring -- DB backbone in IR-NSF.

    Data flow:
        Encoder:
          e0, e0d = EBlock_0(x)
          e1, e1d = EBlock_1(cat(e0d, x_half))
          e2, _   = EBlock_2(cat(e1d, x_qtr))
        Decoder:
          d2, out_qtr  = DBlock_2(e2)
          d1, out_half = DBlock_1(cat(d2, e1))
          _,  out_full = DBlock_0(cat(d1, e0))

    At inference: only out_full is used. H and W must be divisible by 4.
    """

    def __init__(
        self,
        in_channels:  int = 3,
        out_channels: int = 3,
        n_feats:      int = 32,
        n_res:        int = 8,
    ) -> None:
        super().__init__()
        C, IC, OC = n_feats, in_channels, out_channels

        self.enc0 = _EBlock(IC,         C,     n_res)
        self.enc1 = _EBlock(C + IC,     C * 2, n_res)
        self.enc2 = _EBlock(C * 2 + IC, C * 4, n_res)

        self.dec2 = _DBlock(C * 4,           C * 2, OC, n_res)
        self.dec1 = _DBlock(C * 2 + C * 2,   C,     OC, n_res)
        self.dec0 = _DBlock(C + C,            C,     OC, n_res)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.dec0.to_img.weight.data.mul_(0.01)
        self.dec1.to_img.weight.data.mul_(0.01)
        self.dec2.to_img.weight.data.mul_(0.01)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) blurry RGB in [0,1]. H, W divisible by 4.
        Returns:
            out_full  (B, 3, H, W)   <- inference output
            out_half  (B, 3, H/2, W/2)  <- auxiliary (training only)
            out_qtr   (B, 3, H/4, W/4)  <- auxiliary (training only)
        """
        import torch.nn.functional as F
        x_half = F.avg_pool2d(x, 2)
        x_qtr  = F.avg_pool2d(x_half, 2)

        e0, e0d = self.enc0(x)
        e1, e1d = self.enc1(torch.cat([e0d, x_half], dim=1))
        e2, _   = self.enc2(torch.cat([e1d, x_qtr],  dim=1))

        d2, out_qtr  = self.dec2(e2)
        d1, out_half = self.dec1(torch.cat([d2, e1], dim=1))
        _,  out_full = self.dec0(torch.cat([d1, e0], dim=1))

        out_full = out_full + x
        out_half = out_half + x_half
        out_qtr  = out_qtr  + x_qtr

        return out_full.clamp(0, 1), out_half.clamp(0, 1), out_qtr.clamp(0, 1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_mimo(
    n_feats: int = 32,
    n_res:   int = 8,
    device: Optional[torch.device] = None,
) -> MIMO:
    """Build the MIMO-UNet DB backbone (RGB, multi-scale)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MIMO(n_feats=n_feats, n_res=n_res).to(device)
    print(f"[model] MIMO (DB) | n_feats={n_feats} | n_res={n_res} | "
          f"params={model.num_parameters:,} | device={device}")
    return model


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DN: ResNet
    net_dn = build_model(n_feats=64, n_resblocks=16, in_channels=1, device=dev)
    net_dn.eval()
    y_dn   = net_dn(torch.rand(2, 1, 64, 64).to(dev))
    assert y_dn.shape == (2, 1, 64, 64) and 0 <= y_dn.min().item() <= y_dn.max().item() <= 1
    print(f"[ResNet/DN]  out={y_dn.shape}  range=[{y_dn.min():.3f},{y_dn.max():.3f}]")

    # SR: EDSR x4
    net_sr = build_edsr(scale=4, n_feats=64, n_resblocks=16, device=dev)
    y_sr   = net_sr(torch.rand(2, 3, 32, 32).to(dev))
    assert y_sr.shape == (2, 3, 128, 128)
    print(f"[EDSR/SR x4] out={y_sr.shape}  range=[{y_sr.min():.3f},{y_sr.max():.3f}]")

    # SR: EDSR x2
    net_sr2 = build_edsr(scale=2, n_feats=64, n_resblocks=16, device=dev)
    y_sr2   = net_sr2(torch.rand(2, 3, 32, 32).to(dev))
    assert y_sr2.shape == (2, 3, 64, 64)
    print(f"[EDSR/SR x2] out={y_sr2.shape}")

    # DB: MIMO
    net_db = build_mimo(n_feats=32, n_res=8, device=dev)
    y_full, y_half, y_qtr = net_db(torch.rand(2, 3, 64, 64).to(dev))
    assert y_full.shape == (2, 3, 64, 64)
    assert y_half.shape == (2, 3, 32, 32)
    assert y_qtr.shape  == (2, 3, 16, 16)
    print(f"[MIMO/DB]    full={y_full.shape}  half={y_half.shape}  qtr={y_qtr.shape}")

    print("\nmodel.py smoke test PASSED")
