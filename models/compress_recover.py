"""
Switchable Compress-Recover modules (Section III-B of DMSF paper).

Edge side:
  1. Conv 2×2: channel reduction (÷4), spatial dims preserved via asymmetric pad
  2. Sign(): 1-bit quantization {-1, +1}  — STE for training gradients
  3. Record μ, σ of original features for Local Feature Compensation

Cloud side:
  1. ConvTranspose 2×2: channel recovery (×4), spatial dims preserved via crop
  2. AdaIN (LFC): align recovered features to original μ, σ

Spatial preservation:
  Compress: F.pad(x, (0,1,0,1)) + Conv2d(k=2,p=0) → (H+1-2+1)=(H) ✓
  Recover:  ConvTranspose2d(k=2,p=0) → (H+2-1)=(H+1), then [:,:,:-1,:-1] → H ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Straight-Through Estimator for Sign quantization
# --------------------------------------------------------------------------- #
class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


def sign_ste(x):
    return _SignSTE.apply(x)


# --------------------------------------------------------------------------- #
# Edge-side compression
# --------------------------------------------------------------------------- #
class CompressModule(nn.Module):
    """
    Edge side at split point p:
      x (C, H, W) → channel compress → 1-bit quantize → x_bit (C//r, H, W)
      Also returns (μ, σ) of x for LFC.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        assert in_channels % reduction == 0
        c_out = in_channels // reduction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c_out, 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )
        self.reduction = reduction

    def forward(self, x: torch.Tensor):
        # Statistics of original features (per-channel mean & std)
        mu = x.mean(dim=[2, 3], keepdim=True)       # (B, C, 1, 1)
        sigma = x.std(dim=[2, 3], keepdim=True) + 1e-5

        x_comp = self.conv(F.pad(x, (0, 1, 0, 1)))  # pad→(H+1,W+1), conv2×2→(H,W)
        x_bit = sign_ste(x_comp)                     # 1-bit quantized

        return x_bit, mu, sigma                      # all on original device


# --------------------------------------------------------------------------- #
# Cloud-side recovery
# --------------------------------------------------------------------------- #
class RecoverModule(nn.Module):
    """
    Cloud side at split point p:
      x_bit (C//r, H, W) + (μ, σ) → channel recover → LFC → x_rec (C, H, W)
    """

    def __init__(self, out_channels: int, reduction: int = 4):
        super().__init__()
        c_in = out_channels // reduction
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c_in, out_channels, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.lfc = LocalFeatureCompensation()

    def forward(self, x_bit: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        x_rec = self.deconv(x_bit)[:, :, :-1, :-1]  # deconv2×2→(H+1,W+1), crop→(H,W)
        x_norm = self.lfc(x_rec, mu, sigma)
        return x_norm


# --------------------------------------------------------------------------- #
# Local Feature Compensation — AdaIN (Eq. 6)
# --------------------------------------------------------------------------- #
class LocalFeatureCompensation(nn.Module):
    """
    F_norm = σ(F_p) * (F_rec - μ(F_rec)) / σ(F_rec) + μ(F_p)
    Aligns channel-wise mean/variance of recovered features to original.
    """

    def forward(self, x_rec: torch.Tensor, mu_orig: torch.Tensor, sigma_orig: torch.Tensor):
        mu_rec = x_rec.mean(dim=[2, 3], keepdim=True)
        sigma_rec = x_rec.std(dim=[2, 3], keepdim=True) + 1e-5
        return sigma_orig * (x_rec - mu_rec) / sigma_rec + mu_orig


# --------------------------------------------------------------------------- #
# Switchable module: wraps CompressModule + RecoverModule for one split point
# --------------------------------------------------------------------------- #
class SwitchableCRModule(nn.Module):
    """
    Used during training: simulates the full compress→transmit→recover pipeline
    in a single differentiable forward pass (STE keeps gradients flowing).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.compress = CompressModule(channels, reduction)
        self.recover = RecoverModule(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bit, mu, sigma = self.compress(x)
        return self.recover(x_bit, mu, sigma)

    # ---- Separate edge/cloud calls for real deployment ---- #
    def forward_edge(self, x: torch.Tensor):
        """Returns (x_bit, mu, sigma) to be transmitted."""
        return self.compress(x)

    def forward_cloud(self, x_bit: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        """Reconstructs features on the cloud side."""
        return self.recover(x_bit, mu, sigma)


# --------------------------------------------------------------------------- #
# Channel dims for each split candidate in YOLOv5s (width=0.5)
# --------------------------------------------------------------------------- #
SPLIT_CHANNELS = {3: 128, 5: 256, 7: 512, 10: 256}


def build_cr_modules(reduction: int = 4) -> nn.ModuleDict:
    """Build one SwitchableCRModule per candidate split point (YOLOv5s)."""
    return nn.ModuleDict({
        str(p): SwitchableCRModule(c, reduction)
        for p, c in SPLIT_CHANNELS.items()
    })


# --------------------------------------------------------------------------- #
# YOLO26n (scale n, width=0.25) split-point channel map
# --------------------------------------------------------------------------- #
SPLIT_CHANNELS_26N = {3: 64, 5: 128, 7: 256, 10: 256}


def build_cr_modules_26n(reduction: int = 4) -> nn.ModuleDict:
    """Build one SwitchableCRModule per candidate split point (YOLO26n)."""
    return nn.ModuleDict({
        str(p): SwitchableCRModule(c, reduction)
        for p, c in SPLIT_CHANNELS_26N.items()
    })
