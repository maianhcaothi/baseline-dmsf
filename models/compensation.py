"""
Global Structure Compensation (GSC) — Section III-C of DMSF paper.

DMSF removes the FPN's skip connections from backbone layers 4 (128ch, 80×80)
and 6 (256ch, 40×40).  Instead, GSC reconstructs those two feature maps from
the backbone's final output (layer 9, 512ch, 20×20) using cascaded upsampling,
making the model transmit only ONE feature map regardless of split point.

Upsampling building blocks (as described in the paper):
  • PixelShuffleUpsample  — 2× upsample via channel reorganisation (71% cheaper
                            than deconvolution, replaces Conv inversions)
  • DeBottleneck          — inverse of C3's Bottleneck (transposed convs + skip)
  • DeMaxPool             — inverse of MaxPool via transposed conv

GSC architecture:
  backbone_out (512ch, 20×20)
    └─ PSUp(512→256) + DeBottleneck  →  gsc_p4 (256ch, 40×40)   [replaces layer-6 skip]
        └─ PSUp(256→128) + DeBottleneck  →  gsc_p3 (128ch, 80×80) [replaces layer-4 skip]
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# PixelShuffleUpsample: channel halved, spatial doubled
# --------------------------------------------------------------------------- #
class PixelShuffleUpsample(nn.Module):
    """
    1×1 conv: in_ch → out_ch*4, then PixelShuffle(2) → (out_ch, 2H, 2W).
    71% fewer FLOPs than transposed convolution for the same 2× upsample.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 1, bias=False)
        self.ps = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.ps(self.conv(x))))


# --------------------------------------------------------------------------- #
# DeBottleneck: inverse mapping of Bottleneck (transposed convs + residual)
# --------------------------------------------------------------------------- #
class DeBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c_ = channels // 2
        self.cv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )
        self.cv2 = nn.Sequential(
            nn.ConvTranspose2d(c_, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


# --------------------------------------------------------------------------- #
# GSC: reconstruct P3 & P4 skip features from final backbone output
# --------------------------------------------------------------------------- #
class GlobalStructureCompensation(nn.Module):
    """
    Input : backbone output — layer 9 output (512ch, H/32)
    Output: (gsc_p3, gsc_p4)
      gsc_p4 : 256ch, H/16   (replaces layer-6 skip to FPN)
      gsc_p3 : 128ch, H/8    (replaces layer-4 skip to FPN)

    Used on the cloud side after the edge->cloud handoff, so no dependency on
    the original backbone intermediate tensors.
    """

    def __init__(self):
        super().__init__()
        # 512ch 20×20 → 256ch 40×40
        self.up_p4 = PixelShuffleUpsample(512, 256)
        self.de_p4 = DeBottleneck(256)
        # 256ch 40×40 → 128ch 80×80
        self.up_p3 = PixelShuffleUpsample(256, 128)
        self.de_p3 = DeBottleneck(128)

    def forward(self, backbone_out: torch.Tensor):
        """
        backbone_out: (B, 512, H/32, W/32)
        Returns:
          gsc_p3: (B, 128, H/8,  W/8)
          gsc_p4: (B, 256, H/16, W/16)
        """
        gsc_p4 = self.de_p4(self.up_p4(backbone_out))   # 256ch, ×2
        gsc_p3 = self.de_p3(self.up_p3(gsc_p4))          # 128ch, ×4
        return gsc_p3, gsc_p4


# --------------------------------------------------------------------------- #
# GSC for YOLO26n (scale n, width=0.25)
# --------------------------------------------------------------------------- #
class GlobalStructureCompensation26n(nn.Module):
    """
    GSC adapted for YOLO26n.

    FPN skips in YOLO26n come from:
      layer 4  →  128ch, H/8   (P3 skip, head layer 15: Concat)
      layer 6  →  128ch, H/16  (P4 skip, head layer 12: Concat)

    We reconstruct both from backbone_final = b10 output (256ch, H/32).

    Input : backbone_final (B, 256, H/32, W/32)
    Output: (gsc_p3, gsc_p4)
      gsc_p4 : (B, 128, H/16, W/16)  replaces layer-6 skip
      gsc_p3 : (B, 128, H/8,  W/8)   replaces layer-4 skip
    """

    def __init__(self):
        super().__init__()
        self.up_p4 = PixelShuffleUpsample(256, 128)   # 256ch H/32 → 128ch H/16
        self.de_p4 = DeBottleneck(128)
        self.up_p3 = PixelShuffleUpsample(128, 128)   # 128ch H/16 → 128ch H/8
        self.de_p3 = DeBottleneck(128)

    def forward(self, backbone_out: torch.Tensor):
        """
        backbone_out: (B, 256, H/32, W/32)
        Returns:
          gsc_p3: (B, 128, H/8,  W/8)
          gsc_p4: (B, 128, H/16, W/16)
        """
        gsc_p4 = self.de_p4(self.up_p4(backbone_out))   # 128ch, ×2
        gsc_p3 = self.de_p3(self.up_p3(gsc_p4))          # 128ch, ×4
        return gsc_p3, gsc_p4
