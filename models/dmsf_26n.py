"""
DMSF26n — DMSF applied to YOLO26n backbone (scale n, width=0.25).

Architecture channels with width_multiple=0.25, depth_multiple=0.5:

  Backbone:
    b0:  Conv(3→16,  3×3/2)                    H/2
    b1:  Conv(16→32, 3×3/2)                    H/4
    b2:  C3k2(32→64,  n=1, c3k=False, e=0.25)  H/4
    b3:  Conv(64→64,  3×3/2)                   H/8   ← split 3 (64ch)
    b4:  C3k2(64→128, n=1, c3k=False, e=0.25)  H/8
    b5:  Conv(128→128, 3×3/2)                  H/16  ← split 5 (128ch)
    b6:  C3k2(128→128, n=1, c3k=True)          H/16
    b7:  Conv(128→256, 3×3/2)                  H/32  ← split 7 (256ch)
    b8:  C3k2(256→256, n=1, c3k=True)          H/32
    b9:  SPPF(256→256)                         H/32
    b10: C2PSA(256→256, n=1)                   H/32  ← split 10 (256ch)

  Neck / FPN:
    n11: Upsample ×2                            H/16
    n13: C3k2(384→128, c3k=True)  [cat n11(256) + gsc_p4(128) = 384]  H/16
    n14: Upsample ×2                            H/8
    n16: C3k2(256→64,  c3k=True)  [cat n14(128) + gsc_p3(128) = 256]  H/8   P3
    n17: Conv(64→64, 3×3/2)                    H/16
    n19: C3k2(192→128, c3k=True)  [cat n17(64)  + n13(128)   = 192]  H/16  P4
    n20: Conv(128→128, 3×3/2)                  H/32
    n22: C3k2(384→256, c3k=True)  [cat n20(128) + b10(256)   = 384]  H/32  P5
    Detect([P3, P4, P5]), ch=[64, 128, 256]

Split point 10: edge transmits b10 output (256ch) = backbone_final directly,
so no recover-conv is needed (unlike YOLOv5s where sp=10 is a neck layer).

Compatible with the existing ComputeLoss and evaluate() from utils/, which
expect the YOLOv5s anchor-based Detect head format.
"""

import math
import random
import torch
import torch.nn as nn

from .common import Conv, Bottleneck, SPPF, Detect, C3k2, C2PSA
from .compress_recover import build_cr_modules_26n, SPLIT_CHANNELS_26N
from .compensation import GlobalStructureCompensation26n


# YOLOv5s COCO anchors reused — these will adapt to VisDrone during training.
ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]


class DMSF26n(nn.Module):
    def __init__(self, nc: int = 80, anchors=ANCHORS, cr_reduction: int = 4):
        super().__init__()
        self.nc = nc
        self.split_points = sorted(SPLIT_CHANNELS_26N.keys())   # [3, 5, 7, 10]

        # ── Backbone ─────────────────────────────────────────────────────────
        self.b0  = Conv(3,   16,  3, 2)                          # H/2
        self.b1  = Conv(16,  32,  3, 2)                          # H/4
        self.b2  = C3k2(32,  64,  1, c3k=False, e=0.25)          # H/4
        self.b3  = Conv(64,  64,  3, 2)                          # H/8   ← split 3
        self.b4  = C3k2(64,  128, 1, c3k=False, e=0.25)          # H/8
        self.b5  = Conv(128, 128, 3, 2)                          # H/16  ← split 5
        self.b6  = C3k2(128, 128, 1, c3k=True)                   # H/16
        self.b7  = Conv(128, 256, 3, 2)                          # H/32  ← split 7
        self.b8  = C3k2(256, 256, 1, c3k=True)                   # H/32
        self.b9  = SPPF(256, 256)                                # H/32
        self.b10 = C2PSA(256, 256, 1)                            # H/32  ← split 10

        # ── Neck (FPN) ────────────────────────────────────────────────────────
        self.n11 = nn.Upsample(scale_factor=2, mode='nearest')   # H/32→H/16
        # n12 = Concat([n11, gsc_p4])  256+128=384  ─ in forward()
        self.n13 = C3k2(384, 128, 1, c3k=True)                   # H/16

        self.n14 = nn.Upsample(scale_factor=2, mode='nearest')   # H/16→H/8
        # n15 = Concat([n14, gsc_p3])  128+128=256  ─ in forward()
        self.n16 = C3k2(256, 64,  1, c3k=True)                   # H/8  P3

        self.n17 = Conv(64,  64,  3, 2)                          # H/8→H/16
        # n18 = Concat([n17, n13])  64+128=192  ─ in forward()
        self.n19 = C3k2(192, 128, 1, c3k=True)                   # H/16 P4

        self.n20 = Conv(128, 128, 3, 2)                          # H/16→H/32
        # n21 = Concat([n20, b10])  128+256=384  ─ in forward()
        self.n22 = C3k2(384, 256, 1, c3k=True)                   # H/32 P5

        # ── Detection head ───────────────────────────────────────────────────
        self.detect = Detect(nc, anchors, ch=[64, 128, 256])
        self._init_detect_strides()

        # ── DMSF modules ─────────────────────────────────────────────────────
        self.cr  = build_cr_modules_26n(cr_reduction)
        self.gsc = GlobalStructureCompensation26n()

    # ----------------------------------------------------------------------- #
    def _init_detect_strides(self):
        self.detect.stride = torch.tensor([8., 16., 32.])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        for m in self.detect.m:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    # ----------------------------------------------------------------------- #
    # Backbone helpers
    # ----------------------------------------------------------------------- #
    def _backbone_to(self, x: torch.Tensor, stop: int) -> torch.Tensor:
        """Run backbone layers b0 .. b_{stop} (inclusive)."""
        layers = [self.b0, self.b1, self.b2, self.b3, self.b4,
                  self.b5, self.b6, self.b7, self.b8, self.b9, self.b10]
        for i in range(min(stop + 1, len(layers))):
            x = layers[i](x)
        return x

    def _backbone_from(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """Run backbone layers b_{start} .. b10 (inclusive)."""
        layers = [self.b0, self.b1, self.b2, self.b3, self.b4,
                  self.b5, self.b6, self.b7, self.b8, self.b9, self.b10]
        for i in range(start, 11):
            x = layers[i](x)
        return x

    # ----------------------------------------------------------------------- #
    # FPN neck + Detect head
    # ----------------------------------------------------------------------- #
    def _neck_head(self, backbone_final, gsc_p3, gsc_p4):
        """
        backbone_final : (B, 256, H/32)   ← b10 output
        gsc_p4         : (B, 128, H/16)   ← replaces layer-6 skip
        gsc_p3         : (B, 128, H/8)    ← replaces layer-4 skip
        """
        up  = self.n11(backbone_final)                       # 256ch, H/16
        n13 = self.n13(torch.cat([up, gsc_p4], dim=1))      # 128ch, H/16

        up2 = self.n14(n13)                                  # 128ch, H/8
        p3  = self.n16(torch.cat([up2, gsc_p3], dim=1))     # 64ch,  H/8

        d   = self.n17(p3)                                   # 64ch,  H/16
        p4  = self.n19(torch.cat([d, n13], dim=1))           # 128ch, H/16

        d2  = self.n20(p4)                                   # 128ch, H/32
        p5  = self.n22(torch.cat([d2, backbone_final],
                                  dim=1))                    # 256ch, H/32

        return self.detect([p3, p4, p5])

    # ----------------------------------------------------------------------- #
    # Shared cloud-side forward
    # ----------------------------------------------------------------------- #
    def _edge_features(self, x: torch.Tensor, split_point: int) -> torch.Tensor:
        return self._backbone_to(x, split_point)

    def _cloud_forward(self, x_rec: torch.Tensor, split_point: int):
        """
        For sp=10: x_rec IS backbone_final (b10 output, 256ch).
        For sp<10: run remaining backbone b_{sp+1}..b10 first.
        """
        if split_point == 10:
            backbone_final = x_rec
        else:
            backbone_final = self._backbone_from(x_rec, split_point + 1)
        gsc_p3, gsc_p4 = self.gsc(backbone_final)
        return self._neck_head(backbone_final, gsc_p3, gsc_p4)

    # ----------------------------------------------------------------------- #
    # Training forward (differentiable via STE)
    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, split_point: int = None):
        if split_point is None:
            split_point = random.choice(self.split_points)
        x_edge = self._edge_features(x, split_point)
        x_rec  = self.cr[str(split_point)](x_edge)
        return self._cloud_forward(x_rec, split_point)

    # ----------------------------------------------------------------------- #
    # Edge-side inference
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def forward_edge(self, x: torch.Tensor, split_point: int):
        self.eval()
        x_edge = self._edge_features(x, split_point)
        cr = self.cr[str(split_point)]
        x_bit, mu, sigma = cr.compress(x_edge)
        return {"x_bit": x_bit.cpu(), "mu": mu.cpu(),
                "sigma": sigma.cpu(), "split": split_point}

    # ----------------------------------------------------------------------- #
    # Cloud-side inference
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def forward_cloud(self, payload: dict, device: str = "cpu"):
        self.eval()
        split_point = payload["split"]
        x_bit  = payload["x_bit"].to(device)
        mu     = payload["mu"].to(device)
        sigma  = payload["sigma"].to(device)
        cr = self.cr[str(split_point)]
        x_rec = cr.recover(x_bit, mu, sigma)
        return self._cloud_forward(x_rec, split_point)

    # ----------------------------------------------------------------------- #
    # Load pretrained YOLO26n backbone weights (ultralytics checkpoint)
    # ----------------------------------------------------------------------- #
    def load_yolo26n_weights(self, ckpt_path: str):
        """
        Initialise backbone and neck from a pretrained ultralytics YOLO26n
        checkpoint (yolo26n.pt).  Only matching keys are loaded; DMSF-specific
        layers (cr, gsc, detect) stay randomly initialised.

        Requires ultralytics to be installed: pip install ultralytics
        """
        try:
            from ultralytics import YOLO as _UL
            state = _UL(ckpt_path).model.state_dict()
        except Exception:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt)
            if hasattr(state, 'state_dict'):
                state = state.state_dict()

        # backbone: model.0 → b0 … model.10 → b10
        mapping = {f'model.{i}.': f'b{i}.' for i in range(11)}
        # neck parameterised layers (skip Upsample/Concat layers 11,12,14,15,18,21)
        mapping.update({
            'model.13.': 'n13.',
            'model.16.': 'n16.',
            'model.17.': 'n17.',
            'model.19.': 'n19.',
            'model.20.': 'n20.',
            'model.22.': 'n22.',
        })

        new_state = {}
        for k, v in state.items():
            new_k = k
            for old, new in mapping.items():
                if k.startswith(old):
                    new_k = new + k[len(old):]
                    break
            new_state[new_k] = v

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        loaded = sum(1 for k in new_state if k not in
                     {m for m in missing})
        print(f'[DMSF26n] Pretrained weights: {loaded} loaded, '
              f'{len(missing)} missing, {len(unexpected)} unexpected')
        return self
