"""
DMSF — Dynamic Model Splitting Framework (GLOBECOM 2025).

Full YOLOv5s modified with:
  • SwitchableCRModule after each backbone split candidate (layers 3,5,7,10)
  • GlobalStructureCompensation replacing FPN skip connections from backbone
  • LocalFeatureCompensation (inside RecoverModule) for distribution alignment

Three usage modes
─────────────────
  Training  : DMSF.forward(x, split_point)
                Random split point each step. Compress+recover is fully
                differentiable via STE. GSC replaces backbone skip to FPN.

  Edge      : DMSF.forward_edge(x, split_point)  →  payload dict
  Cloud     : DMSF.forward_cloud(payload)         →  detection output

Candidate split points: 3, 5, 7, 10  (layer indices in YOLOv5s)
"""

import math
import random
import torch
import torch.nn as nn

from .common import Conv, C3, SPPF, Concat, Detect
from .compress_recover import build_cr_modules, SPLIT_CHANNELS
from .compensation import GlobalStructureCompensation


# --------------------------------------------------------------------------- #
# Default YOLOv5s anchors (COCO, stride 8/16/32)
# --------------------------------------------------------------------------- #
ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]


class DMSF(nn.Module):
    def __init__(self, nc: int = 80, anchors=ANCHORS, cr_reduction: int = 4):
        super().__init__()
        self.nc = nc
        self.split_points = sorted(SPLIT_CHANNELS.keys())   # [3, 5, 7, 10]

        # ── Backbone (layers 0–9) ─────────────────────────────────────────────
        self.b0  = Conv(3,   32,  6, 2, 2)   # stem, 320×320
        self.b1  = Conv(32,  64,  3, 2)      # 160×160
        self.b2  = C3(64,   64,  1)          # 160×160
        self.b3  = Conv(64,  128, 3, 2)      # 80×80   ← split 3
        self.b4  = C3(128,  128, 2)          # 80×80   (P3 skip for FPN)
        self.b5  = Conv(128, 256, 3, 2)      # 40×40   ← split 5
        self.b6  = C3(256,  256, 3)          # 40×40   (P4 skip for FPN)
        self.b7  = Conv(256, 512, 3, 2)      # 20×20   ← split 7
        self.b8  = C3(512,  512, 1)          # 20×20
        self.b9  = SPPF(512, 512)            # 20×20   backbone end

        # ── Neck (layers 10–23) ───────────────────────────────────────────────
        self.n10 = Conv(512, 256, 1, 1)      # 20×20   ← split 10
        self.n11 = nn.Upsample(scale_factor=2, mode='nearest')
        # n12 = Concat([n11, p4_skip])        → handled in forward
        self.n13 = C3(512, 256, 1, False)
        self.n14 = Conv(256, 128, 1, 1)
        self.n15 = nn.Upsample(scale_factor=2, mode='nearest')
        # n16 = Concat([n15, p3_skip])        → handled in forward
        self.n17 = C3(256, 128, 1, False)    # P3 detect output

        self.n18 = Conv(128, 128, 3, 2)
        # n19 = Concat([n18, n14])            → handled in forward
        self.n20 = C3(256, 256, 1, False)    # P4 detect output

        self.n21 = Conv(256, 256, 3, 2)
        # n22 = Concat([n21, n10])            → handled in forward
        self.n23 = C3(512, 512, 1, False)    # P5 detect output

        # ── Detection head ───────────────────────────────────────────────────
        self.detect = Detect(nc, anchors, ch=[128, 256, 512])
        self._init_detect_strides()

        # ── DMSF modules ─────────────────────────────────────────────────────
        self.cr = build_cr_modules(cr_reduction)   # one per split candidate
        self.gsc = GlobalStructureCompensation()   # replaces backbone skips
        # For sp=10: x_rec is 256ch (n10 output); GSC needs 512ch backbone_final.
        # This 1×1 conv reconstructs a pseudo backbone_final for GSC.
        self.n10_recover = Conv(256, 512, 1, 1)

    # ----------------------------------------------------------------------- #
    def _init_detect_strides(self):
        self.detect.stride = torch.tensor([8., 16., 32.])
        # anchors must be grid-unit (pixel/stride) for ComputeLoss matching
        # anchor_grid keeps pixel units for the eval-mode wh output
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        for m in self.detect.m:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    # ----------------------------------------------------------------------- #
    # Backbone helpers
    # ----------------------------------------------------------------------- #
    def _backbone_to(self, x: torch.Tensor, stop: int):
        """Run backbone layers 0..stop (inclusive). Returns (output, saved)."""
        layers = [self.b0, self.b1, self.b2, self.b3, self.b4,
                  self.b5, self.b6, self.b7, self.b8, self.b9]
        for i in range(min(stop + 1, len(layers))):
            x = layers[i](x)
        return x

    def _backbone_from(self, x: torch.Tensor, start: int):
        """Run backbone layers start..9 (inclusive). Returns final output."""
        layers = [self.b0, self.b1, self.b2, self.b3, self.b4,
                  self.b5, self.b6, self.b7, self.b8, self.b9]
        for i in range(start, 10):
            x = layers[i](x)
        return x

    def _neck_head(self, backbone_final, gsc_p3, gsc_p4):
        """FPN neck + Detect head using GSC skip features."""
        n10 = self.n10(backbone_final)            # 256ch, 20×20

        up = self.n11(n10)                        # 256ch, 40×40
        cat = torch.cat([up, gsc_p4], dim=1)      # 512ch, 40×40
        n13 = self.n13(cat)                        # 256ch, 40×40
        n14 = self.n14(n13)                       # 128ch, 40×40

        up2 = self.n15(n14)                       # 128ch, 80×80
        cat2 = torch.cat([up2, gsc_p3], dim=1)    # 256ch, 80×80
        p3 = self.n17(cat2)                        # 128ch, 80×80

        d = self.n18(p3)                          # 128ch, 40×40
        cat3 = torch.cat([d, n14], dim=1)         # 256ch, 40×40
        p4 = self.n20(cat3)                        # 256ch, 40×40

        d2 = self.n21(p4)                         # 256ch, 20×20
        cat4 = torch.cat([d2, n10], dim=1)        # 512ch, 20×20
        p5 = self.n23(cat4)                        # 512ch, 20×20

        return self.detect([p3, p4, p5])

    # ----------------------------------------------------------------------- #
    # Training forward — full differentiable pipeline
    # ----------------------------------------------------------------------- #
    def _edge_features(self, x: torch.Tensor, split_point: int):
        """Compute edge-side features for any split point."""
        if split_point == 10:
            # sp=10 splits after n10 (first neck layer), so edge runs b0..b9 + n10
            return self.n10(self._backbone_to(x, 9))
        return self._backbone_to(x, split_point)

    def _cloud_forward(self, x_rec: torch.Tensor, split_point: int):
        """Run cloud-side computation (backbone tail + GSC + neck + head)."""
        if split_point == 10:
            # x_rec is 256ch (n10 output); reconstruct pseudo backbone_final for GSC
            pseudo_bf = self.n10_recover(x_rec)
            gsc_p3, gsc_p4 = self.gsc(pseudo_bf)
            return self._neck_head_sp10(x_rec, gsc_p3, gsc_p4)
        backbone_final = self._backbone_from(x_rec, split_point + 1)
        gsc_p3, gsc_p4 = self.gsc(backbone_final)
        return self._neck_head(backbone_final, gsc_p3, gsc_p4)

    def _neck_head_sp10(self, x_n10, gsc_p3, gsc_p4):
        """Neck + head when cloud receives x_n10 (n10 output, 256ch) directly."""
        up = self.n11(x_n10)
        cat = torch.cat([up, gsc_p4], dim=1)
        n13 = self.n13(cat)
        n14 = self.n14(n13)
        up2 = self.n15(n14)
        cat2 = torch.cat([up2, gsc_p3], dim=1)
        p3 = self.n17(cat2)
        d = self.n18(p3)
        cat3 = torch.cat([d, n14], dim=1)
        p4 = self.n20(cat3)
        d2 = self.n21(p4)
        cat4 = torch.cat([d2, x_n10], dim=1)   # x_n10 plays role of n10 skip
        p5 = self.n23(cat4)
        return self.detect([p3, p4, p5])

    def forward(self, x: torch.Tensor, split_point: int = None):
        """
        split_point: one of [3, 5, 7, 10], or None to pick randomly.
        During training the loss is computed on the final detection output.
        """
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
        """
        Returns a payload dict to be transmitted to the cloud:
          x_bit : quantized feature tensor
          mu    : per-channel mean of original features
          sigma : per-channel std of original features
          split : split point index
        """
        self.eval()
        x_edge = self._edge_features(x, split_point)
        cr = self.cr[str(split_point)]
        x_bit, mu, sigma = cr.compress(x_edge)
        return {
            "x_bit": x_bit.cpu(),
            "mu":    mu.cpu(),
            "sigma": sigma.cpu(),
            "split": split_point,
        }

    # ----------------------------------------------------------------------- #
    # Cloud-side inference
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def forward_cloud(self, payload: dict, device: str = "cpu"):
        """
        Receives payload from edge, runs recovery + backbone tail + GSC + head.
        """
        self.eval()
        split_point = payload["split"]
        x_bit  = payload["x_bit"].to(device)
        mu     = payload["mu"].to(device)
        sigma  = payload["sigma"].to(device)

        cr = self.cr[str(split_point)]
        x_rec = cr.recover(x_bit, mu, sigma)
        return self._cloud_forward(x_rec, split_point)

    # ----------------------------------------------------------------------- #
    # Convenience: load pretrained YOLOv5s weights into backbone/neck/head
    # (loads only matching keys, skips DMSF-specific layers)
    # ----------------------------------------------------------------------- #
    def load_yolov5s_weights(self, ckpt_path: str):
        import sys, types
        import torch.nn as nn

        # YOLOv5 checkpoints pickle the full model object; we need stub classes
        # that are proper nn.Module subclasses so __setstate__ restores parameters.
        class _NNStub(nn.Module):
            def __init__(self, *a, **kw): super().__init__()
            def forward(self, *a, **kw): pass

        class _StubNS(types.ModuleType):
            """Returns an nn.Module subclass for any non-dunder attribute lookup."""
            def __getattr__(self, name):
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                cls = type(name, (_NNStub,), {'__module__': self.__name__})
                setattr(self, name, cls)
                return cls

        for ns in ('models', 'models.yolo', 'models.common', 'models.experimental',
                   'utils', 'utils.general', 'utils.torch_utils',
                   'utils.activations', 'utils.autoanchor'):
            if ns not in sys.modules or not isinstance(sys.modules[ns], _StubNS):
                sys.modules[ns] = _StubNS(ns)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        # Remap: yolov5 "model.{i}.*" → our "b{i}.*" / "n{i}.*" / "detect.*"
        mapping = {}
        for i in range(10):
            mapping[f"model.model.{i}."] = f"b{i}."
        for i in range(10, 24):
            mapping[f"model.model.{i}."] = f"n{i}."
        mapping["model.model.24."] = "detect."
        new_state = {}
        for k, v in state.items():
            new_k = k
            for old, new in mapping.items():
                if k.startswith(old):
                    new_k = new + k[len(old):]
                    break
            new_state[new_k] = v
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        print(f"[DMSF] Loaded pretrained weights. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return self
