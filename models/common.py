"""
YOLOv5s building blocks.
Architecture (width=0.5, depth=0.33, input 640×640):
  Layer  Module   out_ch  spatial
  0      Conv     32      320×320   stem
  1      Conv     64      160×160
  2      C3(n=1)  64      160×160
  3      Conv     128     80×80    ← split candidate P3
  4      C3(n=2)  128     80×80    → FPN P3 skip (to neck layer 16)
  5      Conv     256     40×40    ← split candidate P4
  6      C3(n=3)  256     40×40    → FPN P4 skip (to neck layer 12)
  7      Conv     512     20×20    ← split candidate P5
  8      C3(n=1)  512     20×20
  9      SPPF     512     20×20    backbone end
  10     Conv     256     20×20    ← split candidate (neck entry)
  11     Upsample         40×40
  12     Concat          512      cat with layer 6 (or GSC_p4)
  13     C3(n=1)  256     40×40
  14     Conv     128     40×40
  15     Upsample         80×80
  16     Concat          256      cat with layer 4 (or GSC_p3)
  17     C3(n=1)  128     80×80    P3 detect output
  18     Conv     128     40×40
  19     Concat          256      cat with layer 14
  20     C3(n=1)  256     40×40    P4 detect output
  21     Conv     256     20×20
  22     Concat          512      cat with layer 10
  23     C3(n=1)  512     20×20    P5 detect output
  24     Detect   nc             from [17,20,23]
"""

import torch
import torch.nn as nn


def autopad(k, p=None):
    return k // 2 if p is None else p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    stride = None

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                xy = (y[..., :2] * 2 - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat([xy, wh, y[..., 4:]], dim=-1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()


# --------------------------------------------------------------------------- #
# YOLO26n / YOLO11 building blocks
# --------------------------------------------------------------------------- #
class _BnC2f(nn.Module):
    """Bottleneck with 3×3 kernels, e=0.5 — inner block of C3k2 when c3k=False.
    Matches ultralytics C2f Bottleneck: cv1=3×3, cv2=3×3."""

    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        c_ = c2 // 2                      # e = 0.5
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class _BnC3k(nn.Module):
    """Bottleneck with configurable kernel — inner block of C3k."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3):
        super().__init__()
        self.cv1 = Conv(c1, c2, k, 1, g=g)
        self.cv2 = Conv(c2, c2, k, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    """C3-style block with configurable bottleneck kernel — inner block of C3k2
    when c3k=True.  Matches ultralytics key layout: cv1, cv2, cv3, m."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(_BnC3k(c_, c_, shortcut, g, k) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))


class C3k2(nn.Module):
    """C2f-style block used in YOLO26 / YOLO11 backbone and neck.

    Matches ultralytics C3k2 key structure exactly:
      c3k=False → inner block is _BnC2f  (cv1=3×3, cv2=3×3, e=0.5)
      c3k=True  → inner block is C3k     (cv1,cv2,cv3,m — C3-style)
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        if c3k:
            self.m = nn.ModuleList(C3k(c_, c_, 2, shortcut) for _ in range(n))
        else:
            self.m = nn.ModuleList(_BnC2f(c_, c_, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class _Attention(nn.Module):
    """Lightweight multi-head self-attention over spatial feature maps (C×H×W)."""

    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, 3*C, H, W) → (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(0)                              # each: (B, h, N, d)
        attn = (q @ k.transpose(-2, -1)) * self.scale        # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out)


class PSABlock(nn.Module):
    """Position-Sensitive Attention block (attention + FFN with residuals)."""

    def __init__(self, c, num_heads=1):
        super().__init__()
        self.attn = _Attention(c, num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C2PSA(nn.Module):
    """Cross-Stage Partial with PSA attention — YOLO26 layer 10 (backbone end)."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.ModuleList(PSABlock(c_) for _ in range(n))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        for m in self.m:
            b = m(b)
        return self.cv2(torch.cat([a, b], 1))
