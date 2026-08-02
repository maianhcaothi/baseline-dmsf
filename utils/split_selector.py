"""
Latency-optimal split point selection (Algorithm 1, DMSF paper).

Minimises total latency:
  T_total(p) = T_edge(p) + T_trans(p) + T_cloud(p)
  T_trans(p) = S_p / B

where:
  T_edge(p)  = cumulative edge inference time up to layer p
  T_cloud(p) = remaining backbone + GSC + FPN + Head time on cloud
  S_p        = bytes transmitted at split point p (after 1-bit quantization + LFC stats)
  B          = current bandwidth (bytes/s)

Usage:
  profiler = LayerProfiler(model, device)
  edge_times, cloud_times = profiler.profile()
  selector = SplitSelector(edge_times, cloud_times, feature_sizes)
  p = selector.select(bandwidth_bps=10e6)
"""

import time
import torch
import numpy as np
from typing import Dict


# Feature map channels at each split point
SPLIT_CHANNELS     = {3: 128, 5: 256, 7: 512, 10: 256}   # YOLOv5s
SPLIT_CHANNELS_26N = {3: 64,  5: 128, 7: 256, 10: 256}   # YOLO26n

# Spatial dims at 640×640 input (same for both variants)
SPLIT_SPATIAL = {3: (80, 80), 5: (40, 40), 7: (20, 20), 10: (20, 20)}


def feature_bytes_1bit(split_point: int, batch_size: int = 1,
                        reduction: int = 4,
                        channels: dict = None) -> int:
    """
    Bytes transmitted at split point p after CR compression:
      1-bit quantized feature: C//r * H * W * (1/8) bytes per sample
      LFC statistics (μ, σ): C * 4 bytes each (float32) per sample
    Pass channels=SPLIT_CHANNELS_26N for DMSF26n.
    """
    ch = channels if channels is not None else SPLIT_CHANNELS
    c = ch[split_point]
    h, w = SPLIT_SPATIAL[split_point]
    c_comp = c // reduction
    bits = c_comp * h * w          # 1-bit per element
    bytes_feat = (bits + 7) // 8   # ceil to bytes
    bytes_stats = 2 * c * 4        # μ and σ, float32
    return (bytes_feat + bytes_stats) * batch_size


# --------------------------------------------------------------------------- #
# Layer-level profiler
# --------------------------------------------------------------------------- #
class LayerProfiler:
    def __init__(self, model, device, warmup=5, runs=20):
        self.model = model
        self.device = device
        self.warmup = warmup
        self.runs = runs

    @torch.no_grad()
    def profile(self, img_size=640, batch_size=1):
        """
        Returns:
          edge_times  : {split_point: ms}  — edge-side cumulative time (0..p)
          cloud_times : {split_point: ms}  — cloud-side time (p+1..end)
        """
        m = self.model.eval().to(self.device)
        dummy = torch.randn(batch_size, 3, img_size, img_size, device=self.device)

        # Profile each segment using torch.cuda.Event or perf_counter
        use_cuda = self.device.type != 'cpu' if hasattr(self.device, 'type') else str(self.device) != 'cpu'

        def _measure(fn):
            for _ in range(self.warmup):
                fn()
            times = []
            for _ in range(self.runs):
                if use_cuda:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                else:
                    t0 = time.perf_counter()
                    fn()
                    times.append((time.perf_counter() - t0) * 1000)
            return float(np.mean(times))

        # Map backbone layers (b10 exists in DMSF26n)
        backbone_layers = [m.b0, m.b1, m.b2, m.b3, m.b4,
                           m.b5, m.b6, m.b7, m.b8, m.b9]
        if hasattr(m, 'b10'):
            backbone_layers.append(m.b10)

        edge_times = {}
        for sp in sorted(SPLIT_CHANNELS.keys()):
            def _edge_fn(stop=sp):
                x = dummy
                for i in range(stop + 1):
                    x = backbone_layers[i](x)
            edge_times[sp] = _measure(_edge_fn)

        cloud_times = {}
        for sp in sorted(SPLIT_CHANNELS.keys()):
            # Use dummy tensor matching the recovered feature shape
            c = SPLIT_CHANNELS[sp]
            h, w = SPLIT_SPATIAL[sp]
            dummy_feat = torch.randn(batch_size, c, h, w, device=self.device)

            def _cloud_fn(start=sp + 1, feat=dummy_feat):
                x = m._backbone_from(feat, start)
                p3, p4 = m.gsc(x)
                m._neck_head(x, p3, p4)

            cloud_times[sp] = _measure(_cloud_fn)

        return edge_times, cloud_times


# --------------------------------------------------------------------------- #
# Split point selector
# --------------------------------------------------------------------------- #
class SplitSelector:
    def __init__(self, edge_times: Dict[int, float],
                 cloud_times: Dict[int, float],
                 batch_size: int = 1, cr_reduction: int = 4,
                 channels: dict = None):
        self.edge_times  = edge_times
        self.cloud_times = cloud_times
        self.batch_size  = batch_size
        self.cr_reduction = cr_reduction
        self.channels    = channels  # None → YOLOv5s, SPLIT_CHANNELS_26N → DMSF26n

    def transmission_ms(self, split_point: int, bandwidth_bps: float) -> float:
        """bandwidth_bps: bits per second."""
        nbytes = feature_bytes_1bit(split_point, self.batch_size,
                                    self.cr_reduction, self.channels)
        return (nbytes * 8 / bandwidth_bps) * 1000   # ms

    def total_latency_ms(self, split_point: int, bandwidth_bps: float) -> float:
        return (self.edge_times[split_point] +
                self.transmission_ms(split_point, bandwidth_bps) +
                self.cloud_times[split_point])

    def select(self, bandwidth_bps: float) -> int:
        """Return the split point that minimises end-to-end latency."""
        best_p, best_t = None, float('inf')
        for p in sorted(self.edge_times):
            t = self.total_latency_ms(p, bandwidth_bps)
            if t < best_t:
                best_t = t
                best_p = p
        return best_p

    def report(self, bandwidth_bps: float):
        mbps = bandwidth_bps / 1e6
        print(f"\n{'Split':>6}  {'Edge(ms)':>10}  {'Trans(ms)':>10}  "
              f"{'Cloud(ms)':>10}  {'Total(ms)':>10}")
        print("-" * 52)
        for p in sorted(self.edge_times):
            e = self.edge_times[p]
            tr = self.transmission_ms(p, bandwidth_bps)
            c = self.cloud_times[p]
            mark = " ← optimal" if p == self.select(bandwidth_bps) else ""
            print(f"{p:>6}  {e:>10.2f}  {tr:>10.2f}  {c:>10.2f}  {e+tr+c:>10.2f}{mark}")
        print(f"\nBandwidth: {mbps:.1f} Mbps")
