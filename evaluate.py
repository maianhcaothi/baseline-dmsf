"""
Evaluate a trained DMSF checkpoint on VisDrone val set.

Runs evaluation at every candidate split point (3, 5, 7, 10) plus
edge-only and cloud-only baselines.  Reports mAP@50 and mAP@50:95
per split point, end-to-end latency, and transmitted bytes.

Usage:
  python evaluate.py --weights runs/train/best.pt --data data/visdrone
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.dmsf import DMSF
from utils.visdrone import VisDroneDataset
from utils.metrics import evaluate
from utils.split_selector import LayerProfiler, SplitSelector, feature_bytes_1bit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',  required=True)
    p.add_argument('--data',     default='data/visdrone')
    p.add_argument('--imgsz',    type=int, default=640)
    p.add_argument('--batch',    type=int, default=16)
    p.add_argument('--nc',       type=int, default=10)
    p.add_argument('--conf',     type=float, default=0.001)
    p.add_argument('--iou',      type=float, default=0.6)
    p.add_argument('--device',   default='')
    p.add_argument('--bandwidth',type=float, default=10.0, help='Mbps for latency report')
    p.add_argument('--workers',  type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # ── Model ────────────────────────────────────────────────────────────────
    model = DMSF(nc=args.nc).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict(),
                          strict=False)
    print(f"Loaded weights: {args.weights}")

    # ── Val loader ───────────────────────────────────────────────────────────
    val_ds = VisDroneDataset(
        str(Path(args.data) / 'images' / 'val'),
        img_size=args.imgsz, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=VisDroneDataset.collate_fn)

    # ── Layer profiling ───────────────────────────────────────────────────────
    print("\nProfiling layer times …")
    profiler = LayerProfiler(model, device)
    edge_times, cloud_times = profiler.profile(img_size=args.imgsz, batch_size=1)
    selector = SplitSelector(edge_times, cloud_times, batch_size=1)
    selector.report(args.bandwidth * 1e6)

    # ── Evaluate per split point ─────────────────────────────────────────────
    results = {}
    for sp in [3, 5, 7, 10]:
        print(f"\n--- Split point {sp} ---")
        t0 = time.perf_counter()
        m = evaluate(model, val_loader, device,
                     split_point=sp,
                     conf_thres=args.conf,
                     iou_thres=args.iou,
                     img_size=args.imgsz)
        elapsed = time.perf_counter() - t0
        nbytes = feature_bytes_1bit(sp, batch_size=1)
        results[sp] = {**m, 'elapsed_s': elapsed, 'feat_bytes': nbytes}
        print(f"  mAP@50={m['map50']:.4f}  mAP@50:95={m['map50_95']:.4f}  "
              f"feat={nbytes/1024:.1f} KB  time={elapsed:.1f}s")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Split':>6}  {'mAP@50':>8}  {'mAP@50:95':>10}  {'Feat(KB)':>9}  {'Total(ms)':>10}")
    print("-" * 70)
    bw = args.bandwidth * 1e6
    for sp, r in results.items():
        total_ms = (edge_times[sp] +
                    selector.transmission_ms(sp, bw) +
                    cloud_times[sp])
        print(f"{sp:>6}  {r['map50']:>8.4f}  {r['map50_95']:>10.4f}  "
              f"{r['feat_bytes']/1024:>9.1f}  {total_ms:>10.2f}")
    print("=" * 70)

    # Mark optimal
    best_sp = selector.select(bw)
    print(f"\nOptimal split at {args.bandwidth} Mbps: split_point={best_sp}")
    print(f"  → mAP@50={results[best_sp]['map50']:.4f}")


if __name__ == '__main__':
    main()
