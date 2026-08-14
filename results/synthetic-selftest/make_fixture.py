"""Drive the REAL DmsfServer / DmsfResults writers with synthetic device reports.

This is a format harness, not a result. It exists to prove three things without
a broker, a GPU or a video:

  1. the server's on_fps callback emits both live series consistently,
  2. every shutdown writer in DmsfResults produces conformant lines,
  3. the notebook renders those lines end to end.

The numbers are invented. Anything it writes is labelled SYNTHETIC.
"""
import os
import random
import sys
import types
from pathlib import Path

# Anchored to this file, never to a machine — same rule as build_nb.py. The
# project has already been moved once, and a hardcoded root survives that move
# silently: mkdir(parents=True) happily creates the stale path, so the fixture
# regenerates into a directory nothing reads while the real one goes stale.
OUT = Path(__file__).resolve().parent          # this script lives IN the fixture
BASE = OUT.parents[1]                          # results/synthetic-selftest -> repo root

# ---- stub the deps this box does not have; none are used on the paths we call
class _Any(types.ModuleType):
    """Permissive stand-in: any attribute resolves to a no-op callable/object."""

    def __getattr__(self, item):
        def _noop(*a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]                    # used as a bare decorator
            return _Any(f"{self.__name__}.{item}")
        _noop.__enter__ = lambda *a: None
        _noop.__exit__ = lambda *a: None
        return _noop

    def __call__(self, *a, **k):
        return self


for name in ("pika", "torch", "cv2", "psutil"):
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = _Any(name)
for sub in ("torch.nn", "torch.nn.functional"):
    if sub not in sys.modules and isinstance(sys.modules.get("torch"), _Any):
        sys.modules[sub] = _Any(sub)

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))

import DmsfResults as R                      # noqa: E402
import DmsfFreeTime as FT                     # noqa: E402
import DmsfMapEval as ME                      # noqa: E402
from DmsfScheduler import CLUSTER_ID          # noqa: E402
from DmsfServer import DmsfServer             # noqa: E402

rng = random.Random(20260802)
BATCH = 32
N_EDGE, N_CLOUD = 3, 2
N_UNITS = 430


class _Method:
    delivery_tag = 0


class _Ch:
    def basic_ack(self, delivery_tag=None):
        pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R.truncate_all(str(OUT))

    # --- a bare server, wired only with the state on_fps touches --------------
    s = object.__new__(DmsfServer)
    s._fps_times, s._cluster_times = [], {}
    s._last_done_t = None
    s._fps_window = R.WINDOW
    s._batch_size = BATCH
    s._batch_log_path = str(OUT / "batch_done_ns.log")
    s._rate_ns_log_path = str(OUT / "fps_cluster_ns.log")

    # --- replay arrivals through the real callback ---------------------------
    t0 = 1785127877_000000000
    s._fps_start_t = t0 / 1e9 - 3.1          # shared START, before the first arrival
    t_ns = t0
    ch, method = _Ch(), _Method()
    import DmsfServer as srv
    real_time_ns = srv.time.time_ns
    for i in range(N_UNITS):
        # ~11 fps of 32-frame batches, with a slow stretch in the middle
        gap = 2.90 + 0.35 * rng.random()
        if 150 < i < 210:
            gap += 0.45
        t_ns += int(gap * 1e9)
        srv.time.time_ns = (lambda v: (lambda: v))(t_ns)
        s.on_fps(ch, method, None, CLUSTER_ID.encode())
    srv.time.time_ns = real_time_ns

    # --- one control-plane decision, at dispatch -----------------------------
    R.append_event(str(OUT / "cut_change_ns.log"), t0 - 3_100_000_000, CLUSTER_ID,
                   "split point set to 7 (auto)")

    # --- synthetic device reports --------------------------------------------
    records = []
    for role, n_dev, svc_mean, util in (("edge", N_EDGE, 11_600.0, 0.47),
                                        ("cloud", N_CLOUD, 3_100.0, 0.35)):
        for d in range(n_dev):
            pkgs = N_UNITS // n_dev + (1 if d < N_UNITS % n_dev else 0)
            jitter = 1.0 + 0.06 * (d - (n_dev - 1) / 2)
            service = [max(1.0, rng.gauss(svc_mean * jitter, svc_mean * 0.16))
                       for _ in range(pkgs)]
            busy_ns = int(sum(service) * 1e6)
            rec = {
                "client_id": f"{role}-{d}-0000-0000-{d:012d}",
                "role": role, "cluster": CLUSTER_ID, "packages": pkgs,
                "busy_ns": busy_ns,
                "total_ns": int(busy_ns / (util * jitter)),
                "utilization": util * jitter,
                "service_ms": service,
                # in-stage residency contains service; the edge buffers a whole
                # batch of frames before the unit exists, the cloud barely any
                "pipeline_ms": [v * (1.55 if role == "edge" else 1.02) for v in service],
                "e2e_ms": ([max(1.0, rng.gauss(96_600, 14_000)) for _ in range(pkgs)]
                           if role == "cloud" else []),
            }
            records.append(rec)

    # --- synthetic free-time reports -----------------------------------------
    # Built through the REAL FreeTimeTracker, one work span and one wait span
    # per unit, so the interval algebra (union, complement, priority-ordered
    # attribution) is what produces the numbers rather than the fixture.
    ft_reports = []
    for rec in records:
        tracker = FT.FreeTimeTracker(
            enabled=True, cluster=CLUSTER_ID, client_id=rec["client_id"],
            role=rec["role"], device="cpu", bucket_s=1.0, log_path=str(OUT))
        tracker.start()
        # Replay the device's life on the tracker's own monotonic clock.
        tracker._mono0 = 0
        tracker._epoch0 = t0 + (0 if rec["role"] == "edge" else 3_000_000_000)
        cursor = 0
        for sample_ms in rec["service_ms"]:
            busy_ns = int(sample_ms * 1e6)
            idle_ns = int(busy_ns * (0.35 if rec["role"] == "edge" else 0.08))
            tracker.add_work("inference", cursor, cursor + busy_ns)
            tracker.add_wait("input", cursor + busy_ns, cursor + busy_ns + idle_ns)
            cursor += busy_ns + idle_ns
        tracker._mono1 = cursor
        report = tracker.report()
        report["arrival_ns"] = t_ns + 1_000_000
        ft_reports.append(report)

    # --- synthetic broker-RAM samples ----------------------------------------
    ram, base = [], 1600.0
    for i in range(240):
        phase = "idle" if i < 30 else ("run" if i < 225 else "tail")
        load = 0.0 if phase == "idle" else (420.0 if phase == "run" else 120.0)
        ram.append({"ts_ns": t0 - 30_000_000_000 + i * 1_000_000_000,
                    "phase": phase, "source": "ssh", "total_mb": 5921.5,
                    "used_mb": base + load + 40 * rng.random(),
                    "avail_mb": 4000.0, "rss_mb": 90.0 + load / 3,
                    "swap_used_mb": 1032.3})

    # --- one measured worker's payload sizes ---------------------------------
    sizes = [int(rng.gauss(39_000_000, 400_000)) for _ in range(N_UNITS)]
    msg_report = {
        "client_id": records[0]["client_id"], "role": "edge",
        "cluster": CLUSTER_ID, "arrival_ns": t_ns + 1_000_000,
        "context": {"mode": "split", "split_point": 7, "compress": "on",
                    "num_bit": 1, "machine": "machine-2"},
        "n": len(sizes), "total_bytes": sum(sizes),
        "mean_bytes": sum(sizes) / len(sizes),
        "p50_bytes": sorted(sizes)[len(sizes) // 2],
        "p95_bytes": sorted(sizes)[int(len(sizes) * 0.95)],
        "max_bytes": max(sizes), "min_bytes": min(sizes),
        "span_s": N_UNITS * 3.0,
        "series": [(i, i * 3.0, i, v) for i, v in enumerate(sizes)],
    }

    # --- the real shutdown writers -------------------------------------------
    R.write_rate_summary(str(OUT / "fps_cluster.log"), s._fps_start_t,
                         s._cluster_times, BATCH)
    for r in sorted(records, key=lambda x: (x["role"], x["client_id"])):
        R.append_utilization_device(str(OUT / "utilization.log"), r)
    R.write_utilization_cluster(str(OUT / "utilization_cluster.log"), records)
    R.write_latency_cluster(str(OUT / "latency_cluster.log"), records)
    R.write_free_time(str(OUT / "free_time.log"), ft_reports)
    R.write_free_time_cluster(str(OUT / "free_time_cluster.log"), ft_reports,
                              server_host="synthetic-controller",
                              server_host_idle=0.97)
    R.write_free_time_series(str(OUT / "free_time_series.log"), ft_reports)
    R.write_broker_ram(str(OUT / "broker_ram.log"), ram,
                       {"host": "192.168.101.91", "source": "ssh",
                        "interval_s": 1.0, "reason": ""})
    with open(OUT / "broker_ram_ns.log", "w") as f:
        for sample in ram:
            f.write(f"{sample['ts_ns']} host=192.168.101.91 source=ssh "
                    f"phase={sample['phase']} total_mb={sample['total_mb']:.1f} "
                    f"used_mb={sample['used_mb']:.1f} "
                    f"used={sample['used_mb'] / sample['total_mb'] * 100:.2f}% "
                    f"avail_mb={sample['avail_mb']:.1f} "
                    f"swap_used_mb={sample['swap_used_mb']:.1f} "
                    f"rabbit_rss_mb={sample['rss_mb']:.1f}\n")
    R.write_message_size(str(OUT / "message_size.log"), [msg_report], BATCH)
    R.write_message_size_series(str(OUT / "message_size_series.log"), [msg_report])

    # --- accuracy: the two files that are NOT part of the contract -----------
    # Scored through the real metric, on invented boxes: a jittered copy of the
    # "truth" is the prediction, so the score is a real computation over a fake
    # detection set.
    gts, preds = {}, {}
    for frame in range(1, 16 * BATCH * 2 + 1):
        boxes = []
        for _ in range(6):
            cx, cy = 0.1 + 0.8 * rng.random(), 0.1 + 0.8 * rng.random()
            boxes.append((rng.randrange(3), cx, cy, 0.08, 0.06))
        gts[frame] = [(c, 1.0, (cx - w / 2) * 640, (cy - h / 2) * 640,
                       (cx + w / 2) * 640, (cy + h / 2) * 640)
                      for c, cx, cy, w, h in boxes]
        preds[frame] = [(c, 0.3 + 0.7 * rng.random(),
                         (cx - w / 2) * 640 + rng.gauss(0, 6),
                         (cy - h / 2) * 640 + rng.gauss(0, 6),
                         (cx + w / 2) * 640 + rng.gauss(0, 6),
                         (cy + h / 2) * 640 + rng.gauss(0, 6))
                        for c, cx, cy, w, h in boxes]
    per_cluster = {CLUSTER_ID: ME.evaluate_cluster(preds, gts, BATCH, 16)}
    map_ts = t_ns + 2_000_000
    R.write_map(str(OUT / "map.log"), map_ts, per_cluster, 16)
    R.write_map_window(str(OUT / "map_window.log"), map_ts, per_cluster)

    (OUT / "config.yaml").write_text(
        "# SYNTHETIC self-test fixture — not a real run.\n"
        "server:\n  batch-size: 32\n  clients: [3, 2]\ndmsf:\n  split-point: 7\n",
        encoding="utf-8")

    for name in R.RESULT_FILES + R.MAP_FILES:
        p = OUT / name
        print(f"  {name:<26} {p.stat().st_size:>8} bytes  "
              f"{sum(1 for _ in p.open()):>5} lines")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
