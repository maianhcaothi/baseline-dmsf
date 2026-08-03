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

BASE = Path(r"d:\SplitInference\DMSF\baseline-dmsf")
OUT = Path(r"d:\SplitInference\DMSF\results\synthetic-selftest")

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
    R.append_event(str(OUT / "events_ns.log"), t0 - 3_100_000_000, CLUSTER_ID,
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

    # --- the real shutdown writers -------------------------------------------
    R.write_rate_summary(str(OUT / "fps_cluster.log"), s._fps_start_t,
                         s._cluster_times, BATCH)
    for r in sorted(records, key=lambda x: (x["role"], x["client_id"])):
        R.append_utilization_device(str(OUT / "utilization.log"), r)
    R.write_utilization_cluster(str(OUT / "utilization_cluster.log"), records)
    R.write_latency_cluster(str(OUT / "latency_cluster.log"), records)

    (OUT / "config.yaml").write_text(
        "# SYNTHETIC self-test fixture — not a real run.\n"
        "server:\n  batch-size: 32\n  clients: [3, 2]\ndmsf:\n  split-point: 7\n",
        encoding="utf-8")

    for name in R.RESULT_FILES:
        p = OUT / name
        print(f"  {name:<26} {p.stat().st_size:>8} bytes  "
              f"{sum(1 for _ in p.open()):>5} lines")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
