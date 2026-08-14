"""Free time — implements ``guide/10-free-time.md`` for the DMSF baseline.

Free time is the wall clock in which a device did **nothing at all**: no frame
capture, no inference, no serialize, no publish, no receive, no post-process, no
bookkeeping. It is a property of the *device*, not of a stage, and it is neither
``utilization`` nor ``1 - utilization``:

* utilization (`guide/03`) measures ``busy / total`` over each unit's
  ``get input -> output`` window. A wait *inside* that window counts as busy,
  and work done *outside* it counts as nothing.
* on this project's edge, decoding and resizing a whole batch of frames happens
  **before** ``get input``. Utilization cannot see it; free time counts it as
  busy. That gap is the entire reason this file exists here.

The two rules that make the numbers correct:

* **Merge, never sum.** ``busy`` is the *union* of every lane's work intervals.
  A sum can exceed the span outright once two lanes overlap, and it silently
  misses the gaps *between* stages, which is precisely where free time lives.
  The workers here are single-threaded today, so the union has one lane — but
  the union is what is computed, so a threaded worker later stays correct.
* **Attribute in a fixed priority.** Wait spans explain free time; they do not
  define it. Each reason claims only what busy time and higher-priority reasons
  have not, so the reasons sum to ``free_ns`` **exactly** and whatever no reason
  covers is reported as ``unaccounted`` rather than dropped.

Timing uses ``perf_counter_ns`` (monotonic — a clock step mid-run must not be
able to produce a negative interval) and is converted to epoch only on export,
where it is needed for one thing: unioning the device *processes* that share one
machine (`guide/10 §4`). That is the single place two processes' timestamps are
compared, and it is valid only because they share a clock.
"""

import os
import socket
import threading
import time

try:                                   # psutil is a hard dep of the run, but
    import psutil                      # free time must degrade, not raise
except Exception:                      # pragma: no cover
    psutil = None


#: Free-time attribution order, published because it is part of the definition
#: (guide/10 §3). Earlier reasons claim a contested instant; whatever no reason
#: covers becomes ``unaccounted``.
WAIT_PRIORITY = ("input", "backpressure", "downstream", "idle")

#: A 2 ms poll loop produces ~500 intervals per idle second. Extending the open
#: wait span instead of appending keeps a long stall as one interval. Work spans
#: are never coalesced with a tolerance — busy has to stay exact.
COALESCE_NS = 1_000_000

#: Bound the shipped series and interval list so a long run cannot turn one
#: device's report into a multi-megabyte message.
MAX_SERIES_POINTS = 3600
MAX_SHIPPED_INTERVALS = 2000


# ---------------------------------------------------------------------------- #
# Interval algebra — shared by the device (its own lanes) and the server (the
# processes on one machine). Every function takes and returns disjoint, sorted
# [start, stop) pairs.
# ---------------------------------------------------------------------------- #
def merge_intervals(spans):
    """Union of possibly-overlapping intervals."""
    out = []
    for pair in sorted(spans):
        a, b = pair[0], pair[1]
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return out


def measure(intervals):
    """Total length of a disjoint interval list."""
    return sum(b - a for a, b in intervals)


def clip(intervals, lo, hi):
    out = []
    for a, b in intervals:
        a, b = max(a, lo), min(b, hi)
        if b > a:
            out.append([a, b])
    return out


def complement(intervals, lo, hi):
    """Everything in [lo, hi) that ``intervals`` does not cover."""
    out, cursor = [], lo
    for a, b in intervals:
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        if a > cursor:
            out.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < hi:
        out.append([cursor, hi])
    return out


def intersect(xs, ys):
    """Intersection of two disjoint sorted interval lists."""
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        a = max(xs[i][0], ys[j][0])
        b = min(xs[i][1], ys[j][1])
        if b > a:
            out.append([a, b])
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(xs, ys):
    """Everything in ``xs`` that ``ys`` does not cover."""
    if not ys:
        return [list(x) for x in xs]
    out = []
    for a, b in xs:
        cursor = a
        for c, d in ys:
            if d <= cursor or c >= b:
                continue
            if c > cursor:
                out.append([cursor, min(c, b)])
            cursor = max(cursor, d)
            if cursor >= b:
                break
        if cursor < b:
            out.append([cursor, b])
    return out


def cap_intervals(intervals, limit=MAX_SHIPPED_INTERVALS):
    """Shrink an interval list to ``limit`` entries by closing the SMALLEST gaps.

    Biases the answer toward *less* free time — the safe direction — and returns
    the swallowed total so the report can state its own error bar as
    ``merge_slop_s`` (guide/10 §4).
    """
    if len(intervals) <= limit:
        return [list(x) for x in intervals], 0
    gaps = sorted(range(len(intervals) - 1),
                  key=lambda i: intervals[i + 1][0] - intervals[i][1])
    close = set(gaps[:len(intervals) - limit])
    out, slop = [], 0
    for i, (a, b) in enumerate(intervals):
        if out and (i - 1) in close:
            slop += a - out[-1][1]
            out[-1][1] = b
        else:
            out.append([a, b])
    return out, slop


def cpu_snapshot():
    if psutil is None:
        return None
    try:
        times = psutil.cpu_times()
        return float(times.idle), float(sum(times))
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
# Device side
# ---------------------------------------------------------------------------- #
class FreeTimeTracker:
    """Records what one device was doing, and reduces it once at the end.

    A disabled tracker is a no-op costing one attribute lookup per call, so the
    instrumentation can sit unconditionally in the hot path — and a disabled
    tracker reports **nothing** rather than reporting zeros, which would read as
    "this device was never idle" (guide/09 phase 4b).
    """

    def __init__(self, enabled=False, cluster="unknown", client_id="unknown",
                 role="unknown", device="cpu", bucket_s=1.0, log_path="."):
        self.enabled = bool(enabled)
        self.cluster = cluster
        self.client_id = str(client_id)
        self.role = role
        self.device = str(device)
        self.machine = socket.gethostname()
        self.bucket_ns = max(int(float(bucket_s) * 1e9), 1_000_000)
        self.log_path = log_path
        self._work = []           # (start_ns, stop_ns, kind, lane)
        self._wait = {}           # reason -> [start_ns, stop_ns] list
        self._mono0 = None
        self._mono1 = None
        self._epoch0 = None
        self._cpu0 = None
        self._warned = False

    # -- lifecycle ---------------------------------------------------------- #
    def now(self):
        return time.perf_counter_ns()

    def start(self):
        """Open the run window. Called where the timing log writes ``start``, so
        ``span_s`` here and ``total_s`` in utilization.log describe the same
        stretch of this device's life."""
        if not self.enabled:
            return
        self._mono0 = time.perf_counter_ns()
        self._epoch0 = time.time_ns()
        self._cpu0 = cpu_snapshot()

    def stop(self):
        """Close the run window, where the timing log writes ``end``."""
        if not self.enabled or self._mono0 is None:
            return
        self._mono1 = time.perf_counter_ns()

    # -- recording ---------------------------------------------------------- #
    def add_work(self, kind, t0, t1=None):
        """A real operation: it makes the device busy for [t0, t1)."""
        if not self.enabled or self._mono0 is None:
            return
        t1 = self.now() if t1 is None else t1
        if t1 > t0:
            # The lane is the thread, taken automatically: a lane a caller has
            # to name rots the moment a stage moves to a different thread.
            self._work.append((t0, t1, kind, threading.get_ident()))

    def add_wait(self, reason, t0, t1=None):
        """A block: it *explains* free time, it does not create it."""
        if not self.enabled or self._mono0 is None:
            return
        t1 = self.now() if t1 is None else t1
        if t1 <= t0:
            return
        spans = self._wait.setdefault(reason, [])
        if spans and t0 - spans[-1][1] <= COALESCE_NS:
            spans[-1][1] = max(spans[-1][1], t1)     # one long stall, one interval
        else:
            spans.append([t0, t1])

    # -- reduction ---------------------------------------------------------- #
    def report(self):
        """One device's whole answer, or ``None`` when there is nothing to say."""
        if not self.enabled or self._mono0 is None:
            return None
        if self._mono1 is None:
            self.stop()
        lo, hi = self._mono0, self._mono1
        if hi <= lo:
            return None
        span_ns = hi - lo

        busy = merge_intervals([(a, b) for a, b, _k, _l in self._work])
        busy = merge_intervals(clip(busy, lo, hi))
        busy_ns = measure(busy)
        free = complement(busy, lo, hi)
        free_ns = span_ns - busy_ns                  # exact by construction

        # Reasons partition the free time: each claims only what busy time and
        # the earlier reasons left, and the remainder is named, never dropped.
        reasons, remaining = {}, free
        for reason in WAIT_PRIORITY:
            spans = self._wait.get(reason)
            if not spans or not remaining:
                continue
            claimed = intersect(remaining, merge_intervals(spans))
            if claimed:
                reasons[reason] = measure(claimed)
                remaining = subtract(remaining, claimed)
        leftover = measure(remaining)
        if leftover > 0:
            reasons["unaccounted"] = leftover

        kinds, lanes = {}, {}
        for a, b, kind, lane in self._work:
            kinds[kind] = kinds.get(kind, 0) + (b - a)
            lanes.setdefault(lane, []).append((a, b))
        lanes = {str(k): measure(merge_intervals(v)) for k, v in lanes.items()}

        intervals, slop = cap_intervals(busy)
        offset = self._epoch0 - lo                   # mono -> epoch, this process
        epoch_intervals = [(a + offset, b + offset) for a, b in intervals]

        host_idle = None
        cpu1 = cpu_snapshot()
        if self._cpu0 and cpu1 and cpu1[1] > self._cpu0[1]:
            host_idle = (cpu1[0] - self._cpu0[0]) / (cpu1[1] - self._cpu0[1])
            host_idle = min(max(host_idle, 0.0), 1.0)

        return {
            "action": "FREE_TIME",
            "client_id": self.client_id,
            "role": self.role,
            "cluster": self.cluster,
            "machine": self.machine,
            "device": self.device,
            "span_ns": span_ns,
            "busy_ns": busy_ns,
            "free_ns": free_ns,
            "gaps": len(free),
            "longest_free_ns": max((b - a for a, b in free), default=0),
            "kinds": kinds,
            "reasons": reasons,
            "lanes": lanes,
            "intervals": epoch_intervals,
            "merge_slop_ns": slop,
            "epoch_start_ns": self._epoch0,
            "epoch_end_ns": self._epoch0 + span_ns,
            "host_idle": host_idle,
            "series": self._series(busy, lo, hi),
        }

    def _series(self, busy, lo, hi):
        """The plottable "when was it idle" curve, as free% per fixed bucket.

        ``bucket_ns`` travels with the series rather than being assumed, so a
        long run may widen its buckets to bound the file size without breaking
        any reader (guide/01 §3.10).
        """
        span = hi - lo
        bucket = self.bucket_ns
        if span // bucket + 1 > MAX_SERIES_POINTS:
            bucket = span // MAX_SERIES_POINTS + 1
        out, i, idx = [], lo, 0
        while i < hi:
            stop = min(i + bucket, hi)
            covered = 0
            while idx < len(busy) and busy[idx][1] <= i:
                idx += 1
            j = idx
            while j < len(busy) and busy[j][0] < stop:
                covered += min(busy[j][1], stop) - max(busy[j][0], i)
                j += 1
            width = stop - i
            out.append(max(0.0, 1.0 - covered / width) if width else 0.0)
            i = stop
        return {"bucket_ns": bucket, "free": out}

    # -- side-car ----------------------------------------------------------- #
    def write_local(self, report):
        """This device's own copy — ``free_time_<role>_<cluster>_<id>.log``.

        The roll-ups on the server already carry every number in here. This file
        exists because it survives a broker or server failure, and because it is
        the artifact you read when one device behaves differently from its peers
        (guide/10 §5). Failure to write it is a warning; it never touches the run.
        """
        if not report:
            return None
        tag = f"{self.role}_{_safe(self.cluster)}_{self.client_id.replace('-', '')[:12]}"
        path = os.path.join(self.log_path, f"free_time_{tag}.log")
        try:
            with open(path, "w") as f:
                f.write(f"# free time — device {self.client_id} role={self.role} "
                        f"machine={self.machine} cluster={self.cluster}\n")
                f.write(f"# busy is a UNION of lanes, never a sum; "
                        f"reasons are attributed in priority order "
                        f"{'>'.join(WAIT_PRIORITY)}>unaccounted\n")
                f.write(f"span_s={report['span_ns'] / 1e9:.3f} "
                        f"busy_s={report['busy_ns'] / 1e9:.3f} "
                        f"free_s={report['free_ns'] / 1e9:.3f} "
                        f"gaps={report['gaps']} "
                        f"longest_free_ms={report['longest_free_ns'] / 1e6:.3f}\n")
                for reason, ns in sorted(report["reasons"].items(),
                                         key=lambda kv: -kv[1]):
                    f.write(f"FREE reason={reason} free_s={ns / 1e9:.3f}\n")
                for kind, ns in sorted(report["kinds"].items(), key=lambda kv: -kv[1]):
                    f.write(f"KIND kind={kind} busy_s={ns / 1e9:.3f}\n")
                bucket_s = report["series"]["bucket_ns"] / 1e9
                for i, value in enumerate(report["series"]["free"]):
                    f.write(f"SERIES i={i} t_offset_s={i * bucket_s:.3f} "
                            f"bucket_s={bucket_s:.3f} free={value * 100:.2f}%\n")
            return path
        except Exception as e:
            if not self._warned:
                print(f"[FreeTime] Warning: could not write {path}: {e}")
                self._warned = True
            return None


def _safe(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))
