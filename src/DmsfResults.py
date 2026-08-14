"""Result-format writers for the DMSF baseline.

Implements the fourteen-file contract in ``guide/01-result-format.md`` using the
**cluster** naming scheme (one scheme per project — never mix in ``group_*``)::

    batch_done_ns.log        system throughput series       live      required
    fps_cluster_ns.log       per-cluster throughput series  live      required
    fps_cluster.log          throughput summary             shutdown  required
    utilization.log          per-device busy ratio          shutdown  required
    utilization_cluster.log  utilization rolled up          shutdown  required
    latency_cluster.log      latency distributions          shutdown  required
    cut_change_ns.log        control-plane events           live      optional
    free_time.log            per-device idle time           shutdown  optional
    free_time_cluster.log    free time rolled up, and why   shutdown  optional
    free_time_series.log     free time over the run         shutdown  optional
    broker_ram_ns.log        queue-host RAM series          live      optional
    broker_ram.log           queue-host RAM summary         shutdown  optional
    message_size.log         payload size summary           shutdown  optional
    message_size_series.log  payload size over the run      shutdown  optional

plus two files that are **not** part of that contract and must never stand in
for one of it: ``map.log`` and ``map_window.log`` (see ``DmsfMapEval``).

The invariants that shaped this module, so they survive later edits:

* **One clock.** Every timestamp written to these files is ``time.time_ns()`` on
  the server. Device-local timestamps never reach them — only ratios and
  durations computed entirely inside one device do.
* **Aggregate before you divide.** Throughput is total frames / total time,
  never the mean of per-interval rates.
* **Raw samples, not pre-reduced stats.** Devices ship latency arrays;
  percentiles are taken here, once, over the pooled population, nearest-rank.
* **Telemetry never kills the run.** Nothing in this module is load-bearing;
  every caller wraps it so a measurement failure loses a number, not the run.

Every writer takes already-collected records, so the format is testable without
a broker.
"""

import datetime
import glob
import math
import os
import shutil
import time

# Rolling-window size for the live throughput series. Charts assume 16.
WINDOW = 16

# The fourteen result files, in the order 01 §2 lists them. This tuple is also
# the archive manifest — 05 §4 wants an explicit list rather than "every
# non-empty file", so a stale log can never be stamped into a tagged archive.
RESULT_FILES = (
    "batch_done_ns.log",
    "fps_cluster_ns.log",
    "fps_cluster.log",
    "utilization.log",
    "utilization_cluster.log",
    "latency_cluster.log",
    "cut_change_ns.log",
    "free_time.log",
    "free_time_cluster.log",
    "free_time_series.log",
    "broker_ram_ns.log",
    "broker_ram.log",
    "message_size.log",
    "message_size_series.log",
)

# This project's own extension, outside the portable contract (00 §7): fixed
# names and the same line grammar, but the readers of the fourteen know nothing
# about them. Truncated and archived on the same terms; never counted as one of
# the fourteen.
MAP_FILES = (
    "map.log",
    "map_window.log",
)

# Scratch, not results (05 §5). These are per-device working files and a
# write-once lock; none of them belongs in an archive, and every one of them
# silently wins over the current run if a previous run left it behind.
SCRATCH_GLOBS = (
    "metrics_pivot.lock",      # a crashed run leaves it; _pivot_and_save then
                               # returns early forever and no pivot is written
    "metrics_raw_*.csv",       # per-device rows; leftovers get folded into the
                               # next run's pivot as if they were this run's
)

# Device-written side-cars (00 §6). Not results, but device ids change every
# run, so last run's file belongs to nothing this run — and an archiver that
# copied by glob would fold it into this run's results. Cleared centrally at
# startup for the same reason, and archived into their own subdirectory where
# they cannot be mistaken for results.
SIDECAR_GLOBS = (
    "free_time_*.log",
    "message_size_*.log",
)


# ---------------------------------------------------------------------------- #
# Lifecycle
# ---------------------------------------------------------------------------- #
def truncate_all(log_path):
    """Empty every file this run emits, once, centrally, before any worker writes.

    Unconditional by design, including the files of features that are switched
    off: a file truncated only when the feature that writes it is enabled leaks
    the previous run's data into this run's archive (05 §4), and the reader
    cannot tell. It is also what makes "the files exist even when empty" true —
    a missing file is a hard error for the reader, an empty one is a valid
    "this run had none".
    """
    os.makedirs(log_path, exist_ok=True)
    for name in RESULT_FILES + MAP_FILES:
        open(os.path.join(log_path, name), "w").close()


def purge_scratch(log_path):
    """Delete last run's scratch files, once, centrally, at startup — 05 §5.

    A write-once artifact that is never cleared "wins" forever: the pivot lock
    left behind by a crashed run makes every later run skip its own pivot, and
    leftover per-device CSVs get merged into the next run's summary as though
    they belonged to it. The numbers stay plausible, which is what makes this
    worth deleting rather than detecting.

    Centrally, in the component that starts **once** — doing it per worker lets
    a late starter delete files an earlier worker is already writing.

    Returns the number of files removed. Failure is never fatal.
    """
    removed = 0
    for pattern in SCRATCH_GLOBS + SIDECAR_GLOBS:
        for path in glob.glob(os.path.join(log_path, pattern)):
            # A side-car glob can match a RESULT file — `free_time_*.log` also
            # matches `free_time_cluster.log` and `free_time_series.log`, and
            # `message_size_*.log` matches `message_size_series.log`. Deleting
            # those here would undo the truncation that just created them, and a
            # run whose collector then times out would be MISSING a required-
            # shaped file instead of leaving an empty one, which is a hard error
            # for the reader (01 §2). Never let scratch cleanup touch a result.
            if os.path.basename(path) in RESULT_FILES + MAP_FILES:
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                print(f"[Startup] Warning: could not remove {path}: {e}")
    if removed:
        print(f"[Startup] Cleared {removed} leftover scratch file(s) from a previous run")
    return removed


# ---------------------------------------------------------------------------- #
# Live series
# ---------------------------------------------------------------------------- #
def append_batch_done(path, t_ns, window_fps=None):
    """One line per completed unit — 01 §3.1.

    Two arities on purpose: units 1..W-1 have no window yet and get one column.
    A padded ``0.00`` would be indistinguishable from a genuine stall.
    """
    with open(path, "a") as f:
        if window_fps is None:
            f.write(f"{t_ns}\n")
        else:
            f.write(f"{t_ns} {window_fps:.2f}\n")


def append_cluster_rate(path, t_ns, cluster, done, window_fps=None):
    """The same arrival as :func:`append_batch_done`, bucketed by cluster — 01 §3.2.

    Line count here MUST equal ``batch_done_ns.log``'s, so this is called on
    every arrival including the ones with no window yet.
    """
    line = f"{t_ns} cluster={cluster} done={done}"
    if window_fps is not None:
        line += f" window_fps={window_fps:.2f}"
    with open(path, "a") as f:
        f.write(line + "\n")


def append_event(path, t_ns, scope, description):
    """One control-plane decision — 01 §3.7.

    The timestamp is taken by the caller *before* the decision is broadcast, so
    it marks when the decision was made rather than when it landed. Each append
    opens and closes the file so no handle is shared across threads.
    """
    with open(path, "a") as f:
        f.write(f"{t_ns} {scope}: {description}\n")


# ---------------------------------------------------------------------------- #
# Statistics
# ---------------------------------------------------------------------------- #
def nearest_rank(sorted_samples, q):
    """Nearest-rank percentile, no interpolation. ``q`` in [0, 100].

    Every number printed is therefore a latency some unit actually experienced.
    An interpolated p95 is a value that never happened.
    """
    if not sorted_samples:
        return None
    k = max(1, math.ceil(q / 100.0 * len(sorted_samples)))
    return sorted_samples[k - 1]


def summarize_ms(samples_ms):
    """Pool raw samples and reduce once — 04 §1. Returns None on an empty pool.

    Percentiles cannot validly be averaged, so callers must concatenate every
    device's raw array before calling this, never pre-reduce per device.
    """
    s = sorted(float(x) for x in samples_ms)
    if not s:
        return None
    return {
        "n": len(s),
        "mean_ms": sum(s) / len(s),
        "p50_ms": nearest_rank(s, 50),
        "p95_ms": nearest_rank(s, 95),
        "max_ms": s[-1],
    }


# ---------------------------------------------------------------------------- #
# Shutdown: throughput summary
# ---------------------------------------------------------------------------- #
def write_rate_summary(path, start_t, cluster_times, unit_size, ts_ns=None):
    """``fps_cluster.log`` — one line per cluster, then one SYSTEM line (01 §3.3).

    ``cluster_times`` maps cluster id -> list of arrival times in seconds, on the
    server clock. ``start_t`` is the *shared* START, recorded once when work was
    dispatched; using a per-cluster start instead is what makes cluster rates
    stop relating to the system total.

    Returns the SYSTEM dict, or None when the run produced no completions (in
    which case the file is left empty, which 01 §2 permits).
    """
    ts_ns = ts_ns or time.time_ns()
    buckets = {c: sorted(t) for c, t in cluster_times.items() if t}
    total_done = sum(len(t) for t in buckets.values())
    if not total_done or start_t is None:
        return None

    all_last = max(t[-1] for t in buckets.values())
    lines, system = [], None

    for cluster in sorted(buckets):
        t = buckets[cluster]
        done = len(t)
        frames = done * unit_size
        # Shared START, this cluster's own end: that is what keeps the SYSTEM
        # span equal to the largest cluster span (the validator checks exactly
        # this, because it is the shared-START check).
        span = t[-1] - start_t
        fps = frames / span if span > 0 else 0.0
        line = f"{ts_ns} cluster={cluster} fps={fps:.3f}"
        # steady_fps drops warm-up and uses the cluster's OWN first completion,
        # or a cluster that started late is unfairly penalised. This is the fair
        # number for comparing clusters.
        if done >= 2 and t[-1] > t[0]:
            steady = (done - 1) * unit_size / (t[-1] - t[0])
            line += f" steady_fps={steady:.3f}"
        share = done / total_done * 100.0
        line += f" done={done} frames={frames} share={share:.1f}%"
        lines.append(line)

    total_frames = total_done * unit_size
    sys_span = all_last - start_t
    sys_fps = total_frames / sys_span if sys_span > 0 else 0.0
    # The SYSTEM line carries neither steady_fps nor share — both are per-scope
    # readings that have no system-wide meaning.
    lines.append(f"{ts_ns} SYSTEM fps={sys_fps:.3f} done={total_done} "
                 f"frames={total_frames} clusters={len(buckets)}")
    system = {"fps": sys_fps, "done": total_done, "frames": total_frames,
              "clusters": len(buckets), "span_s": sys_span}

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return system


# ---------------------------------------------------------------------------- #
# Shutdown: utilization
# ---------------------------------------------------------------------------- #
def append_utilization_device(path, record, ts_ns=None):
    """One line per device, appended as its report is drained — 01 §3.4.

    Column 1 is the report's **arrival** at the server, not a device timestamp.
    ``busy_s`` and ``total_s`` both come from the reporting device's own clock,
    so the ratio is immune to clock skew — but the two numbers are not
    comparable against any other device's.
    """
    ts_ns = ts_ns or record.get("arrival_ns") or time.time_ns()
    busy_s = record.get("busy_ns", 0) / 1e9
    total_s = record.get("total_ns", 0) / 1e9
    util = record.get("utilization", 0.0) * 100.0
    with open(path, "a") as f:
        f.write(f"{ts_ns} client={record.get('client_id', 'unknown')} "
                f"role={record.get('role', 'unknown'):<5s} "
                f"packages={record.get('packages', 0)} "
                f"busy_s={busy_s:.3f} total_s={total_s:.3f} "
                f"utilization={util:.2f}%\n")


def _pooled_and_mean(records):
    """Pooled Σbusy/Σtotal, and the plain mean of per-device ratios — 03 §6.

    Both are emitted because a pooled figure can hide one idle device inside a
    busy cluster. When the two diverge, the cluster is imbalanced; that
    divergence is the whole reason the second column exists.
    """
    busy = sum(r.get("busy_ns", 0) for r in records)
    total = sum(r.get("total_ns", 0) for r in records)
    pooled = busy / total if total else 0.0
    mean = (sum(r.get("utilization", 0.0) for r in records) / len(records)
            if records else 0.0)
    return busy, total, pooled, mean


def write_utilization_cluster(path, records, ts_ns=None):
    """``utilization_cluster.log`` — three line kinds in one file (01 §3.5).

    ``cluster=<c> ALL``      every device in that cluster, pooled + mean
    ``cluster=<c> role=<r>`` devices of one role in one cluster, pooled
    ``SYSTEM``               every device, pooled + mean

    This file groups; it does not replace ``utilization.log``.
    """
    ts_ns = ts_ns or time.time_ns()
    if not records:
        return
    by_cluster = {}
    for r in records:
        by_cluster.setdefault(r.get("cluster", "unknown"), []).append(r)

    lines = []
    for cluster in sorted(by_cluster):
        devs = by_cluster[cluster]
        busy, total, pooled, mean = _pooled_and_mean(devs)
        lines.append(f"{ts_ns} cluster={cluster} ALL devices={len(devs)} "
                     f"utilization={pooled * 100:.2f}% "
                     f"utilization_mean={mean * 100:.2f}% "
                     f"busy_s={busy / 1e9:.3f} total_s={total / 1e9:.3f} "
                     f"packages={sum(d.get('packages', 0) for d in devs)}")
        by_role = {}
        for d in devs:
            by_role.setdefault(d.get("role", "unknown"), []).append(d)
        for role in sorted(by_role):
            rdevs = by_role[role]
            rbusy, rtotal, rpooled, _ = _pooled_and_mean(rdevs)
            lines.append(f"{ts_ns} cluster={cluster} role={role} "
                         f"devices={len(rdevs)} utilization={rpooled * 100:.2f}% "
                         f"busy_s={rbusy / 1e9:.3f} total_s={rtotal / 1e9:.3f} "
                         f"packages={sum(d.get('packages', 0) for d in rdevs)}")

    busy, total, pooled, mean = _pooled_and_mean(records)
    lines.append(f"{ts_ns} SYSTEM devices={len(records)} clusters={len(by_cluster)} "
                 f"utilization={pooled * 100:.2f}% "
                 f"utilization_mean={mean * 100:.2f}% "
                 f"busy_s={busy / 1e9:.3f} total_s={total / 1e9:.3f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- #
# Shutdown: latency
# ---------------------------------------------------------------------------- #
def write_latency_cluster(path, records, ts_ns=None):
    """``latency_cluster.log`` — pooled distributions per scope (01 §3.6).

    Three kinds, which must not be confused:

    ``service``   the device's own ``get input -> output``. One clock, exact.
                  Its samples sum to the matching ``busy_s``, which makes it the
                  only latency directly comparable against utilization.
    ``pipeline``  in-stage residency: everything from the unit entering the
                  stage to it leaving. One clock, exact. Contains ``service``,
                  so ``pipeline >> service`` means buffering, not a slow device.
    ``e2e``       edge start -> cloud output. **Two machines**, so it inherits
                  any offset between their clocks — indicative, not exact.
                  Reported per cluster and for SYSTEM, never per role.
    """
    ts_ns = ts_ns or time.time_ns()
    if not records:
        return

    by_cluster_role = {}
    by_cluster_e2e = {}
    system_e2e = []
    for r in records:
        cluster = r.get("cluster", "unknown")
        role = r.get("role", "unknown")
        for kind in ("service_ms", "pipeline_ms"):
            samples = r.get(kind) or []
            if samples:
                key = (cluster, role, kind[:-3])
                by_cluster_role.setdefault(key, []).extend(samples)
        e2e = r.get("e2e_ms") or []
        if e2e:
            by_cluster_e2e.setdefault(cluster, []).extend(e2e)
            system_e2e.extend(e2e)

    lines = []
    for (cluster, role, kind) in sorted(by_cluster_role):
        st = summarize_ms(by_cluster_role[(cluster, role, kind)])
        lines.append(f"{ts_ns} cluster={cluster} role={role} kind={kind:<8s} "
                     f"n={st['n']} mean_ms={st['mean_ms']:.3f} "
                     f"p50_ms={st['p50_ms']:.3f} p95_ms={st['p95_ms']:.3f} "
                     f"max_ms={st['max_ms']:.3f}")
    # role= is absent on e2e lines: e2e is not a property of one role, and
    # carrying it would double-count in any chart that groups by role.
    for cluster in sorted(by_cluster_e2e):
        st = summarize_ms(by_cluster_e2e[cluster])
        lines.append(f"{ts_ns} cluster={cluster} kind=e2e "
                     f"n={st['n']} mean_ms={st['mean_ms']:.3f} "
                     f"p50_ms={st['p50_ms']:.3f} p95_ms={st['p95_ms']:.3f} "
                     f"max_ms={st['max_ms']:.3f}")
    if system_e2e:
        st = summarize_ms(system_e2e)
        lines.append(f"{ts_ns} SYSTEM kind=e2e "
                     f"n={st['n']} mean_ms={st['mean_ms']:.3f} "
                     f"p50_ms={st['p50_ms']:.3f} p95_ms={st['p95_ms']:.3f} "
                     f"max_ms={st['max_ms']:.3f}")

    if lines:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- #
# Shutdown: free time (01 §3.8-3.10, method in guide 10)
# ---------------------------------------------------------------------------- #
def _pct(part, whole):
    """A ratio as a percentage, clamped to 100.

    The clamp is not cosmetic: a value above 100% is how the validator catches
    overlapping intervals that were summed instead of merged. Float noise on an
    exact 100% would raise that alarm falsely, and anything genuinely above it
    is a bug the merge is supposed to make impossible.
    """
    if not whole:
        return 0.0
    return min(part / whole * 100.0, 100.0)


def _write_lines(path, lines):
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    return lines


def write_free_time(path, records, ts_ns=None):
    """``free_time.log`` — one line per device (01 §3.8).

    ``busy_s`` is the **union** of that device's lanes, never a sum, so
    ``busy_s + free_s == span_s`` holds exactly. ``free`` and ``utilization``
    measure different things and must not be expected to sum to 100%.
    """
    ts_ns = ts_ns or time.time_ns()
    lines = []
    for r in sorted(records, key=lambda x: (x.get("role", ""),
                                            str(x.get("client_id", "")))):
        span_s = r.get("span_ns", 0) / 1e9
        busy_s = r.get("busy_ns", 0) / 1e9
        free_s = r.get("free_ns", 0) / 1e9
        line = (f"{r.get('arrival_ns') or ts_ns} "
                f"client={r.get('client_id', 'unknown')} "
                f"role={r.get('role', 'unknown')} "
                f"machine={r.get('machine', 'unknown')} "
                f"cluster={r.get('cluster', 'unknown')} "
                f"device={r.get('device', 'unknown')} "
                f"span_s={span_s:.3f} busy_s={busy_s:.3f} free_s={free_s:.3f} "
                f"free={_pct(r.get('free_ns', 0), r.get('span_ns', 0)):.2f}% "
                f"gaps={r.get('gaps', 0)} "
                f"longest_free_ms={r.get('longest_free_ns', 0) / 1e6:.3f}")
        if r.get("host_idle") is not None:
            line += f" host_idle={min(r['host_idle'] * 100.0, 100.0):.2f}%"
        lines.append(line)
    return _write_lines(path, lines)


def _free_scope(ts_ns, prefix, records):
    """``free`` (pooled) and ``free_mean`` for one scope, plus its two breakdowns.

    Pooled weights each device by how long it ran; the mean is the plain average
    of the ratios. Both are emitted because a pooled figure can hide one idle
    device inside a busy cluster — the divergence is the signal.
    """
    span = sum(r.get("span_ns", 0) for r in records)
    free = sum(r.get("free_ns", 0) for r in records)
    mean = (sum(_pct(r.get("free_ns", 0), r.get("span_ns", 0)) for r in records)
            / len(records)) if records else 0.0
    return (f"{ts_ns} {prefix} devices={len(records)} "
            f"free={_pct(free, span):.2f}% free_mean={min(mean, 100.0):.2f}% "
            f"free_s={free / 1e9:.3f} span_s={span / 1e9:.3f}")


def _sum_maps(records, key):
    out = {}
    for r in records:
        for name, ns in (r.get(key) or {}).items():
            out[name] = out.get(name, 0) + ns
    return out


def write_free_time_cluster(path, records, ts_ns=None, server_host=None,
                            server_host_idle=None):
    """``free_time_cluster.log`` — six line kinds in one file (01 §3.9).

    ``MACHINE`` lines come from the **union of the busy intervals** of the
    device processes on that host, never from averaging their percentages: two
    devices that are each 50% free can keep a machine 100% busy by interleaving.
    This is the one place two processes' timestamps are compared, and it is
    valid only because processes on one host share a clock — intervals are never
    unioned across machines.

    ``FREE reason=`` shares sum to 100% of the scope's free time (whatever no
    reason covers is reported as ``unaccounted``). ``KIND`` shares may sum past
    100%: per-kind sums overlap across lanes by construction, and only the
    merged ``busy_s`` is exclusive.
    """
    from DmsfFreeTime import merge_intervals, measure

    ts_ns = ts_ns or time.time_ns()
    if not records:
        return _write_lines(path, [])

    by_cluster = {}
    for r in records:
        by_cluster.setdefault(r.get("cluster", "unknown"), []).append(r)

    lines = []
    for cluster in sorted(by_cluster):
        devs = by_cluster[cluster]
        lines.append(_free_scope(ts_ns, f"cluster={cluster} ALL", devs))
        by_role = {}
        for d in devs:
            by_role.setdefault(d.get("role", "unknown"), []).append(d)
        for role in sorted(by_role):
            lines.append(_free_scope(ts_ns, f"cluster={cluster} role={role}",
                                     by_role[role]))
        free_total = sum(d.get("free_ns", 0) for d in devs)
        for reason, ns in sorted(_sum_maps(devs, "reasons").items(),
                                 key=lambda kv: -kv[1]):
            lines.append(f"{ts_ns} cluster={cluster} FREE reason={reason} "
                         f"free_s={ns / 1e9:.3f} share={_pct(ns, free_total):.2f}%")
        span_total = sum(d.get("span_ns", 0) for d in devs)
        for kind, ns in sorted(_sum_maps(devs, "kinds").items(),
                               key=lambda kv: -kv[1]):
            lines.append(f"{ts_ns} cluster={cluster} KIND kind={kind} "
                         f"busy_s={ns / 1e9:.3f} share={_pct(ns, span_total):.2f}%")

    by_machine = {}
    for r in records:
        by_machine.setdefault(r.get("machine", "unknown"), []).append(r)
    for machine in sorted(by_machine):
        devs = by_machine[machine]
        starts = [d.get("epoch_start_ns") for d in devs if d.get("epoch_start_ns")]
        ends = [d.get("epoch_end_ns") for d in devs if d.get("epoch_end_ns")]
        if not starts or not ends:
            continue
        lo, hi = min(starts), max(ends)
        busy = merge_intervals([iv for d in devs for iv in (d.get("intervals") or [])])
        span_ns = max(hi - lo, 1)
        busy_ns = min(measure(busy), span_ns)
        idles = [d["host_idle"] for d in devs if d.get("host_idle") is not None]
        line = (f"{ts_ns} MACHINE machine={machine} devices={len(devs)} "
                f"free={_pct(span_ns - busy_ns, span_ns):.2f}% "
                f"free_s={(span_ns - busy_ns) / 1e9:.3f} "
                f"span_s={span_ns / 1e9:.3f} "
                f"merge_slop_s={sum(d.get('merge_slop_ns', 0) for d in devs) / 1e9:.3f}")
        if idles:
            line += f" host_idle={min(sum(idles) / len(idles) * 100.0, 100.0):.2f}%"
        lines.append(line)
    if (server_host and server_host_idle is not None
            and server_host not in by_machine):
        # A fleet view that omits the machine holding the controller is not a
        # fleet view — even though it runs no pipeline stage (guide 10 §4). When
        # the controller shares a host WITH devices, that host already has its
        # line and a second one would double-count the machine.
        lines.append(f"{ts_ns} MACHINE machine={server_host} role=server devices=0 "
                     f"host_idle={min(server_host_idle * 100.0, 100.0):.2f}%")

    system = _free_scope(ts_ns, "SYSTEM", records).replace(
        "SYSTEM devices=", f"SYSTEM clusters={len(by_cluster)} "
                           f"machines={len(by_machine)} devices=")
    lines.append(system)
    free_total = sum(r.get("free_ns", 0) for r in records)
    for reason, ns in sorted(_sum_maps(records, "reasons").items(),
                             key=lambda kv: -kv[1]):
        lines.append(f"{ts_ns} SYSTEM FREE reason={reason} "
                     f"free_s={ns / 1e9:.3f} share={_pct(ns, free_total):.2f}%")
    span_total = sum(r.get("span_ns", 0) for r in records)
    for kind, ns in sorted(_sum_maps(records, "kinds").items(), key=lambda kv: -kv[1]):
        lines.append(f"{ts_ns} SYSTEM KIND kind={kind} "
                     f"busy_s={ns / 1e9:.3f} share={_pct(ns, span_total):.2f}%")
    return _write_lines(path, lines)


def write_free_time_series(path, records, ts_ns=None):
    """``free_time_series.log`` — one line per device per bucket (01 §3.10).

    Column 1 is the report's arrival on the **server's** clock, identical on
    every line of one report; the position in the run is ``t_offset_s``, on the
    **device's** clock. The two must never be conflated: devices start at
    different moments and their offsets are not comparable across devices.
    ``bucket_s`` rides on every line rather than being assumed, so a long run
    may widen its buckets without breaking a reader.
    """
    ts_ns = ts_ns or time.time_ns()
    lines = []
    for r in sorted(records, key=lambda x: (x.get("role", ""),
                                            str(x.get("client_id", "")))):
        series = r.get("series") or {}
        values = series.get("free") or []
        bucket_s = series.get("bucket_ns", 0) / 1e9
        head = (f"{r.get('arrival_ns') or ts_ns} "
                f"client={r.get('client_id', 'unknown')} "
                f"role={r.get('role', 'unknown')} "
                f"machine={r.get('machine', 'unknown')} "
                f"cluster={r.get('cluster', 'unknown')}")
        for i, value in enumerate(values):
            lines.append(f"{head} i={i} t_offset_s={i * bucket_s:.3f} "
                         f"bucket_s={bucket_s:.3f} free={min(value * 100.0, 100.0):.2f}%")
    return _write_lines(path, lines)


# ---------------------------------------------------------------------------- #
# Shutdown: infrastructure-host RAM summary (01 §2 files 11-12, method in 11)
# ---------------------------------------------------------------------------- #
def _ram_stats(samples):
    used = sorted(s["used_mb"] for s in samples)
    return {
        "min": used[0], "mean": sum(used) / len(used),
        "p50": nearest_rank(used, 50), "p95": nearest_rank(used, 95),
        "max": used[-1],
    }


def write_broker_ram(path, samples, meta, ts_ns=None):
    """``broker_ram.log`` — four whole-window lines, one per phase, then COMPARE.

    ``DELTA`` measures the window's own ends, so with the window opening at
    server start ``start_mb`` **is** the at-rest figure and ``growth_mb`` is what
    the whole session added. ``COMPARE`` measures phase against phase, which is
    the stronger statement and the answer the file exists to give: *running the
    system costs this host +X MB on average, +Y MB at peak*.

    A phase with no samples is **omitted**, never written as zeros — a run with
    no idle window must look like one. With no samples at all the file still
    gets a ``BROKER`` line carrying ``samples=0`` and the reason, because a
    missing file is indistinguishable from a run where the host was fine.
    """
    ts_ns = ts_ns or time.time_ns()
    host = meta.get("host", "unknown")
    source = meta.get("source", "none")
    if not samples:
        return _write_lines(path, [
            f"{ts_ns} BROKER host={host} source={source} samples=0 "
            f"interval_s={meta.get('interval_s', 0.0):.3f} "
            f"({meta.get('reason') or 'sampler produced no samples'})"])

    first, last = samples[0], samples[-1]
    span_s = (last["ts_ns"] - first["ts_ns"]) / 1e9
    stats = _ram_stats(samples)
    total_mb = max(s["total_mb"] for s in samples)
    peak = max(s["used_mb"] for s in samples)
    note = ("" if source != "rabbitmq_api" else
            " (source=rabbitmq_api: used_mb is the BROKER PROCESS, "
            "total_mb its high-water limit)")
    lines = [
        f"{ts_ns} BROKER host={host} source={source} samples={len(samples)} "
        f"interval_s={meta.get('interval_s', 0.0):.3f} span_s={span_s:.3f} "
        f"total_mb={total_mb:.1f} t_start_ns={first['ts_ns']} "
        f"t_end_ns={last['ts_ns']}{note}",
        f"{ts_ns} USED min_mb={stats['min']:.1f} mean_mb={stats['mean']:.1f} "
        f"p50_mb={stats['p50']:.1f} p95_mb={stats['p95']:.1f} max_mb={stats['max']:.1f} "
        f"min={_pct(stats['min'], total_mb):.2f}% mean={_pct(stats['mean'], total_mb):.2f}% "
        f"p95={_pct(stats['p95'], total_mb):.2f}% max={_pct(stats['max'], total_mb):.2f}%",
        f"{ts_ns} DELTA start_mb={first['used_mb']:.1f} end_mb={last['used_mb']:.1f} "
        f"growth_mb={last['used_mb'] - first['used_mb']:.1f} "
        f"peak_over_start_mb={peak - first['used_mb']:.1f}",
    ]
    rss = [s.get("rss_mb") for s in samples if s.get("rss_mb") is not None]
    if rss:
        line = (f"{ts_ns} RABBIT mean_rss_mb={sum(rss) / len(rss):.1f} "
                f"max_rss_mb={max(rss):.1f}")
        swap = [s["swap_used_mb"] for s in samples if "swap_used_mb" in s]
        if swap:
            line += f" swap_max_mb={max(swap):.1f}"
        lines.append(line)

    by_phase = {}
    for s in samples:
        by_phase.setdefault(s.get("phase", "run"), []).append(s)
    for phase in ("idle", "run", "tail"):
        group = by_phase.get(phase)
        if not group:
            continue                    # omitted, never written as zeros
        st = _ram_stats(group)
        grss = [s.get("rss_mb") for s in group if s.get("rss_mb") is not None]
        line = (f"{ts_ns} PHASE phase={phase} samples={len(group)} "
                f"span_s={(group[-1]['ts_ns'] - group[0]['ts_ns']) / 1e9:.3f} "
                f"min_mb={st['min']:.1f} mean_mb={st['mean']:.1f} "
                f"p50_mb={st['p50']:.1f} p95_mb={st['p95']:.1f} max_mb={st['max']:.1f} "
                f"mean={_pct(st['mean'], total_mb):.2f}% "
                f"max={_pct(st['max'], total_mb):.2f}%")
        if grss:
            line += (f" mean_rss_mb={sum(grss) / len(grss):.1f} "
                     f"max_rss_mb={max(grss):.1f}")
        lines.append(line + f" t_start_ns={group[0]['ts_ns']} "
                            f"t_end_ns={group[-1]['ts_ns']}")

    idle, run, tail = by_phase.get("idle"), by_phase.get("run"), by_phase.get("tail")
    if idle and run:
        idle_mean = sum(s["used_mb"] for s in idle) / len(idle)
        run_mean = sum(s["used_mb"] for s in run) / len(run)
        idle_rss = [s.get("rss_mb") for s in idle if s.get("rss_mb") is not None]
        run_rss = [s.get("rss_mb") for s in run if s.get("rss_mb") is not None]
        line = (f"{ts_ns} COMPARE idle_mean_mb={idle_mean:.1f} "
                f"run_mean_mb={run_mean:.1f} "
                f"run_minus_idle_mb={run_mean - idle_mean:.1f} "
                f"run_peak_over_idle_mb="
                f"{max(s['used_mb'] for s in run) - idle_mean:.1f}")
        if idle_rss and run_rss:
            i_rss = sum(idle_rss) / len(idle_rss)
            r_rss = sum(run_rss) / len(run_rss)
            line += (f" idle_rss_mb={i_rss:.1f} run_rss_mb={r_rss:.1f} "
                     f"run_rss_over_idle_mb={r_rss - i_rss:.1f}")
        if tail:
            tail_mean = sum(s["used_mb"] for s in tail) / len(tail)
            line += (f" tail_mean_mb={tail_mean:.1f} "
                     f"tail_minus_idle_mb={tail_mean - idle_mean:.1f} "
                     f"tail_span_s="
                     f"{(tail[-1]['ts_ns'] - tail[0]['ts_ns']) / 1e9:.3f}")
        lines.append(line)
    return _write_lines(path, lines)


# ---------------------------------------------------------------------------- #
# Shutdown: message size (01 §3.11-3.12, method in guide 12)
# ---------------------------------------------------------------------------- #
def write_message_size(path, reports, unit_size, ts_ns=None):
    """``message_size.log`` — one line per measured worker (01 §3.11).

    Normally exactly one line: the server elects the first edge to register, so
    nine workers do not produce one number nine times. The line carries the
    context that determines the size (mode, split point, compression, unit
    size); a size without them cannot be reproduced. MB is 10**6 bytes, matching
    guide 11 so a payload and a host's memory growth compare without a
    conversion in between.
    """
    ts_ns = ts_ns or time.time_ns()
    lines = []
    for r in sorted(reports, key=lambda x: str(x.get("client_id", ""))):
        n = r.get("n", 0)
        if not n:
            continue
        ctx = r.get("context") or {}
        span_s = r.get("span_s", 0.0)
        total_mb = r.get("total_bytes", 0) / 1e6
        mean_mb = r.get("mean_bytes", 0) / 1e6
        lines.append(
            f"{r.get('arrival_ns') or ts_ns} client={r.get('client_id', 'unknown')} "
            f"role={r.get('role', 'edge')} machine={ctx.get('machine', 'unknown')} "
            f"cluster={r.get('cluster', 'unknown')} "
            f"mode={ctx.get('mode', 'split')} splits={ctx.get('split_point', '?')} "
            f"compress={ctx.get('compress', 'on')} num_bit={ctx.get('num_bit', 1)} "
            f"batch_size={unit_size} n={n} total_mb={total_mb:.3f} "
            f"mean_mb={mean_mb:.3f} p50_mb={r.get('p50_bytes', 0) / 1e6:.3f} "
            f"p95_mb={r.get('p95_bytes', 0) / 1e6:.3f} "
            f"max_mb={r.get('max_bytes', 0) / 1e6:.3f} "
            f"min_mb={r.get('min_bytes', 0) / 1e6:.3f} "
            f"span_s={span_s:.3f} "
            f"rate_mb_s={(total_mb / span_s) if span_s > 0 else 0.0:.3f} "
            f"per_frame_mb={(mean_mb / unit_size) if unit_size else 0.0:.4f}")
    return _write_lines(path, lines)


def write_message_size_series(path, reports, ts_ns=None):
    """``message_size_series.log`` — one line per published message (01 §3.12).

    ``bytes`` is the exact integer and the authoritative value; ``mb`` is the
    same number for readers that plot without converting. ``i`` stays
    non-decreasing even when the series was decimated.
    """
    ts_ns = ts_ns or time.time_ns()
    lines = []
    for r in sorted(reports, key=lambda x: str(x.get("client_id", ""))):
        head = (f"{r.get('arrival_ns') or ts_ns} "
                f"client={r.get('client_id', 'unknown')} "
                f"cluster={r.get('cluster', 'unknown')}")
        for i, offset_s, batch_id, n_bytes in (r.get("series") or []):
            line = f"{head} i={i} t_offset_s={offset_s:.3f}"
            if batch_id is not None:
                line += f" batch_id={batch_id}"
            lines.append(line + f" bytes={int(n_bytes)} mb={int(n_bytes) / 1e6:.3f}")
    return _write_lines(path, lines)


# ---------------------------------------------------------------------------- #
# Shutdown: accuracy — NOT part of the portable contract (see DmsfMapEval)
# ---------------------------------------------------------------------------- #
def write_map(path, ts_ns, per_cluster, window_batches):
    """``map.log`` — ``WINDOW`` then ``ALL`` per cluster, then the ``OVERALL`` pair.

    ``OVERALL`` is the **mean across clusters**, never a re-pooled score:
    clusters may run the same items through different split points, so pooling
    their detections scores a model that never existed. The reading worth having
    is the *spread* between the per-cluster lines.

    Scores are ``{:.4f}`` ratios in [0, 1] and deliberately **not** percent-
    formatted, unlike utilization and free time — a deliberate exception to
    01 §1's percent rule, because mAP is conventionally read as a fraction and a
    percentage invites comparison against published numbers that are not.

    A cluster with no matched frames is **omitted with a warning**, never
    written as zeros: 0.0000 is a real accuracy claim and it would be a false one.
    """
    lines, all_50_95, all_50, win_50_95, win_50 = [], [], [], [], []
    for cluster in sorted(per_cluster):
        result = per_cluster[cluster] or {}
        windows = result.get("windows") or []
        if windows:
            mean_50_95 = sum(w["map50_95"] for w in windows) / len(windows)
            mean_50 = sum(w["map50"] for w in windows) / len(windows)
            win_50_95.append(mean_50_95)
            win_50.append(mean_50)
            lines.append(f"{ts_ns} cluster={cluster} WINDOW "
                         f"mAP50_95={mean_50_95:.4f} mAP50={mean_50:.4f} "
                         f"(mean of {len(windows)} window(s) x {window_batches} "
                         f"batches, step 1)")
        value_50_95, value_50 = result.get("all") or (None, None)
        if value_50_95 is not None:
            all_50_95.append(value_50_95)
            all_50.append(value_50)
            # The matched/total count is free text — informational, ignored by
            # readers — but it is what tells you a cluster scored well on a
            # third of the workload.
            lines.append(f"{ts_ns} cluster={cluster} ALL    "
                         f"mAP50_95={value_50_95:.4f} mAP50={value_50:.4f} "
                         f"({result.get('matched', 0)}/{result.get('gt_frames', 0)} "
                         f"GT frame(s) matched)")

    for tag, values_50_95, values_50 in (("WINDOW      ", win_50_95, win_50),
                                         ("ALL         ", all_50_95, all_50)):
        if values_50_95:
            lines.append(f"{ts_ns} OVERALL {tag}"
                         f"mAP50_95={sum(values_50_95) / len(values_50_95):.4f} "
                         f"mAP50={sum(values_50) / len(values_50):.4f} "
                         f"(avg over {len(values_50_95)} cluster(s))")
    return _write_lines(path, lines)


def write_map_window(path, ts_ns, per_cluster):
    """``map_window.log`` — one line per (cluster, window): accuracy DRIFT.

    A cluster's line count here must equal the window count its ``WINDOW`` line
    in map.log declares, and ``frames`` must equal ``window_batches x
    unit_size`` on a full window; both are conformance checks. A window with a
    small ``frames`` is noise, and this is how a reader can tell without going
    back to the labels.
    """
    lines = []
    for cluster in sorted(per_cluster):
        for w in (per_cluster[cluster] or {}).get("windows") or []:
            lines.append(f"{ts_ns} cluster={cluster} window={w['window']} "
                         f"batches={w['batch_lo']}-{w['batch_hi']} "
                         f"frames={w['frames']} "
                         f"mAP50_95={w['map50_95']:.4f} mAP50={w['map50']:.4f}")
    return _write_lines(path, lines)


# ---------------------------------------------------------------------------- #
# Archiving (05)
# ---------------------------------------------------------------------------- #
def archive_run(log_path, results_root, tag, config_path=None):
    """Copy this run's results plus the config that produced them — 05.

    Copies rather than moves (the live directory keeps its own copies where the
    existing readers expect them; the next run truncates them itself), skips
    empty files so a zero-length log is never archived as a result, and is
    collision-safe so two runs finishing in the same minute do not overwrite.

    Returns the archive directory, or None if nothing was archived. Failure is
    non-fatal by contract — the caller only logs it.
    """
    stamp = datetime.datetime.now().strftime("%m%d_%H%M")
    base = os.path.join(results_root, f"results_{stamp}_{tag}")
    dest, n = base, 1
    while os.path.exists(dest):
        n += 1
        dest = f"{base}-{n}"
    os.makedirs(dest, exist_ok=True)

    copied = []
    for name in RESULT_FILES:
        src = os.path.join(log_path, name)
        if os.path.exists(src) and os.path.getsize(src) > 0:
            shutil.copy2(src, os.path.join(dest, name))
            copied.append(name)

    # The accuracy pair rides along but is counted separately: it is this
    # project's own extension and must never look like one of the fourteen.
    extras = []
    for name in MAP_FILES:
        src = os.path.join(log_path, name)
        if os.path.exists(src) and os.path.getsize(src) > 0:
            shutil.copy2(src, os.path.join(dest, name))
            extras.append(name)

    # Side-cars are not results (00 §6), so they go into their own subdirectory
    # where they cannot be mistaken for one. Best-effort by contract: only
    # devices that share this filesystem contribute, and a missing one warns
    # rather than failing.
    sidecars = 0
    for pattern, sub in (("free_time_*.log", "free_time_devices"),
                         ("message_size_*.log", "message_size_devices"),
                         ("metrics_raw_*.csv", "metrics_raw"),
                         ("metrics_pivoted_dmsf.csv", "metrics_raw")):
        for src in glob.glob(os.path.join(log_path, pattern)):
            if os.path.getsize(src) == 0:
                continue
            try:
                target = os.path.join(dest, sub)
                os.makedirs(target, exist_ok=True)
                shutil.copy2(src, os.path.join(target, os.path.basename(src)))
                sidecars += 1
            except OSError as e:
                print(f"[Archive] Warning: could not copy {src}: {e}")

    config_copied = False
    if config_path and os.path.exists(config_path):
        # Without the config the numbers are unreadable in a month — you will
        # not remember the batch size, the split point, or the client counts.
        shutil.copy2(config_path, os.path.join(dest, "config.yaml"))
        config_copied = True

    if not copied:
        # A run that produced nothing must not leave a directory that looks like
        # a result. Say so loudly and take the empty shell back out.
        shutil.rmtree(dest, ignore_errors=True)
        print("[Archive] WARNING: every result log was missing or empty - "
              "this run produced nothing, nothing archived")
        return None
    # Print what was copied AND what was expected: "9 result file(s)" is only
    # good news if you expected nine (05 §3). Empty files are skipped by
    # design, so check the count against the inventory, not against the folder.
    print(f"[Archive] {len(copied)}/{len(RESULT_FILES)} result file(s), "
          f"{len(extras)} accuracy file(s), {sidecars} side-car(s)"
          f"{', config.yaml' if config_copied else ', NO CONFIG'} -> {dest}")
    missing = [n for n in RESULT_FILES if n not in copied]
    if missing:
        print(f"[Archive] not archived (empty or absent): {', '.join(missing)}")
    return dest


def run_tag(config):
    """Closed tag vocabulary for a DMSF run: ``<auto|fixed>-sp<N>`` (05 §2).

    A tag names which configuration produced the run. Anything finer belongs in
    the archived ``config.yaml``, which is why it is archived — including
    ``map.enable``, even though it is the one setting that makes two runs
    non-comparable on every timing number. Widening the vocabulary for it was
    tried and reverted: 05 §2 is explicit that a tag is a label and not a
    description, and an accuracy run is already unmistakable in a listing
    because it is the only one carrying ``map.log`` and ``map_window.log``.
    """
    dmsf_cfg = config.get("dmsf", config.get("server", {}))
    sp = str(dmsf_cfg.get("split-point", "auto"))
    return "auto" if sp == "auto" else f"fixed-sp{sp}"
