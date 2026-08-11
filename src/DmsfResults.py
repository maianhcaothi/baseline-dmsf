"""Result-format writers for the DMSF baseline.

Implements the seven-file contract in ``guide/01-result-format.md`` using the
**cluster** naming scheme (one scheme per project — never mix in ``group_*``)::

    batch_done_ns.log        system throughput series       live
    fps_cluster_ns.log       per-cluster throughput series  live
    fps_cluster.log          throughput summary             shutdown
    utilization.log          per-device busy ratio          shutdown
    utilization_cluster.log  utilization rolled up          shutdown
    latency_cluster.log      latency distributions          shutdown
    events_ns.log            control-plane events           live

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

# The seven result files, in the order 01 §2 lists them. This tuple is also the
# archive manifest — 05 §4 wants an explicit list rather than "every non-empty
# file", so a stale log can never be stamped into a tagged archive.
RESULT_FILES = (
    "batch_done_ns.log",
    "fps_cluster_ns.log",
    "fps_cluster.log",
    "utilization.log",
    "utilization_cluster.log",
    "latency_cluster.log",
    "events_ns.log",
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


# ---------------------------------------------------------------------------- #
# Lifecycle
# ---------------------------------------------------------------------------- #
def truncate_all(log_path):
    """Empty all seven result files, once, centrally, before any worker writes.

    Unconditional by design: a file truncated only when the feature that writes
    it is enabled leaks the previous run's data into this run's archive
    (05 §4). ``events_ns.log`` is the one that would bite here.
    """
    os.makedirs(log_path, exist_ok=True)
    for name in RESULT_FILES:
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
    for pattern in SCRATCH_GLOBS:
        for path in glob.glob(os.path.join(log_path, pattern)):
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

    if config_path and os.path.exists(config_path):
        # Without the config the numbers are unreadable in a month — you will
        # not remember the batch size, the split point, or the client counts.
        shutil.copy2(config_path, os.path.join(dest, "config.yaml"))
        copied.append("config.yaml")

    if not copied:
        # A run that produced nothing must not leave a directory that looks like
        # a result. Say so loudly and take the empty shell back out.
        os.rmdir(dest)
        print("[Archive] WARNING: every result log was missing or empty - "
              "this run produced nothing, nothing archived")
        return None
    print(f"[Archive] {len(copied)} file(s) -> {dest}")
    return dest


def run_tag(config):
    """Closed tag vocabulary for a DMSF run: ``<auto|fixed>-sp<N>`` (05 §2).

    A tag names which configuration produced the run. Anything finer belongs in
    the archived ``config.yaml``, which is why it is archived.
    """
    dmsf_cfg = config.get("dmsf", config.get("server", {}))
    sp = str(dmsf_cfg.get("split-point", "auto"))
    return "auto" if sp == "auto" else f"fixed-sp{sp}"
