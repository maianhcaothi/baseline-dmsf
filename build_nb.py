"""build_nb.py — emit the DMSF result-visualization notebook.

Regenerates the whole notebook from scratch, so a style fix is one edit here
rather than twenty in the .ipynb. Never patch the notebook directly — the next
build silently reverts it.

    python build_nb.py && python run_nb.py
"""
import nbformat as nbf
from pathlib import Path

OUT = Path(r"d:\SplitInference\DMSF\results\visual\DMSF Result Visualization.ipynb")

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------------------- #
md(r"""
# DMSF Baseline — Result Visualization

Renders one conformant run directory (`guide/01-result-format.md`, **cluster**
naming scheme) produced by `baseline-dmsf`. Charts are written to
`<run-dir>/imgs/`.

The run is a single edge→cloud cluster, so every "per cluster" chart has one
series and the `SYSTEM` line equals it. That is the expected shape here, not a
missing series — the code stays cluster-generic so an N-cluster run needs no
edit.

| Input file | Feeds |
|---|---|
| `batch_done_ns.log` | 02 (+ the event overlay) |
| `fps_cluster_ns.log` | 03, 04 |
| `fps_cluster.log` | 01, and the shared-START derivation |
| `utilization.log` | 08 |
| `utilization_cluster.log` | 07 |
| `latency_cluster.log` | 05, 06, 11 |
| `events_ns.log` | overlaid on 02 |

**Not produced: 09 / 10 (detection accuracy).** Model-accuracy metrics are out of
scope for this result format, and the streaming path has no ground truth — the
cloud runs NMS and discards the predictions. The numbers are left as a gap so the
chart numbering stays stable if a labelled stream is added later.
""")


# ---------------------------------------------------------------------------- #
md("## 0 · Setup — paths, palette, chart style")
code(r'''
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT    = Path(r"d:\SplitInference\DMSF")
RESULTS = ROOT / "results"

# Newest directory that actually holds a run. Set RUN_DIR by hand to pin one.
def _pick_run_dir():
    cands = [p for p in RESULTS.glob("*")
             if p.is_dir() and (p / "batch_done_ns.log").exists()]
    if not cands:
        raise FileNotFoundError(f"no run directory with batch_done_ns.log under {RESULTS}")
    return max(cands, key=lambda p: p.stat().st_mtime)

RUN_DIR = _pick_run_dir()
IMG_DIR = RUN_DIR / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)
print(f"run     {RUN_DIR}")
print(f"images  {IMG_DIR}")

# ---- tokens (guide 06 §3) -----------------------------------------------
SURFACE = "#fcfcfb"; PAGE  = "#f9f9f7"
INK     = "#0b0b0b"; INK_2 = "#52514e"
MUTED   = "#898781"; GRID  = "#e1e0d9"; AXIS = "#c3c2b7"

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, in order
S1_LIGHT   = "#86b6ef"                          # same hue, lighter: raw under smoothed
GOOD, BAD, NEUTRAL = "#0ca30c", "#d03b3b", MUTED

# Colour follows the entity, never its rank or position in a filtered list.
ROLE_COLOR  = {"cloud": S1, "edge": S2, "all": MUTED}
STAT_COLOR  = {"Mean": S1, "p95": S2, "service": S1, "pipeline": S2}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,     # else the PNG is transparent -> black in dark mode
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlepad": 12,
    "axes.labelsize": 10.5, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.8,   # solid hairline
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK_2,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# The surface-coloured edge IS the 2px gap between adjacent fills — it is not a
# contrasting border drawn to separate marks.
BAR_KW  = dict(edgecolor=SURFACE, linewidth=1.2)
LINE_KW = dict(linewidth=2.0, solid_capstyle="round")

SAVED = []

def finish(fig, filename, hide_spines=("top", "right")):
    for ax in fig.get_axes():
        for side in hide_spines:
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
    out = IMG_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    SAVED.append(filename)
    print(f"saved -> {out}")
    plt.show()

def label_bars(ax, bars, fmt="{:.2f}", dy=3, fontsize=9, color=INK_2):
    """Direct value labels above bars — also the relief for sub-3:1 fills."""
    for bar in bars:
        h = bar.get_height()
        if h is None or np.isnan(h):
            continue
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=color)
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 1 · Log parsers

The universal line grammar (`<ts_ns> [FLAG ...] [key=value ...]`) means one
parser core reads every file. No bespoke regex per file.
""")
code(r'''
KV = re.compile(r"(\w+)=([^\s]+)")

def parse_kv_line(line):
    """-> (timestamp, [UPPERCASE flags], {key: value})"""
    parts = line.split()
    if not parts:
        return None, [], {}
    ts    = int(parts[0]) if parts[0].isdigit() else None
    kv    = {k: v for k, v in KV.findall(line)}
    flags = [p for p in parts[1:] if "=" not in p and p.isupper()]
    return ts, flags, kv

def num(v):
    """'55.06%' -> 55.06 ; '336' -> 336.0 ; junk -> nan"""
    if v is None:
        return np.nan
    try:
        return float(str(v).rstrip("%"))
    except ValueError:
        return np.nan

def read_lines(path):
    if not Path(path).exists():
        print(f"!! missing: {path}")
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]

# Raw ids become display labels at parse time, so no chart code ever holds a
# queue name. Any '<name>_<n>' cluster id maps to 'Cluster <n>'.
def cluster_label(raw):
    if raw is None:
        return None
    m = re.search(r"_(\d+)$", raw)
    return f"Cluster {m.group(1)}" if m else str(raw)

def parse_rate_summary(path):                     # fps_cluster.log
    rows = []
    for ln in read_lines(path):
        _, flags, kv = parse_kv_line(ln)
        scope = cluster_label(kv.get("cluster")) or ("System" if "SYSTEM" in flags else None)
        if scope is None:
            continue
        rows.append(dict(scope=scope, fps=num(kv.get("fps")),
                         # SYSTEM carries neither steady_fps nor share. Leave them
                         # NaN — defaulting steady_fps to fps would make the System
                         # row look like it has a steady-state number.
                         steady_fps=num(kv.get("steady_fps")),
                         done=num(kv.get("done")), frames=num(kv.get("frames")),
                         share=num(kv.get("share")),
                         clusters=num(kv.get("clusters"))))
    return rows

def parse_rate_timeline(path):                    # fps_cluster_ns.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if "window_fps" not in kv:
            continue                              # warm-up rows, before the window fills
        rows.append(dict(cluster=cluster_label(kv.get("cluster")), ts=ts,
                         done=int(num(kv.get("done"))),
                         window_fps=num(kv["window_fps"])))
    return rows

def parse_batch_done(path):                       # batch_done_ns.log — TWO arities
    rows, idx = [], 0
    for ln in read_lines(path):
        parts = ln.split()
        idx += 1                                  # counts every line, kept or not
        if len(parts) == 2:                       # warm-up rows carry no rate yet
            rows.append(dict(batch=idx, ts=int(parts[0]), window_fps=float(parts[1])))
    return rows

def parse_latency(path):                          # latency_cluster.log
    rows = []
    for ln in read_lines(path):
        _, flags, kv = parse_kv_line(ln)
        scope = cluster_label(kv.get("cluster")) or ("System" if "SYSTEM" in flags else None)
        if scope is None:
            continue
        rows.append(dict(scope=scope, role=kv.get("role", "all"), kind=kv.get("kind"),
                         n=num(kv.get("n")), mean_ms=num(kv.get("mean_ms")),
                         p50_ms=num(kv.get("p50_ms")), p95_ms=num(kv.get("p95_ms")),
                         max_ms=num(kv.get("max_ms"))))
    return rows

def parse_util_group(path):                       # utilization_cluster.log
    rows = []
    for ln in read_lines(path):
        _, flags, kv = parse_kv_line(ln)
        scope = cluster_label(kv.get("cluster")) or ("System" if "SYSTEM" in flags else None)
        if scope is None:
            continue
        rows.append(dict(scope=scope, role=kv.get("role", "all"),
                         devices=num(kv.get("devices")),
                         utilization=num(kv.get("utilization")),
                         utilization_mean=num(kv.get("utilization_mean")),
                         busy_s=num(kv.get("busy_s")), total_s=num(kv.get("total_s")),
                         packages=num(kv.get("packages"))))
    return rows

def parse_util_device(path):                      # utilization.log
    rows = []
    for ln in read_lines(path):
        _, _, kv = parse_kv_line(ln)
        if "client" not in kv:
            continue
        rows.append(dict(client=kv["client"], role=kv.get("role"),
                         packages=num(kv.get("packages")),
                         busy_s=num(kv.get("busy_s")), total_s=num(kv.get("total_s")),
                         utilization=num(kv.get("utilization"))))
    return rows

def parse_events(path):                           # events_ns.log
    rows = []
    for ln in read_lines(path):
        parts = ln.split(None, 1)                 # split ONCE: descriptions have spaces
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        rows.append(dict(ts=int(parts[0]), description=parts[1]))
    return rows
''')


# ---------------------------------------------------------------------------- #
md("## 2 · Load, and print what you loaded")
code(r'''
# An empty list becomes a DataFrame with NO columns, so a later `df.scope` raises
# KeyError instead of returning an empty frame. Declare the columns.
COLS = {
    "rate":  ["scope", "fps", "steady_fps", "done", "frames", "share", "clusters"],
    "tl":    ["cluster", "ts", "done", "window_fps"],
    "batch": ["batch", "ts", "window_fps"],
    "lat":   ["scope", "role", "kind", "n", "mean_ms", "p50_ms", "p95_ms", "max_ms"],
    "utg":   ["scope", "role", "devices", "utilization", "utilization_mean",
              "busy_s", "total_s", "packages"],
    "utd":   ["client", "role", "packages", "busy_s", "total_s", "utilization"],
    "ev":    ["ts", "description"],
}

df_rate   = pd.DataFrame(parse_rate_summary(RUN_DIR / "fps_cluster.log"),      columns=COLS["rate"])
df_tl     = pd.DataFrame(parse_rate_timeline(RUN_DIR / "fps_cluster_ns.log"),  columns=COLS["tl"])
df_batch  = pd.DataFrame(parse_batch_done(RUN_DIR / "batch_done_ns.log"),      columns=COLS["batch"])
df_lat    = pd.DataFrame(parse_latency(RUN_DIR / "latency_cluster.log"),       columns=COLS["lat"])
df_utg    = pd.DataFrame(parse_util_group(RUN_DIR / "utilization_cluster.log"),columns=COLS["utg"])
df_utd    = pd.DataFrame(parse_util_device(RUN_DIR / "utilization.log"),       columns=COLS["utd"])
df_events = pd.DataFrame(parse_events(RUN_DIR / "events_ns.log"),              columns=COLS["ev"])

for name, df in [("rate summary", df_rate), ("rate timeline", df_tl),
                 ("batch timeline", df_batch), ("latency", df_lat),
                 ("utilization/cluster", df_utg), ("utilization/device", df_utd),
                 ("events", df_events)]:
    print(f"{name:<22} {df.shape}")

CLUSTERS = sorted(c for c in df_rate.scope.unique() if c != "System")
SCOPES   = CLUSTERS + ["System"]
print(f"\nclusters: {CLUSTERS}")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 3 · Verify the assumptions before charting them

Compute what is about to be asserted visually, and branch on the result.
""")
code(r'''
sysrow = df_rate[df_rate.scope == "System"].iloc[0]

# Cluster done/frames ARE additive; cluster fps is NOT (each scope divides by its
# own span). The exact invariant is on the spans, which is the shared-START check.
grp = df_rate[df_rate.scope != "System"]
for key in ("done", "frames"):
    got, want = grp[key].sum(), sysrow[key]
    print(f"cluster {key:<7} sums to {got:.0f}   SYSTEM says {want:.0f}   "
          f"{'OK' if got == want else 'MISMATCH'}")

sys_span = sysrow.frames / sysrow.fps
spans    = (grp.frames / grp.fps).tolist()
print(f"SYSTEM span {sys_span:.2f}s  vs  max cluster span {max(spans):.2f}s   "
      f"{'OK (START is shared)' if abs(max(spans) - sys_span) / sys_span < 0.01 else 'MISMATCH'}")

# Every completion must appear in both live series.
n_batch = sum(1 for _ in open(RUN_DIR / "batch_done_ns.log"))
n_clust = sum(1 for _ in open(RUN_DIR / "fps_cluster_ns.log"))
print(f"live series line counts: batch_done={n_batch}  fps_cluster_ns={n_clust}   "
      f"{'OK' if n_batch == n_clust else 'MISMATCH'}")

# A ratio above 100% is a measurement bug, not a fast device.
over = df_utd[df_utd.utilization > 100]
print(f"devices above 100% utilization: {len(over)}   {'OK' if over.empty else 'BUG'}")

# Sum of service samples must equal that role's busy_s: the two are the same
# intervals, so a gap means one of them is instrumented at the wrong point.
svc = df_lat[df_lat.kind == "service"]
for _, r in svc.iterrows():
    match = df_utg[(df_utg.scope == r.scope) & (df_utg.role == r.role)]
    if match.empty:
        continue
    got, want = r.mean_ms * r.n / 1000.0, match.iloc[0].busy_s
    print(f"Sigma service {r.scope}/{r.role:<5} = {got:9.1f}s   busy_s = {want:9.1f}s   "
          f"{'OK' if abs(got - want) / want < 0.01 else 'MISMATCH'}")

# Is pipeline actually distinguishable from service, or would two identical bars
# be drawn? Two overlapping series read as one, and the reader cannot tell which.
pipe = df_lat[df_lat.kind == "pipeline"]
gap  = (pipe.set_index(["scope", "role"]).mean_ms
        - svc.set_index(["scope", "role"]).mean_ms).abs().max()
BUFFERING_VISIBLE = bool(gap > 1.0)
print(f"\nmax |pipeline - service| mean = {gap:.1f} ms  -> "
      + ("charting both (buffering is measurable)" if BUFFERING_VISIBLE
         else "identical; chart 11 will say so rather than draw two overlapping series"))

# Shared START, recovered exactly: span = frames / fps, so START = last - span.
# This makes "seconds into the run" mean what the axis label claims and puts
# control events on the same origin as the throughput series.
T0_S = df_batch.ts.max() / 1e9 - sys_span
print(f"derived START = {sys_span:.2f}s before the last completion")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 01 · Throughput by cluster

Whole-run rate includes pipeline fill-up — it is what the user experienced.
Steady-state drops warm-up and starts at each cluster's own first completion,
which makes it the fair number for comparing clusters. `SYSTEM` deliberately has
no steady-state bar: the spec does not emit one, and inventing it would mean
picking somebody's first completion to stand for everyone's.
""")
code(r'''
piv = df_rate.set_index("scope").reindex(SCOPES)
series = [("Whole run", "fps", S1), ("Steady state", "steady_fps", S2)]

x, width = np.arange(len(SCOPES)), 0.36
fig, ax = plt.subplots(figsize=(8.6, 4.9))

ymax = np.nanmax(piv[["fps", "steady_fps"]].to_numpy(dtype=float)) * 1.22

for i, (label, col, color) in enumerate(series):
    off  = (i - (len(series) - 1) / 2) * (width + 0.03)   # 0.03 = the surface gap
    vals = piv[col].to_numpy(dtype=float)
    bars = ax.bar(x + off, vals, width, label=label, color=color, **BAR_KW)
    label_bars(ax, bars, fmt="{:.1f}")
    # Name the gap. An unexplained empty slot reads as a dropped series; this one
    # is absent because the spec does not emit it.
    for xi, v in zip(x, vals):
        if np.isnan(v):
            ax.annotate("not emitted\nfor SYSTEM", xy=(xi + off, 0),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=MUTED)

ax.set_xticks(x, SCOPES)
# Pin the x-range to the full grouped-bar geometry. A NaN bar draws nothing, so
# autoscale stops at the last real bar and clips the note that explains the gap.
ax.set_xlim(-0.5, len(SCOPES) - 0.5)
ax.set_ylabel("Throughput (FPS)")
ax.set_ylim(0, ymax)
ax.set_title("Throughput by cluster")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncols=len(series))

# Cluster bars are not additive against the System bar and must not be annotated
# as if they were — each scope divides by its own span (01 §3.3).
ax.annotate(f"system throughput {sysrow.fps:.1f} FPS over {sysrow.done:.0f} units "
            f"({sysrow.frames:.0f} frames)",
            xy=(0.5, -0.15), xycoords="axes fraction",
            ha="center", fontsize=9.5, color=MUTED)
finish(fig, "01_throughput_by_cluster.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 02 · System throughput over the run

Every arrival, on the server clock, with control events overlaid on the same
time origin — which only means anything because both files are stamped by the
same clock. The line starts at ≈ the 16th unit: the first `W-1` completions have
no window yet, and absence is deliberately not written as `0.00`.
""")
code(r'''
s = df_batch.sort_values("batch")
t = s.ts / 1e9 - T0_S
smooth = s.window_fps.rolling(31, center=True, min_periods=1).mean()
m = s.window_fps.mean()

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(t, s.window_fps, color=S1_LIGHT, linewidth=1.0, label="reading")
ax.plot(t, smooth, color=S1, label="31-reading mean", **LINE_KW)
ax.axhline(m, color=S1, linewidth=1.0, alpha=0.45)
# Below the line, not on it: the smoothed series runs along the mean for most of
# the run, so a label above collides with it.
ax.annotate(f"mean {m:.2f}", xy=(0.012, m), xycoords=("axes fraction", "data"),
            xytext=(0, -13), textcoords="offset points", fontsize=9, color=INK_2)

# Selective direct label: the endpoint only, never a number on every point.
ax.plot(t.iloc[-1], smooth.iloc[-1], "o", color=S1, markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.4)
ax.annotate(f"{smooth.iloc[-1]:.1f}", xy=(t.iloc[-1], smooth.iloc[-1]),
            xytext=(9, 0), textcoords="offset points", va="center",
            fontsize=10, fontweight="semibold", color=S1)

# Event rules are chrome, not data: muted hairlines, never a series colour.
ymax = s.window_fps.max() * 1.22
for j, (_, e) in enumerate(df_events.iterrows()):
    et = e.ts / 1e9 - T0_S
    ax.axvline(et, color=MUTED, linewidth=1.0)
    ax.annotate(e.description.split(": ", 1)[-1], xy=(et, ymax),
                xytext=(4, -10 - 13 * (j % 3)),   # stagger: events cluster in time
                textcoords="offset points", fontsize=8, color=MUTED, va="top")

ax.set_xlabel("seconds into the run")
ax.set_ylabel("Rolling window FPS")
ax.set_ylim(0, ymax)
# A rule at t=0 — the split-point decision is made when work is dispatched —
# would otherwise hide underneath the y-axis spine.
ax.set_xlim(-t.max() * 0.012, t.max() * 1.02)
ax.set_title(f"System throughput over the run"
             f"{'' if df_events.empty else f'  —  {len(df_events)} control event(s)'}")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower right", ncols=2)
finish(fig, "02_system_window_fps.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 03 · Throughput per cluster over the run

The x-axis is *per-cluster* completion time, so clusters legitimately end at
different points. A cluster reaches its first full window later than the system
does — it needs 16 completions of its own.
""")
code(r'''
fig, ax = plt.subplots(figsize=(13, 4.6))
cluster_color = {c: col for c, col in zip(CLUSTERS, [S1, S2, S3])}

for cluster in CLUSTERS:
    sub = df_tl[df_tl.cluster == cluster].sort_values("done")
    if sub.empty:
        continue
    tt = sub.ts / 1e9 - T0_S
    sm = sub.window_fps.rolling(31, center=True, min_periods=1).mean()
    col = cluster_color[cluster]
    ax.plot(tt, sm, color=col, label=cluster, **LINE_KW)
    ax.plot(tt.iloc[-1], sm.iloc[-1], "o", color=col, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.4)
    ax.annotate(f"{sm.iloc[-1]:.1f}", xy=(tt.iloc[-1], sm.iloc[-1]),
                xytext=(9, 0), textcoords="offset points", va="center",
                fontsize=10, fontweight="semibold", color=col)

ax.set_xlabel("seconds into the run")
ax.set_ylabel("Rolling window FPS")
ax.set_ylim(0, df_tl.window_fps.max() * 1.22)
ax.set_xlim(0, (df_tl.ts.max() / 1e9 - T0_S) * 1.02)
ax.set_title("Throughput per cluster over the run  (31-reading mean)")
ax.grid(axis="x", visible=False)
# One series needs no legend — the title and the endpoint label name it.
if len(CLUSTERS) > 1:
    ax.legend(loc="lower right", ncols=len(CLUSTERS))
else:
    # This line is the same data as chart 02's. Saying so beats letting the
    # reader wonder whether a second cluster is hidden underneath it.
    ax.annotate(f"one cluster in this run, so this is the system series of "
                f"chart 02 bucketed by cluster — identical by construction",
                xy=(0.5, -0.20), xycoords="axes fraction", ha="center",
                fontsize=9.5, color=MUTED)
finish(fig, "03_cluster_window_fps.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 04 · Window FPS distribution

Stability, not just the average: a high mean with a wide box is a worse result
than a slightly lower mean with a tight one.
""")
code(r'''
fig, ax = plt.subplots(figsize=(8.6, 4.9))
data = [df_tl[df_tl.cluster == c].window_fps.dropna().values for c in CLUSTERS]

bp = ax.boxplot(data, positions=np.arange(len(CLUSTERS)), widths=0.42,
                patch_artist=True, showfliers=False,
                medianprops=dict(color=SURFACE, linewidth=1.8),   # reads on the fill
                whiskerprops=dict(color=AXIS, linewidth=1.0),
                capprops=dict(color=AXIS, linewidth=1.0))
for patch, c in zip(bp["boxes"], CLUSTERS):
    patch.set_facecolor(cluster_color[c])
    patch.set_edgecolor(SURFACE)
    patch.set_linewidth(1.2)

# Draw the mean AT the mean. Hidden low outliers can pull it below Q1, and a
# label parked above the whisker cap would then point at the wrong end of the
# distribution — the defect this chart is most prone to.
lo, hi = [], []
for pos, vals in zip(np.arange(len(CLUSTERS)), data):
    if not len(vals):
        continue
    q1, q3 = np.percentile(vals, [25, 75])
    whisk = vals[(vals >= q1 - 1.5 * (q3 - q1)) & (vals <= q3 + 1.5 * (q3 - q1))]
    mu = vals.mean()
    ax.plot([pos - 0.21, pos + 0.21], [mu, mu], color=INK_2, linewidth=1.4,
            zorder=5)
    ax.annotate(f"mean {mu:.2f}", xy=(pos + 0.23, mu), xytext=(4, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=9.5, color=INK_2)
    lo.append(min(whisk.min(), mu)); hi.append(max(whisk.max(), mu))

pad = (max(hi) - min(lo)) * 0.12 or 1.0
ax.set_ylim(min(lo) - pad, max(hi) + pad)
ax.set_xticks(np.arange(len(CLUSTERS)), CLUSTERS)
ax.set_xlim(-0.6, len(CLUSTERS) - 0.4)
ax.set_ylabel("Rolling window FPS")
ax.set_xlabel("Cluster")
# Say what the box IS — readers do not agree on box conventions. And say that
# the mean can sit outside the box precisely because outliers are suppressed.
ax.set_title("Window FPS distribution  (box = IQR, whiskers = 1.5xIQR, "
             "outliers hidden but still in the mean)")
ax.grid(axis="x", visible=False)
finish(fig, "04_window_fps_distribution.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 05 · Service latency by device role

Two panels, **not** a shared y-axis and **not** a dual axis: the two roles differ
by several times and one scale would flatten the faster one. Each panel is
honestly its own scale, and the panel titles say which is which.

`service` is the device's own `get input → output` — one clock, exact, and the
only latency directly comparable against utilization.
""")
code(r'''
svc   = df_lat[df_lat.kind == "service"]
roles = [r for r in ("cloud", "edge") if r in set(svc.role)]
stats = [("mean_ms", "Mean"), ("p95_ms", "p95")]

fig, axes = plt.subplots(1, len(roles), figsize=(5.9 * len(roles), 4.8))
axes = np.atleast_1d(axes)
x, width = np.arange(len(CLUSTERS)), 0.36

for ax, role in zip(axes, roles):
    sub = svc[svc.role == role].set_index("scope")
    for i, (col, label) in enumerate(stats):
        # Convert units once, here at the pivot — never inside a label formatter.
        vals = [sub.loc[c, col] / 1000.0 if c in sub.index else np.nan for c in CLUSTERS]
        b = ax.bar(x + (i - (len(stats) - 1) / 2) * (width + 0.03), vals, width,
                   label=label, color=STAT_COLOR[label], **BAR_KW)
        label_bars(ax, b, fmt="{:.2f}s")
    ax.set_xticks(x, CLUSTERS)
    ax.set_title(f"{role.capitalize()} devices")
    ax.set_ylim(0, (sub[[c for c, _ in stats]].to_numpy().max() / 1000.0) * 1.22)
    ax.grid(axis="x", visible=False)

axes[0].set_ylabel("Service latency (s)")
# One legend for both panels, above them: in-panel it crowds the tallest bar,
# and repeating it per panel is noise.
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.legend([plt.Rectangle((0, 0), 1, 1, color=STAT_COLOR[l]) for _, l in stats],
           [l for _, l in stats], loc="upper center",
           bbox_to_anchor=(0.5, 0.955), ncols=len(stats))
fig.suptitle("Service latency by device role  (lower is better)", fontsize=14,
             fontweight="semibold", color=INK, y=1.03)
finish(fig, "05_service_latency_by_role.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 06 · End-to-end latency profile

`e2e` runs from the edge starting a unit to the cloud emitting its output — two
machines, so it inherits any offset between their clocks. Report it, but read it
as indicative. It is the number a user experiences; `service` is the number an
engineer optimizes.
""")
code(r'''
e2e    = df_lat[df_lat.kind == "e2e"].set_index("scope")
stats  = [("mean_ms", "Mean"), ("p50_ms", "p50"), ("p95_ms", "p95"), ("max_ms", "Max")]
scopes = [s for s in SCOPES if s in e2e.index]

fig, axes = plt.subplots(1, len(scopes), figsize=(6.0 * len(scopes), 4.8), sharey=True)
axes = np.atleast_1d(axes)
x = np.arange(len(stats))
ymax = e2e[[c for c, _ in stats]].to_numpy().max() / 1000.0

for ax, scope in zip(axes, scopes):
    vals = [e2e.loc[scope, col] / 1000.0 for col, _ in stats]
    # One series per panel: a single colour, and no legend to explain it.
    b = ax.bar(x, vals, 0.58, color=S1, **BAR_KW)
    label_bars(ax, b, fmt="{:.1f}")
    ax.set_xticks(x, [lbl for _, lbl in stats])
    ax.set_title(scope)
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="x", visible=False)

axes[0].set_ylabel("End-to-end latency (s)")
if len(CLUSTERS) == 1:
    # Both panels are the same population. Two identical panels read as an error
    # unless the chart says why they agree.
    axes[-1].annotate("single cluster — the SYSTEM pool is this cluster's pool",
                      xy=(0.5, -0.18), xycoords="axes fraction", ha="center",
                      fontsize=9.5, color=MUTED)
fig.suptitle("End-to-end latency profile  (lower is better; spans two clocks — indicative)",
             fontsize=14, fontweight="semibold", color=INK, y=1.02)
fig.tight_layout()
finish(fig, "06_e2e_latency_profile.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 07 · Device utilization by cluster and role

Pooled `Σbusy / Σtotal`, which weights each device by how long it actually ran.
Where the pooled figure and the plain mean of per-device ratios diverge, the
cluster is imbalanced — one idle device is hiding inside a busy group. Both are
annotated where the spec emits both.
""")
code(r'''
rows = [(c, r) for c in CLUSTERS for r in ("cloud", "edge")] + [("System", "all")]
idx  = df_utg.set_index(["scope", "role"])
rows = [k for k in rows if k in idx.index]
labels = [f"{'System' if s == 'System' else 'C' + s.split()[-1]}\n{r}" for s, r in rows]

x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(1.9 * len(rows) + 3.4, 4.9))

vals   = [float(np.ravel(idx.loc[k, "utilization"])[0]) for k in rows]
colors = [ROLE_COLOR[r] for _, r in rows]
b = ax.bar(x, vals, 0.55, color=colors, **BAR_KW)
label_bars(ax, b, fmt="{:.1f}%", fontsize=9.5)

# Where a plain mean is also emitted, show it as a tick so the divergence — the
# signal that a cluster is imbalanced — is visible rather than inferred.
for xi, k in zip(x, rows):
    mv = float(np.ravel(idx.loc[k, "utilization_mean"])[0])
    if not np.isnan(mv):
        ax.plot([xi - 0.275, xi + 0.275], [mv, mv], color=INK_2, linewidth=1.2)

ax.set_xticks(x, labels)
ax.set_ylabel("Utilization (%)")
ax.set_ylim(0, 118)                    # percentages: fix the ceiling, never autoscale
ax.set_title("Device utilization by cluster and role")
ax.grid(axis="x", visible=False)

handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[r]) for r in ("cloud", "edge", "all")]
ax.legend(handles, ["Cloud", "Edge", "All devices"], loc="upper right", ncols=3)

sysu = float(np.ravel(idx.loc[("System", "all"), "utilization"])[0])
c_u = [v for (s, r), v in zip(rows, vals) if r == "cloud"]
e_u = [v for (s, r), v in zip(rows, vals) if r == "edge"]
note = (f"cloud averages {np.mean(c_u):.0f}% against {np.mean(e_u):.0f}% on the edge"
        if c_u and e_u else f"system utilization {sysu:.0f}%")
ax.annotate(f"{note}  —  bars are pooled utilization; the dark tick is the plain "
            f"per-device mean, emitted only on ALL/SYSTEM lines",
            xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
            fontsize=9.5, color=MUTED)
finish(fig, "07_utilization_by_role.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 08 · Per-device utilization

Sorted within each role, so classes stay blocked *and* rank inside a class is
visible. A bar above 100% would be a measurement bug — overlapping busy
intervals summed — not a fast device, and it is deliberately not clipped.
""")
code(r'''
sub = (df_utd.sort_values(["role", "utilization"], ascending=[True, False])
       .reset_index(drop=True))
pos = np.arange(len(sub))

fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * len(sub) + 3.0), 4.9))
b = ax.bar(pos, sub.utilization, color=[ROLE_COLOR[r] for r in sub.role],
           width=0.66, **BAR_KW)
label_bars(ax, b, fmt="{:.0f}", fontsize=9, color=MUTED)

# cumcount numbers WITHIN each role -> C1 C2 E1 E2. A running enumerate would
# give C1 C2 E3 E4, which reads as missing devices.
ticks = sub.groupby("role").cumcount() + 1
ax.set_xticks(pos, [f"{r[0].upper()}{n}" for r, n in zip(sub.role, ticks)], fontsize=9.5)
ax.set_xlabel("Device  (C = cloud, E = edge)")
ax.set_ylabel("Utilization (%)")
ax.set_ylim(0, 118)
ax.set_title(f"Per-device utilization  —  {sub.utilization.mean():.1f}% mean "
             f"across {len(sub)} device(s)")
ax.grid(axis="x", visible=False)

present = [r for r in ("cloud", "edge") if r in set(sub.role)]
handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[r]) for r in present]
ax.legend(handles, [r.capitalize() for r in present], loc="upper right", ncols=len(present))
finish(fig, "08_device_utilization.png")
''')


# ---------------------------------------------------------------------------- #
md(r"""
## 11 · Buffering — pipeline against service

`pipeline` is in-stage residency: everything from a unit entering the stage to it
leaving. It contains `service`, so the gap between them is pure buffering and
scales with how much a stage holds, not with how fast its devices are.

**If `pipeline ≫ service`, the fix is less buffering, not faster devices** —
throughput does not depend on it, only latency does.

Charts 09 and 10 are intentionally absent: they are the detection-accuracy slots,
which need a labelled stream this pipeline does not have.
""")
code(r'''
if not BUFFERING_VISIBLE:
    print("pipeline and service are identical for every scope/role — no hand-off "
          "buffering to show. Two overlapping series would read as one, so this "
          "chart is skipped rather than shipped empty.")
else:
    kinds = [("service", "Service"), ("pipeline", "Pipeline")]
    fig, axes = plt.subplots(1, len(roles), figsize=(5.9 * len(roles), 4.8))
    axes = np.atleast_1d(axes)
    x, width = np.arange(len(CLUSTERS)), 0.36

    for ax, role in zip(axes, roles):
        top = 0.0
        for i, (kind, label) in enumerate(kinds):
            sub_k = df_lat[(df_lat.kind == kind) & (df_lat.role == role)].set_index("scope")
            vals = [sub_k.loc[c, "mean_ms"] / 1000.0 if c in sub_k.index else np.nan
                    for c in CLUSTERS]
            top = max(top, np.nanmax(vals))
            bb = ax.bar(x + (i - (len(kinds) - 1) / 2) * (width + 0.03), vals, width,
                        label=label, color=STAT_COLOR[kind], **BAR_KW)
            label_bars(ax, bb, fmt="{:.2f}s")
        ax.set_xticks(x, CLUSTERS)
        ax.set_title(f"{role.capitalize()} devices")
        ax.set_ylim(0, top * 1.22)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Mean latency (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=STAT_COLOR[k]) for k, _ in kinds],
               [l for _, l in kinds], loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncols=len(kinds))
    fig.suptitle("Buffering: in-stage residency against service time  (lower is better)",
                 fontsize=14, fontweight="semibold", color=INK, y=1.03)
    finish(fig, "11_pipeline_vs_service.png")
''')


# ---------------------------------------------------------------------------- #
md("## Manifest")
code(r'''
print(f"{len(SAVED)} chart(s) written to {IMG_DIR}\n")
for name in SAVED:
    p = IMG_DIR / name
    print(f"  {name:<34} {p.stat().st_size / 1024:7.1f} KB")

missing = sorted({p.name for p in IMG_DIR.glob('*.png')} - set(SAVED))
if missing:
    print(f"\nstale files left over from an earlier build: {missing}")
''')


# ---------------------------------------------------------------------------- #
nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print("wrote", OUT, f"({len(cells)} cells)")
