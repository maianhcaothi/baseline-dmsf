# Result-format port — what was done, and what is still true afterwards

This project emits the fourteen result files of `guide/01-result-format.md` in the
**cluster** naming scheme, plus two accuracy files that are **not** part of that
contract. This note is the record `guide/09-port-checklist.md` asks for: the
terminology map, what was ported and what was not, what a worker still reads
locally, and what the accuracy path costs the numbers around it.

---

## 1 · Terminology map (`guide/README` "Terminology")

| Guide term | In this project |
|---|---|
| **unit** | one batch of video frames — one `FEATURES` message, one cloud inference, one completion |
| **unit size** | `server.batch-size` in `config.yaml` (8 in the runs below; the paper uses 32) |
| **worker / device** | one `client.py --layer_id {1,2}` process, identified by its `uuid4`; `--name` / `setup.json` gives the human tag used for its side-car filenames |
| **role** | `edge` (`layer_id=1`) / `cloud` (`layer_id=2`) |
| **cluster** | the single edge→cloud pipeline sharing `dmsf_intermediate_queue`; id `dmsf_intermediate_queue_0` (`CLUSTER_ID`). **One** cluster: every edge publishes into one queue and any cloud may take any unit |
| **server** | `DmsfServer` (`server.py` → `src/DmsfServer.py`) — one process, the authoritative clock, the only writer of every result file |
| **completing stage** | the **cloud** (`DmsfScheduler.last_layer`), which publishes exactly one identity message per unit to `fps_queue`. The edge never publishes there |
| **control event** | the split-point decision taken at dispatch, `fixed` or `auto` — one line in `cut_change_ns.log` per run |

Two consequences of the topology that shaped the implementation:

- **Every edge replays the same video from frame 1.** The same frame index is
  therefore processed by several devices, which is exactly the write-once case
  the accuracy path has to survive.
- **A cloud takes an arbitrary subset of a shared queue**, so its own batch
  counter says nothing about *which* frames it just processed. The edge's
  `batch_id` travels in the message and is what names the prediction files.

---

## 2 · Features ported, and what each one says here

| # | Files | Status | Why |
|---|---|---|---|
| 1–6 | required six | **ported** | the contract |
| 7 | `cut_change_ns.log` | **ported** | was `events_ns.log`; renamed, since a project may use only one of the two conformant spellings (`guide/00` §4) |
| 8–10 | `free_time*.log` | **ported** | it is *not* `1 − utilization` here: the edge decodes and resizes a whole batch **before** `get input`, so utilization cannot see that work and free time counts it as busy. The cloud's empty-queue polls become `FREE reason=input`. `busy_s` is a union of lane intervals, so a threaded worker later stays correct |
| 11–12 | `broker_ram*.log` | **ported, source-labelled** | the broker is on loopback, so `guide/11`'s SSH premise (a host we run no code on) does not hold — host memory there is mostly our own workers. The management API is used instead and every line says `source=rabbitmq_api`, where `used_mb` is the **broker process** and `total_mb` its high-water limit, the level at which publishers get blocked. Set `broker_ram.host/user/password` to a remote broker and the SSH path gives real `/proc/meminfo` host memory |
| 13–14 | `message_size*.log` | **ported** | the size was already computed before each publish; the port added the server-side election (first edge to register), the live local file and the report queue |
| — | `map.log`, `map_window.log` | **ported, outside the contract** | this project's extension. Never counted as one of the fourteen, never a substitute for one |

Nothing was skipped. The validator reports **14/14 result files present, 6/6
required**.

---

## 3 · What a worker still reads from its own config (`guide/README` invariant 9)

Every **measurement** setting now lives in the server's `config.yaml` and travels
in the `START` message inside a `measure` block: `free_time`,
`free_time_bucket_s`, `message_size` (true for exactly one elected edge), `map`,
`map_conf`, and the cluster id. A worker reads none of them locally, so the
flags cannot drift between machines and a run cannot silently mix two
measurement configurations.

What a worker **does** still read from its own files, deliberately:

| Read | From | Why it stays local |
|---|---|---|
| broker address, port, credentials, vhost | `config.yaml` `rabbit:` | needed to reach the server at all, before any dispatch message can exist |
| `name`, `device` | `setup.json` (or `--name` / `--device`) | per-host identity. `_id_tag` must be unique per process, and `run_cluster.ps1` runs a whole fleet from one host file |
| `dmsf.profile-cache`, `dmsf.measure_bandwidth`, `dmsf.split-point`, `nc`, `imgsz`, `weights`, batch size | `config.yaml` `dmsf:` / `server:` | consulted **before registration**, to build the profile and the bandwidth probe the server needs in order to *choose* the split point. Everything the run itself uses comes back down in `START` |

---

## 4 · What the accuracy path costs — measured

Two runs, same fleet (2 edge + 1 cloud, batch 8, split 7, 504 units, 2015
frames), same code, differing only in `map.enable`:

| | `results_0814_2321_fixed-sp7` (map on) | `results_0814_2316_fixed-sp7` (map off) | delta |
|---|---|---|---|
| SYSTEM fps | 100.673 | 120.022 | **−16.1%** |
| `busy_s` SYSTEM | 64.328 | 55.618 | **+15.7%** |
| `busy_s` cloud | 34.082 | 28.262 | **+20.6%** |
| `busy_s` edge | 30.246 | 27.356 | +10.6% |
| `service` mean, cloud | 67.624 ms | 56.076 ms | **+20.6%** |
| `pipeline` mean, cloud | 68.124 ms | 56.488 ms | +20.6% |
| `e2e` mean | 9728 ms | 7948 ms | **+22.4%** |
| `e2e` p95 | 13229 ms | 10519 ms | +25.8% |
| `KIND kind=map` | 3.006 s | *(absent)* | — |

Read those three ways:

1. **The direct cost is measured, not inferred.** `free_time_cluster.log` carries
   a `KIND kind=map` line worth **3.006 s** over 504 units — **5.96 ms per unit**
   of prediction writing, 8.8% of the cloud's busy time. That line simply does
   not exist in the accuracy-off run, which is the cleanest possible statement of
   what the feature costs.
2. **The observed cost is roughly double the direct cost.** The cloud's service
   time rose 11.5 ms per unit against 5.96 ms of measured accuracy work. The rest
   is second-order: 2015 small file writes push on the page cache and the
   scheduler, and everything else on the box slows with it.
3. **The edge got 10.6% slower and does no accuracy work at all.** All three
   processes share one laptop here, so the cloud's extra I/O is felt fleet-wide.
   On a real distributed fleet the edge would be untouched — which is another way
   of saying these two runs differ by more than one variable, and neither should
   be quoted as "the cost of mAP" in general.

**A run with `map.enable: true` is not comparable with one without it.** The two
archives are told apart by the archived `config.yaml` (`guide/05` §2 keeps the
tag vocabulary closed at `<auto|fixed-spN>`) and, more simply, by the fact that
only an accuracy run carries `map.log` and `map_window.log`.

---

## 5 · Checklist boxes that could not be ticked

- **`guide/09` Phase 7 (visualization)** — `build_nb.py` was updated for the
  rename (`events_ns.log` → `cut_change_ns.log`) but the notebook was not
  executed: `nbformat` / `nbclient` are not installed in this environment, so
  `python run_nb.py` cannot run here. No chart code was added for the four new
  features; their files are present and conformant, and `guide/07` catalogues no
  charts for them.
- **`guide/09` Phase 4b, free time, "a disabled tracker reports nothing"** — held.
  But note one deliberate attribution choice: a `basic_publish` that blocks under
  broker flow control is counted as `KIND kind=send` **work**, not as
  `FREE reason=backpressure`. This project has no explicit backpressure signal,
  so a blocked publish is indistinguishable from a slow one; counting it as busy
  biases free time **down**, which is the safe direction, and it is why
  `FREE reason=backpressure` never appears in these runs.
- **`guide/00` §6 side-cars** — per-device free-time logs are written at device
  finish rather than continuously (the message-size one *is* live, one line per
  publish, as `guide/12` §3 requires). They still survive a broker or server
  failure, which is the property `guide/10` §5 asks of them. Timing logs are
  per-worker deleted (each device owns a uniquely tagged file) rather than
  centrally, because they live on the worker's filesystem and the server can
  only reach its own host.
- **`guide/09` Phase 8 point 3, "two runs of the same configuration produce
  throughput within a few percent"** — not run: the two runs here differ by the
  accuracy flag *on purpose*, and a same-configuration repeat was not part of
  this port. The two accuracy-on runs that were made scored bit-identically
  (`mAP50_95=0.3173 / mAP50=0.5475` WINDOW both times), which is the accuracy
  half of that check.

---

## 6 · Two traps found while porting, worth keeping written down

1. **The side-car purge glob matched three result files.** `free_time_*.log`
   also matches `free_time_cluster.log` and `free_time_series.log`, and
   `message_size_*.log` matches `message_size_series.log` — so startup cleanup
   deleted three of the fourteen files immediately after truncation created
   them. A healthy run rewrites them at shutdown and looks fine; a run whose
   collector timed out would have been *missing* a file rather than holding an
   empty one, which is a hard error for the reader (`guide/01` §2). Scratch
   cleanup now refuses to touch any name in the manifest.
2. **The mAP scoring pass was inside the broker-RAM `run` phase.** Scoring takes
   minutes on a long run, so `phase=run` stretched over a period where nothing
   was running and `run_minus_idle_mb` drifted toward zero — the accuracy path
   silently distorting a *result* file. The RAM window now closes before
   scoring: `phase=run` covers 60 s of actual running rather than 238 s of
   mostly-idle.
