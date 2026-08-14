# DMSF Baseline — Claude Instructions

## Context

This folder implements DMSF (Dynamic Model Splitting Framework) as a **baseline** for the paper *"Cluster-Aware Split Inference and Task Offloading for Edge-Cloud Traffic Object Detection"* (HUST T2025-TN-003). Two variants exist: DMSF-YOLOv5s (original paper) and DMSF-YOLO26n (extension for comparison).

The DMSF paper PDF is at `DMSF_A_Dynamic_Model_Splitting_Framework_for_Edge-Cloud_Collaborative_Inference.pdf` — read it before making architecture changes.

---

## Critical constraints

- **Do NOT modify** `models/dmsf.py` or `utils/loss.py` `utils/visdrone.py` without explicit request — these implement the paper spec exactly
- **cls_gain = 0.5** in `loss.py` — this is intentional (paper-correct), NOT `0.5 * nc/80`
- **Conv 2×2** in `compress_recover.py` is intentional — matches paper Section III-B
- Both DMSF variants share the same `ComputeLoss` and `evaluate()` — do not diverge them
- The distributed run writes a **fixed result format** (see below). Do not change
  a filename, a key name, or a number format in `src/DmsfResults.py` without
  re-reading `guide/01-result-format.md` — the notebook and the validator both
  depend on it
- **`setup.json` is per-host identity**, gitignored like `config.yaml`: `name` and
  `device` for a client launched as bare `python client.py --layer_id 1`.
  Resolution order is flag → `setup.json` → auto-detect, and it must stay that
  way — `run_cluster.ps1` runs the whole fleet from one host file and relies on
  `--name` winning, since `_id_tag` must be unique per process

---

## Result format (distributed run)

`guide/` is the normative spec. This project uses the **cluster** naming
scheme; never mix in the `group_*` names. **Fourteen** result files plus **two**
accuracy files, all truncated at server startup, all written to
`config['log-path']`:

| File | Written | Granularity |
|---|---|---|
| `batch_done_ns.log` | live | one line per completed unit (1 col during warm-up, 2 after) |
| `fps_cluster_ns.log` | live | same arrivals, bucketed by cluster |
| `fps_cluster.log` | shutdown | one line per cluster + `SYSTEM` |
| `utilization.log` | shutdown | one line per device |
| `utilization_cluster.log` | shutdown | `ALL`, cluster×role, `SYSTEM` |
| `latency_cluster.log` | shutdown | pooled service / pipeline / e2e |
| `cut_change_ns.log` | live | one line per control decision |
| `free_time.log` | shutdown | one line per device |
| `free_time_cluster.log` | shutdown | cluster / role / `FREE` / `KIND` / `MACHINE` / `SYSTEM` |
| `free_time_series.log` | shutdown | one device × one time bucket |
| `broker_ram_ns.log` | live | one RAM sample of the queue host |
| `broker_ram.log` | shutdown | `BROKER` / `USED` / `DELTA` / `RABBIT` / `PHASE` / `COMPARE` |
| `message_size.log` | shutdown | one line per measured worker (normally one) |
| `message_size_series.log` | shutdown | one line per published message |
| `map.log` | shutdown | 2 per cluster (`WINDOW`, `ALL`) + 2 `OVERALL` |
| `map_window.log` | shutdown | one sliding window per cluster |

The last two are **NOT part of the portable contract** (`guide/00` §7) — they are
this project's extension and must never stand in for a missing result file. They
are `{:.4f}` ratios, deliberately not percentages, which is the one place this
project departs from `guide/01` §1's percent rule.

`events_ns.log` was renamed to `cut_change_ns.log`; both spellings are
conformant but only one may exist per project (`guide/00` §4).

### Invariants that are easy to break

- **One clock.** Every timestamp in these files is `time.time_ns()` on the
  server. Device timestamps only ever appear as durations computed inside one
  device (`busy_s`, `total_s`, the latency samples).
- **Exactly one stage publishes per unit.** The cloud (tail) publishes to
  `fps_queue`; the edge does not. Two publishers double the measured throughput.
- **The DONE body is an identity**, `CLUSTER_ID`, never a timestamp. The server
  reads it only to bucket the arrival.
- **Σ service samples == `busy_s`.** Both come from the same `get input →
  output` intervals in the device timing log, which is why
  `_compute_utilization` returns the samples rather than recomputing them.
- **`pipeline` ⊇ `service`.** `pipeline` is in-stage residency: on the edge it
  starts when the batch's first frame is read, on the cloud when the message is
  dequeued. The gap is buffering.
- **Measurement rides its own queue, never `rpc_queue`.** `utilization_queue`,
  `freetime_queue`, `msgsize_queue` and `map_queue` all exist for one reason: the
  server stops reading control the moment the last edge reports, and the clouds
  finish later. On a dedicated queue the report waits on the broker instead.
- **A flag that travels is honoured at both ends.** When a feature is off the
  server must skip its collector too, or shutdown burns the full timeout polling
  a queue nobody will publish to and then warns `0/N` — a stall caused by a
  setting working exactly as intended.
- **The frame index for mAP comes from the EDGE's `batch_id`**, carried in the
  `FEATURES` payload. A cloud's own counter counts an arbitrary subset of a
  shared queue and would name every frame wrongly.
- **Console summary format is pinned** by `guide/02-throughput.md` §7 — do not
  reflow those f-strings, the whitespace is the format. They contain em dashes,
  which is why the entrypoints reconfigure stdout to UTF-8.
- **`W = 16`** for the rolling window. Charts assume it.
- **Truncation and scratch purge are both central**, in `DmsfServer.__init__`.
  `truncate_all` empties all 16 logs; `purge_scratch` deletes
  `metrics_pivot.lock`, `metrics_raw_*.csv` and the device side-car logs;
  `DmsfMapEval.purge_scratch` deletes `map/pred/` and `map/collect/`. All of them
  are write-once leftovers that otherwise win over the current run forever
  (guide 05 §5). `map/label/` is ground truth and is never touched.
- **The shutdown hard cap is absolute.** `fps.shutdown_timeout_s` is measured
  from the STOP broadcast and never extended — an earlier version added 60 s
  whenever a cloud was still missing, which never terminates if that cloud is
  dead. A merely slow cloud finishes through `CLOUD_DONE` long before the cap.

### Optional measurements — all four groups ported

Free time (guide 10), infra-host RAM (guide 11) and message size (guide 12) are
implemented, each behind one flag in the **server's** `config.yaml` that travels
in the dispatch message. Each is all-its-files-or-none: never add or remove one
file of a feature.

- **Free time** is not `1 − utilization` here. The edge's frame capture happens
  before `get input`, so utilization cannot see it; `busy_s` is the **union** of
  lane intervals, and `busy_s + free_s == span_s` exactly. The tracker's
  `start()`/`stop()` sit at the same points as the timing log's `start`/`end`, so
  `span_s` and `total_s` describe the same stretch — do not move one without the
  other. A blocked `basic_publish` counts as `KIND kind=send` work, not as
  `FREE reason=backpressure`: this project has no explicit backpressure signal,
  so a blocked publish is indistinguishable from a slow one, and counting it as
  busy biases free time **down**, which is the safe direction.
- **Broker RAM** uses the management API on a loopback broker
  (`source=rabbitmq_api`, `used_mb` = broker process, `total_mb` = its high-water
  limit). The SSH path is the primary one and needs `broker_ram.host/user/
  password` — those are **host** credentials, not the AMQP ones.
- **Message size** is measured by exactly one edge, elected by the server as the
  first to register. A worker must never decide this for itself.

### Accuracy (`map.log`, `map_window.log`)

Outside the portable contract, and off by default. Points that are easy to break:

- The frame index comes from the **edge's** `batch_id`, which travels in the
  `FEATURES` payload. The cloud's own counter is a count of an arbitrary subset
  of a shared queue and would mis-name every frame.
- Writes are **write-once**: every edge replays the same video, so several
  devices reach the same frame and the first writer must win. `map/pred/` and
  `map/collect/` are deleted in `DmsfServer.__init__` — a surviving write-once
  cache wins forever and run N+1 silently reuses run N's predictions.
- `map/label/` is ground truth and is never deleted. `make_map_labels.py`
  generates **pseudo** labels (the same model at split 10, conf 0.25), so the
  scores read as *agreement with the deepest-cut reference*, not as VisDrone
  accuracy. Never quote them as model accuracy.
- A cluster with no matched frames is omitted with a warning, never written as
  `0.0000` — that would be a real accuracy claim, and a false one.
- **`map.enable: true` makes the run's timing numbers non-comparable** with a
  run without it: the prediction pass and the per-frame write are inside
  `get input → output`, so they inflate `busy_s`, `service`, `pipeline` and
  `e2e`. Measured: fps −16%, cloud `busy_s` +21%, `e2e` +22% (`PORT-NOTES.md` §4).
  The run tag stays `fixed-spN` (guide 05 §2 keeps the vocabulary closed); an
  accuracy run is identifiable because it is the only one carrying `map.log`.
- **Scoring is a shutdown step measured in minutes**, not seconds: 237 windows
  over 2015 frames takes 1–4 min. `map.max_det` (COCO's 100) and
  `map.window_batches` are the two knobs; `max_det` is applied on the server, so
  the archived `.txt` files stay complete and re-scorable.

### Verifying a run

```bash
python guide/validate_results.py <run-dir> --names cluster   # must exit 0
python build_nb.py && python run_nb.py                       # must report 0 cell errors
```

`results/synthetic-selftest/` is an invented fixture that exercises the whole
path without a broker; it is labelled and must never be quoted as a result.
`build_nb.py` / `run_nb.py` anchor every path to `Path(__file__).parent`, and
`build_nb.py` stamps that absolute root into the notebook's setup cell — do not
reintroduce a hardcoded root, the project has already been moved once.

---

## Architecture summary

### DMSF-YOLOv5s (`models/dmsf.py`)
- Backbone: YOLOv5s (b0–b9) + neck entry n10
- Split points: {3:128ch, 5:256ch, 7:512ch, 10:256ch}
- GSC: `GlobalStructureCompensation` — 512ch H/32 → 256ch H/16 (gsc_p4) → 128ch H/8 (gsc_p3)
- Detect head: ch=[128, 256, 512], anchor-based

### DMSF-YOLO26n (`models/dmsf_26n.py`)
- Backbone: YOLO26n scale n (width=0.25, depth=0.5) — b0–b10
- Split points: {3:64ch, 5:128ch, 7:256ch, 10:256ch}
- GSC: `GlobalStructureCompensation26n` — 256ch H/32 → 128ch H/16 (gsc_p4) → 128ch H/8 (gsc_p3)
- Detect head: ch=[64, 128, 256], anchor-based (same ANCHORS as YOLOv5s)
- Split 10 special: x_rec IS backbone_final (b10 output), no recover-conv needed

### Compress-Recover (`models/compress_recover.py`)
- `CompressModule`: `F.pad(x,(0,1,0,1))` + Conv2d(k=2) → same H×W; then `sign_ste()` → 1-bit; returns (x_bit, μ, σ)
- `RecoverModule`: ConvTranspose2d(k=2) → crop `[:,:,:-1,:-1]` → LFC (AdaIN, Eq. 6)
- `SwitchableCRModule`: wraps Compress+Recover; one instance per split point
- Channel dicts: `SPLIT_CHANNELS` (YOLOv5s), `SPLIT_CHANNELS_26N` (YOLO26n)

### GSC (`models/compensation.py`)
- `PixelShuffleUpsample(in, out)`: 1×1 conv → out×4ch → PixelShuffle(2) → out ch, 2× spatial
- `DeBottleneck(c)`: transposed conv residual block, preserves channel count
- `GlobalStructureCompensation`: for YOLOv5s backbone (512ch input)
- `GlobalStructureCompensation26n`: for YOLO26n backbone (256ch input)

### Building blocks (`models/common.py`)
- `Conv`, `Bottleneck`, `C3`, `SPPF`, `Detect` — YOLOv5s standard blocks
- `C3k2(c1,c2,n,c3k,e,shortcut)` — YOLO26n block:
  - `c3k=False` → inner block is `_BnC2f` (cv1=3×3, cv2=3×3, e=0.5) — matches ultralytics
  - `c3k=True` → inner block is `C3k` (cv1/cv2/cv3 + inner bottleneck) — matches ultralytics
- `C2PSA(c1,c2,n,e)` — YOLO26n layer 10, cross-stage partial with PSA attention

---

## Training loop (identical for both variants)

```python
sp = random.choice(model.split_points)
loss, _ = compute_loss(model(imgs, split_point=sp), targets)
```

Optimizer: SGD with 3 param groups (bn, weight, bias). LR schedule: `(1 - epoch/EPOCHS)*0.9 + 0.1`.
Validation every 10 epochs at `split_point=10`. Save checkpoint every 10 epochs.

---

## Notebook structure (`DMSF_Colab.ipynb`)

| Cell | Purpose |
|---|---|
| 1 | Mount Google Drive, set `CKPT_DIR` |
| 2 | Check GPU |
| 3 | `pip install opencv-python-headless scipy tqdm pyyaml ultralytics` |
| 4 | Extract `baseline_DMSF.zip` from Drive → `/content/baseline_DMSF` |
| 5 | Prepare VisDrone data (3-branch logic: local SSD / Drive copy / fresh convert) |
| 6 | **Training** — set `MODEL_TYPE = 'yolov5s'` or `'yolo26n_dmsf'` |
| 7 | Evaluate all split points + profile latency |
| 8 | Download checkpoints |

---

## Checkpoint naming

| MODEL_TYPE | last | best |
|---|---|---|
| `yolov5s` | `last.pt` | `best.pt` |
| `yolo26n_dmsf` | `last_dmsf26n.pt` | `best_dmsf26n.pt` |

Stored in `MyDrive/DMSF_checkpoints/`.

---

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `size mismatch` on load_yolo26n_weights | C3k2 Bottleneck structure mismatch | Already fixed in common.py — _BnC2f and C3k match ultralytics |
| Cell 5 goes to `else` branch | `.done` flag not found or Drive shortcut delay | Wait for Drive sync; shortcut needs to be added to My Drive |
| Old code runs despite new zip upload | Cell 4 skips extraction if folder exists | `shutil.rmtree('/content/baseline_DMSF')` then re-run Cell 4 |
| `ModuleNotFoundError: ultralytics` | Cell 3 not re-run after adding ultralytics | Re-run Cell 3 |
| No `metrics_pivoted_dmsf.csv` after a run | A crashed previous run left `metrics_pivot.lock` behind, so `_pivot_and_save` returns early | The server now purges it at startup (`R.purge_scratch`). If it recurs, the server did not start this run — delete `<log-path>/metrics_pivot.lock` by hand. Never affected the seven result logs, which are truncated unconditionally |
| Server never exits after the run | a cloud died before sending `CLOUD_DONE` | it now stops at `fps.shutdown_timeout_s` (default 300 s) and prints the partial count in the stop reason. Raise the key for a genuinely long drain — do not reintroduce an extending cap |

---

## What has been verified

- Full review against DMSF paper PDF — no critical discrepancies
- DMSF26n architecture matches YOLO26n scale n channel dims exactly
- C3k2 key names match ultralytics checkpoint for pretrained weight loading
- Training loop produces differentiable gradients through compress-recover (STE)
