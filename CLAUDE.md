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

---

## Result format (distributed run)

`guide/` is the normative spec. This project uses the **cluster** naming
scheme; never mix in the `group_*` names. Seven files, all truncated at server
startup, all written to `config['log-path']`:

| File | Written | Granularity |
|---|---|---|
| `batch_done_ns.log` | live | one line per completed unit (1 col during warm-up, 2 after) |
| `fps_cluster_ns.log` | live | same arrivals, bucketed by cluster |
| `fps_cluster.log` | shutdown | one line per cluster + `SYSTEM` |
| `utilization.log` | shutdown | one line per device |
| `utilization_cluster.log` | shutdown | `ALL`, cluster×role, `SYSTEM` |
| `latency_cluster.log` | shutdown | pooled service / pipeline / e2e |
| `events_ns.log` | live | one line per control decision |

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
- **Measurement rides `utilization_queue`, not `rpc_queue`.** The server stops
  reading control the moment the last edge reports; the clouds finish later.
- **Console summary format is pinned** by `guide/02-throughput.md` §7 — do not
  reflow those f-strings, the whitespace is the format. They contain em dashes,
  which is why the entrypoints reconfigure stdout to UTF-8.
- **`W = 16`** for the rolling window. Charts assume it.
- **Truncation and scratch purge are both central**, in `DmsfServer.__init__`.
  `truncate_all` empties the seven logs; `purge_scratch` deletes
  `metrics_pivot.lock` and `metrics_raw_*.csv`, which are write-once leftovers
  that otherwise win over the current run forever (guide 05 §5).
- **The shutdown hard cap is absolute.** `fps.shutdown_timeout_s` is measured
  from the STOP broadcast and never extended — an earlier version added 60 s
  whenever a cloud was still missing, which never terminates if that cloud is
  dead. A merely slow cloud finishes through `CLOUD_DONE` long before the cap.

### Optional measurements deliberately not ported

Free time (guide 10), infra-host RAM (guide 11) and message size (guide 12) are
**not** implemented, and each is all-its-files-or-none. Do not add one file of a
feature. The reasons are in `README.md`; porting one means working the whole
§4b block of `guide/09-port-checklist.md`, feature flag included.

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
