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

---

## What has been verified

- Full review against DMSF paper PDF — no critical discrepancies
- DMSF26n architecture matches YOLO26n scale n channel dims exactly
- C3k2 key names match ultralytics checkpoint for pretrained weight loading
- Training loop produces differentiable gradients through compress-recover (STE)
