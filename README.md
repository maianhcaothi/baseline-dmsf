# DMSF Baseline — VisDrone

Implementation of **DMSF (Dynamic Model Splitting Framework)** applied to YOLOv5s and YOLO26n, used as baseline comparison for the paper *"Cluster-Aware Split Inference and Task Offloading for Edge-Cloud Traffic Object Detection"* (HUST T2025-TN-003).

## Paper reference

> DMSF: A Dynamic Model Splitting Framework for Edge-Cloud Collaborative Inference  
> GLOBECOM 2025 — PDF: `DMSF_A_Dynamic_Model_Splitting_Framework_for_Edge-Cloud_Collaborative_Inference.pdf`

---

## What DMSF does

DMSF modifies a YOLO model so that it can be split at any of 4 candidate layers during inference:

1. **CompressModule** (edge side): Conv 2×2 + BN + SiLU → 1-bit quantization via STE, records μ/σ
2. **RecoverModule** (cloud side): ConvTranspose 2×2 + BN + SiLU → crop → LFC (AdaIN, Eq. 6)
3. **GSC** (cloud side): reconstructs FPN skip connections (P3, P4) from backbone_final using PixelShuffleUpsample + DeBottleneck — avoids sending multiple feature maps
4. **Training**: random split point per step, full pipeline is differentiable via STE

---

## Models

| Class | File | Backbone | Split points | Channel dims |
|---|---|---|---|---|
| `DMSF` | `models/dmsf.py` | YOLOv5s | {3,5,7,10} | {128,256,512,256} |
| `DMSF26n` | `models/dmsf_26n.py` | YOLO26n (width=0.25) | {3,5,7,10} | {64,128,256,256} |

Both models are **compatible with the same** `ComputeLoss` and `evaluate()` — the training loop is identical.

---

## Training

Use `DMSF_Colab.ipynb` (Google Colab T4) or `DMSF_Kaggle.ipynb` (Kaggle T4).

```python
MODEL_TYPE = 'yolov5s'      # DMSF-YOLOv5s  → checkpoints: last.pt / best.pt
MODEL_TYPE = 'yolo26n_dmsf' # DMSF-YOLO26n  → checkpoints: last_dmsf26n.pt / best_dmsf26n.pt
```

- **Resume**: delete the corresponding `last*.pt` from Drive/Kaggle, re-run Cell 6
- **Fresh start**: delete both `last*.pt` and `best*.pt`
- `EPOCHS = 600`, `BATCH = 32`, `IMGSZ = 640`, `NC = 10` (VisDrone classes)
- Validation every 10 epochs; checkpoint synced to Drive every 10 epochs

---

## Project structure

```
baseline_DMSF/
├── models/
│   ├── dmsf.py              # DMSF-YOLOv5s
│   ├── dmsf_26n.py          # DMSF-YOLO26n
│   ├── common.py            # Conv, C3, SPPF, Detect, C3k2, C2PSA, ...
│   ├── compress_recover.py  # CompressModule, RecoverModule, SwitchableCRModule
│   └── compensation.py      # GlobalStructureCompensation (YOLOv5s + YOLO26n)
├── utils/
│   ├── visdrone.py          # VisDroneDataset, letterbox, mosaic augmentation
│   ├── loss.py              # ComputeLoss (CIoU + BCE, anchor-based)
│   ├── metrics.py           # evaluate(), mAP@50 / mAP@50:95
│   └── split_selector.py    # LayerProfiler, SplitSelector (latency-optimal cut)
├── DMSF_Colab.ipynb         # Google Colab training notebook
├── DMSF_Kaggle.ipynb        # Kaggle training notebook
└── DMSF_A_Dynamic_...pdf    # Original DMSF paper
```

---

## Dataset: VisDrone2019-DET

10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor  
Train: 6471 images | Val: 548 images

Stored in Google Drive at `DMSF_checkpoints/data/visdrone/` with `.done` flag.
Link: https://drive.google.com/drive/u/0/folders/1yZRoQ4fGPZ3e5O2owXpRquSCV39DAZ4e
---

## Distributed run — measurement output

`server.py` + `client.py --layer_id {1,2}` run the split pipeline over RabbitMQ.
Each run writes seven plain-text logs to `log-path`, in the portable format
specified in `guide/` (**cluster** naming scheme), then archives them
alongside the `config.yaml` that produced them:

```
<log-path>/
├── batch_done_ns.log        system throughput series      one line per completed batch
├── fps_cluster_ns.log       per-cluster throughput series one line per completed batch
├── fps_cluster.log          throughput summary            per cluster + SYSTEM
├── utilization.log          per-device busy ratio         one line per device
├── utilization_cluster.log  utilization rolled up         per cluster / role / SYSTEM
├── latency_cluster.log      service, pipeline, e2e        pooled, nearest-rank percentiles
├── events_ns.log            control-plane events          split-point decision
└── results/results_<MMDD>_<HHMM>_<auto|fixed-spN>/   archived copy + config.yaml
```

What each latency kind measures — they are not interchangeable:

| kind | span | clock | exact? |
|---|---|---|---|
| `service` | the device's own `get input → output` | one | yes |
| `pipeline` | in-stage residency, contains `service` | one | yes |
| `e2e` | edge start → cloud output | **two machines** | indicative |

```bash
python guide/validate_results.py <run-dir> --names cluster   # conformance, exits 0
python build_nb.py && python run_nb.py                       # renders the charts
```

Charts land in `<run-dir>/imgs/`. Detection-accuracy charts (09, 10) are not
produced: the streaming path has no ground truth, and model accuracy is outside
this result format's scope.

### Optional measurements not ported

`guide/` also specifies three optional measurements. Each is all-its-files or
none (`guide/01-result-format.md` §2), so none of them is half-implemented here:

| Feature | Files | Why not ported |
|---|---|---|
| Free time (`guide/10`) | `free_time*.log` | needs per-lane interval merging on every device; the workers here are single-threaded, so free time collapses to `1 − utilization` and would report nothing utilization does not |
| Infra-host RAM (`guide/11`) | `broker_ram*.log` | the broker runs on loopback in this setup, so there is no separate host to sample over SSH |
| Message size (`guide/12`) | `message_size*.log` | the payload size per publish is already recorded per batch in the per-device metrics CSV; promoting it to the result format needs the server-side "which worker measures" election |

Adding any of them means porting its whole checklist block in
`guide/09-port-checklist.md` §4b, including the feature flag living in the
server's config and travelling in the dispatch message.

---

## Key implementation notes

- **Conv 2×2 spatial preservation**: `F.pad(x,(0,1,0,1))` + Conv2d(k=2,p=0) → H×W unchanged
- **STE gradient**: `sign().backward()` passes gradient unchanged (enables training)
- **LFC (Eq. 6)**: `σ_orig × (x_rec − μ_rec)/σ_rec + μ_orig` — aligns distribution after recovery
- **GSC for YOLO26n**: 256ch H/32 → 128ch H/16 (gsc_p4) → 128ch H/8 (gsc_p3)
- **C3k2**: c3k=False → `_BnC2f` (cv1=3×3, e=0.5); c3k=True → `C3k` (cv1/cv2/cv3 structure)
- **cls_gain = 0.5** in `loss.py` (paper-correct; NOT `0.5 × nc/80`)
