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
`--layer_id` is the only required flag: the device's own name and compute device
come from `setup.json` next to `config.yaml` (machine-specific, gitignored),

```json
{ "name": "machine-2", "device": "cpu" }
```

`--name` / `--device` still override it, which is how `run_cluster.ps1` gives each
of its 12 processes a distinct name from one host file.

### Updating the fleet

`deploy.ps1` fast-forwards every host in the fleet over SSH. Copy
`hosts.example.json` to `hosts.json` and fill in the ssh target, logical name
and repo path per host — `hosts.json` names your machines, so it is gitignored
like `config.yaml` and `setup.json`.

```powershell
.\deploy.ps1 -DryRun                     # report only, changes nothing
.\deploy.ps1                             # fetch + merge --ff-only everywhere
.\deploy.ps1 -Only machine-4,device-2    # a subset
.\deploy.ps1 -Ref v1.2                   # pin a tag or SHA
```

It updates by **pull, never by copy**. `config.yaml` and `setup.json` are
per-host and gitignored, and git cannot modify an ignored file — so the code
moves and each host's identity stays put. A file copy would overwrite both, every
host would come up under the same `name`, and since `_id_tag` is derived from it
the hosts' `timing_*.log` and `metrics_raw_*.csv` would overwrite each other.

`--ff-only` is deliberate: a host that has diverged **fails** rather than being
merged into a state nobody has tested. One unreachable host never stops the
fleet — every host is attempted and the closing table is the report, which also
warns when the fleet ends up on more than one commit. Runs from a mixed fleet are
not comparable.

### Replacing the input video across the fleet

`deploy.ps1` moves **code**, by pull, and deliberately never copies files. The
video is the opposite case — `*.mp4` is gitignored, so git cannot carry it and a
copy is the only way. `push_data.ps1` does that, from the same `hosts.json`:

```powershell
# DRY RUN by default: prints the old and new path per host, changes nothing
powershell -ExecutionPolicy Bypass -File push_data.ps1 -Source dai@<host>:/path/video.mp4
powershell -ExecutionPolicy Bypass -File push_data.ps1 -Source dai@<host>:/path/video.mp4 -Apply
```

It stages the file once, then for each host: asks that host where its **own**
`data:` points (config.yaml is per-host, so the old path is a fact only it
holds), copies to a `.incoming` temp name, verifies SHA-256 on both ends, moves
it into place, and only then deletes the old file. A host that fails keeps its
old video and is reported — a partially updated fleet is called out loudly,
because a run across a mixed fleet is not comparable with anything.

> **It also deletes `map/label/`.** Ground truth is generated *from* the video,
> so labels made from the old one describe frames that no longer exist. Leaving
> them would score the new video's detections against the old video's frames —
> plausible numbers, entirely false. Regenerate with `python make_map_labels.py`
> before any run with `map.enable: true`. `-KeepLabels` opts out; do not use it
> unless the video genuinely did not change.

Execution policy on this machine is `AllSigned`, so both `.ps1` scripts need:

```powershell
powershell -ExecutionPolicy Bypass -File deploy.ps1 -DryRun
```

That invocation used to crash `deploy.ps1` on `Join-Path`: `$PSScriptRoot` is
empty inside a `param()` block under `powershell -File`, so the `hosts.json`
default resolved to nothing. Both scripts now resolve the script root in the
body instead.

Each run writes **all fourteen** result logs to `log-path`, in the portable
format specified in `guide/` (**cluster** naming scheme), plus this project's own
two accuracy logs, then archives them alongside the `config.yaml` that produced
them:

```
<log-path>/
├── batch_done_ns.log        system throughput series      one line per completed batch   required
├── fps_cluster_ns.log       per-cluster throughput series one line per completed batch   required
├── fps_cluster.log          throughput summary            per cluster + SYSTEM           required
├── utilization.log          per-device busy ratio         one line per device            required
├── utilization_cluster.log  utilization rolled up         per cluster / role / SYSTEM    required
├── latency_cluster.log      service, pipeline, e2e        pooled, nearest-rank pcts      required
├── cut_change_ns.log        control-plane events          split-point decision           optional
├── free_time.log            per-device idle time          one line per device            optional ┐
├── free_time_cluster.log    idle rolled up, and why       cluster/role/MACHINE/SYSTEM    optional ├ 10
├── free_time_series.log     when each device was idle     one device x time bucket       optional ┘
├── broker_ram_ns.log        queue-host RAM over the run   one sample                     optional ┐ 11
├── broker_ram.log           what the run cost that host   BROKER/USED/DELTA/PHASE        optional ┘
├── message_size.log         bytes one edge puts on wire   one measured worker            optional ┐ 12
├── message_size_series.log  payload size over the run     one published message          optional ┘
├── map.log                  accuracy, both pipelines      2 per cluster + 2 OVERALL      NOT a result file
├── map_window.log           accuracy over the run         one sliding window             NOT a result file
└── results/results_<MMDD>_<HHMM>_<auto|fixed-spN>/   archived copy + config.yaml
```

The last two are **outside the portable contract** (`guide/00-file-inventory.md`
§7): nothing in the fourteen depends on ground truth. They are archived
alongside, and they must never stand in for a missing result file.

What each latency kind measures — they are not interchangeable:

| kind | span | clock | exact? |
|---|---|---|---|
| `service` | the device's own `get input → output` | one | yes |
| `pipeline` | in-stage residency, contains `service` | one | yes |
| `e2e` | edge start → cloud output | **two machines** | indicative |

```bash
python guide/validate_results.py <run-dir> --names cluster   # 14/14, 6/6, exits 0
python build_nb.py && python run_nb.py                       # renders the charts
```

Charts land in `<run-dir>/imgs/`. The notebook covers the six required files and
the control events; the four optional features are written and validated but not
yet charted (`guide/07` catalogues no charts for them either). `run_nb.py` needs
`nbformat` and `nbclient`, which are not in `requirements.txt`.

### The optional measurements, and what each one says here

Every flag lives in `config.yaml` **on the server** and travels to the workers in
the dispatch message; no client reads one from its own config file, and turning
one off also skips the server's collector for it. Each feature is all its files
or none.

| Feature | Flag | What it adds on this project |
|---|---|---|
| Control events (`guide/01` §3.7) | — | the split-point decision, timestamped before it is broadcast |
| Free time (`guide/10`) | `free_time.enable` | **not** `1 − utilization`: the edge decodes and resizes a whole batch *before* `get input`, so that work is invisible to utilization and busy here; the cloud's empty-queue polls become `FREE reason=input`. `busy_s` is a union of lanes, so it stays correct if a worker is ever threaded |
| Queue-host RAM (`guide/11`) | `broker_ram.enable` | the broker is on loopback here, so the SSH premise (a host we run no code on) does not hold and the management API is used instead — every line says `source=rabbitmq_api`, where `used_mb` is the **broker process** and `total_mb` its high-water limit, the level at which publishers get blocked. Point `broker_ram.host/user/password` at a remote broker and the SSH path gives real host memory |
| Message size (`guide/12`) | `message_size.enable` | the size of one intermediate feature map, recorded *before* each publish by exactly one edge that the **server** elects (first to register) |

### Accuracy — `map.log`, and the ground truth it needs

`map.enable` turns on the whole accuracy path. Each cloud writes its
low-threshold detections **write-once** to
`map/pred/<cluster>/frame_NNNNNN.txt` (`class cx cy w h conf`, normalised to
`imgsz`), ships them at shutdown, and the server scores them against
`<map.label_path>/frame_NNNNNN.txt` with two independent pipelines: a 16-batch sliding
window (step 1) into `map_window.log`, and one score over every matched frame
into `map.log`. Frame numbers come from the **edge's** batch id, which travels in
the message — several edges replaying the same video hit the same frame, and
write-once means the first one wins rather than the last.

The video has no labels, so make the reference first:

```bash
python make_map_labels.py            # whole video, split 10, conf 0.25
```

That writes **pseudo** ground truth: the same model at the deepest cut. `map.log`
therefore reads as *agreement with the deepest-cut reference*, not as VisDrone
accuracy — quote it that way, or point `map.label_path` at real labels in the
same layout and every number becomes a real mAP with no code change.

`map.label_path` is where the ground truth lives; blank keeps the default
`<log-path>/map/label`, a relative value resolves against `log-path`, an
absolute one is taken as given. The generator and the server share one resolver
(`DmsfMapEval.resolve_label_dir`), so the setting moves the write and the read
together — `make_map_labels.py --label-path <dir>` overrides it for one
invocation and says so, since the server still scores whatever the config names.
Only the **server** ever reads labels; no device does. Only frames present in
both folders are scored, so a partial label set narrows what is scored instead
of counting the rest as misses. The server prints the folder and its file count
at startup, so a wrong path costs a line rather than a whole run.

> **Never measure accuracy and throughput in the same run.** The prediction pass
> and the per-frame file write sit inside `get input → output`, so they land in
> `busy_s`, `service`, `pipeline` and `e2e`. Runs with `map.enable: true` are not
> comparable with runs without it — measured on this fleet: SYSTEM fps −16%,
> cloud `busy_s` +21%, `e2e` mean +22%, of which **5.96 ms per unit** is the
> accuracy work itself (`KIND kind=map` in `free_time_cluster.log`). An accuracy
> run is the only kind carrying `map.log`, which is how you tell two archives
> apart in a listing. Full numbers: `PORT-NOTES.md` §4.

---

## Key implementation notes

- **Conv 2×2 spatial preservation**: `F.pad(x,(0,1,0,1))` + Conv2d(k=2,p=0) → H×W unchanged
- **STE gradient**: `sign().backward()` passes gradient unchanged (enables training)
- **LFC (Eq. 6)**: `σ_orig × (x_rec − μ_rec)/σ_rec + μ_orig` — aligns distribution after recovery
- **GSC for YOLO26n**: 256ch H/32 → 128ch H/16 (gsc_p4) → 128ch H/8 (gsc_p3)
- **C3k2**: c3k=False → `_BnC2f` (cv1=3×3, e=0.5); c3k=True → `C3k` (cv1/cv2/cv3 structure)
- **cls_gain = 0.5** in `loss.py` (paper-correct; NOT `0.5 × nc/80`)
