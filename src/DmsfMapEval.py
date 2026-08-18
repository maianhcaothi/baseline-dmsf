"""Detection accuracy — ``map.log`` and ``map_window.log``.

**This is not part of the portable result contract.** ``guide/README`` puts
model-accuracy metrics deliberately out of scope: nothing in the fourteen result
files depends on ground truth, or on the pipeline doing inference at all. These
two files are this project's own extension, so their format is pinned here
rather than in ``guide/``. They must never stand in for a missing result file —
an accuracy log next to twelve of fourteen results is still an incomplete run.

Every other measurement describes what the run *cost*. This one describes what
it *produced*, and it is the only file that can show a configuration which
improved every other number and made the output worse.

Three rules shape the whole module:

* **Workers produce predictions; the SERVER scores them.** mAP is a ranking
  metric over a pooled detection set. The mean of two devices' mAP is not the
  mAP of their combined detections — unlike a mean it has no weighted
  combination that reconstructs it at all. So devices ship raw per-frame
  prediction files and the server pools per cluster and reduces once.
* **Pool per cluster, never across clusters.** One cluster's devices share a
  split point, so their detections belong to one model configuration. ``OVERALL``
  is therefore a mean *across* cluster scores, never a re-pooled score.
* **This measurement changes the run it is in.** Writing the predictions costs
  one file per frame (plus a second post-process pass whenever ``map.conf``
  differs from the run's own ``conf``), all of it inside ``get input -> output``
  and therefore inside ``busy_s``, ``service``, ``pipeline`` and ``e2e``. Off by
  default. Never measure accuracy and throughput in the same run.

The metric is self-contained — COCO-style AP over IoU 0.50:0.05:0.95 with
101-point interpolation, no torch and no torchmetrics — so an archived run can
be re-scored later from the ``.txt`` files alone, with a different threshold or
a fixed scoring bug, without this codebase and without re-running anything. A
shipped score can never be recomputed.
"""

import io
import os
import shutil
import zipfile

#: The COCO definition of both. Changing either makes the numbers incomparable
#: with every published mAP, so they are constants and not configuration.
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
RECALL_POINTS = tuple(i / 100.0 for i in range(101))

#: Predictions live beside the ground truth, in the same layout and the same
#: format plus a confidence column, so the two folders diff frame for frame and
#: one parser reads both.
PRED_ROOT = os.path.join("map", "pred")
LABEL_ROOT = os.path.join("map", "label")
COLLECT_ROOT = os.path.join("map", "collect")

#: Ground truth is the one input to this measurement that is NOT produced by the
#: run, so it is the one location worth being able to move: real VisDrone labels,
#: a second pseudo-label set made at another reference cut, or one shared folder
#: that several checkouts score against. ``map.label_path`` in the config
#: overrides it; blank keeps ``<log-path>/map/label``.
LABEL_PATH_KEYS = ("label_path", "label-path")

#: Matches W in guide/02 §6, so a step in the accuracy series and a dip in the
#: throughput series cover the same span of the run.
DEFAULT_WINDOW_BATCHES = 16


def safe_name(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


def resolve_label_dir(map_cfg=None, log_path="."):
    """Where ground truth lives: ``map.label_path``, else ``<log-path>/map/label``.

    ONE resolver, called by both the server that scores and the generator that
    writes, because a label path that is read differently by the two is worse
    than no configuration at all: the generator fills one folder, the server
    scores an empty one, and the run reports "no ground truth" for a set that
    exists. Prefer importing this over rebuilding the path from ``LABEL_ROOT``.

    A relative value is resolved against ``log-path``, which is where the
    default already sits, so ``map/label`` and the default name the same folder.
    An absolute value is taken as given — that is the point of the setting.
    Blank, missing, or a config without a ``map`` section all keep the default,
    so an older config.yaml needs no edit.
    """
    value = ""
    for key in LABEL_PATH_KEYS:
        if map_cfg and map_cfg.get(key):
            value = str(map_cfg.get(key)).strip()
            break
    if not value:
        return os.path.join(log_path, LABEL_ROOT)
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(log_path, value))


def purge_scratch(log_path, label_dir=None):
    """Delete ``map/pred/`` and ``map/collect/`` — centrally, at startup.

    A write-once cache that survives a run **wins forever**: run N+1 silently
    reuses run N's predictions for every frame they share, and the numbers look
    entirely plausible, which is what makes this worth deleting rather than
    detecting. Done in the component that starts once — doing it per worker
    would let a late starter wipe files an earlier worker is already writing.
    Ground truth is never touched. ``label_dir`` says where it is now that
    ``map.label_path`` can point it anywhere, INCLUDING inside one of these two
    trees — a config that costs someone a label set they spent an hour
    generating is a bad trade for a startup convenience, so an overlapping root
    is skipped with a warning rather than deleted.
    """
    protected = os.path.abspath(label_dir or os.path.join(log_path, LABEL_ROOT))
    removed = 0
    for root in (PRED_ROOT, COLLECT_ROOT):
        path = os.path.join(log_path, root)
        if not os.path.isdir(path):
            continue
        target = os.path.abspath(path)
        if protected == target or protected.startswith(target + os.sep):
            print(f"[mAP] Not purging {path}: the ground truth ({protected}) "
                  f"lives inside it. Move map.label_path out of map/pred and "
                  f"map/collect, or last run's predictions stay and win.")
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as e:
            print(f"[mAP] Warning: could not remove {path}: {e}")
    return removed


# ---------------------------------------------------------------------------- #
# Device side — produce predictions
# ---------------------------------------------------------------------------- #
class PredictionWriter:
    """Writes ``map/pred/<cluster>/frame_NNNNNN.txt``, one file per frame.

    **Write-once**: a frame index that already has a file is SKIPPED. In this
    project every edge replays the same video, so several devices in one cluster
    process the same frame — first writer wins, and no device can overwrite
    another's work halfway through a run.

    Frame numbers come from the **edge's** batch id, which travels in the
    message, never from the cloud's own counter: a cloud takes an arbitrary
    subset of units off a shared queue, so its local counter would name every
    frame wrongly. A device that skips a unit therefore leaves a HOLE rather
    than shifting every later frame onto the wrong label — a shift scores as a
    total miss everywhere and looks like a broken model instead of a lost batch.

    A disabled writer is a no-op costing one attribute lookup, so the call sits
    unconditionally in the post-process path.
    """

    def __init__(self, enabled=False, cluster="unknown", imgsz=640, log_path="."):
        self.enabled = bool(enabled)
        self.imgsz = float(imgsz)
        self.dir = os.path.join(log_path, PRED_ROOT, safe_name(cluster))
        self.frames = 0
        self.skipped = 0
        self.written = []
        self._warned = False
        if self.enabled:
            try:
                os.makedirs(self.dir, exist_ok=True)
            except Exception as e:
                self._disable(e)

    def write_batch(self, detections, batch_id, batch_size):
        """One ``frame_NNNNNN.txt`` per image of this unit, 1-based frame ids.

        ``detections`` is the NMS output: an iterable of (M, 6) rows
        ``[x1, y1, x2, y2, conf, cls]`` in network-input pixels.
        """
        if not self.enabled or detections is None or batch_id is None:
            return
        for offset, det in enumerate(detections):
            frame = batch_id * batch_size + offset + 1
            path = os.path.join(self.dir, f"frame_{frame:06d}.txt")
            if os.path.exists(path):
                self.skipped += 1        # another device already has this frame
                continue
            try:
                rows = det.tolist() if hasattr(det, "tolist") else list(det)
                with open(path, "w") as f:
                    for x1, y1, x2, y2, conf, cls in rows:
                        f.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                            int(cls),
                            (x1 + x2) / 2 / self.imgsz, (y1 + y2) / 2 / self.imgsz,
                            (x2 - x1) / self.imgsz, (y2 - y1) / self.imgsz,
                            float(conf)))
                self.frames += 1
                self.written.append(os.path.basename(path))
            except Exception as e:
                self._disable(e)
                return

    def _disable(self, err):
        """One warning, then stop — never one warning per frame."""
        if not self._warned:
            print(f"[mAP] Warning: prediction write failed ({err}); "
                  f"predictions disabled for the rest of the run")
            self._warned = True
        self.enabled = False

    def pack(self):
        """Zip the frames THIS device wrote, for the trip to the server.

        Thousands of tiny text files compress to a few hundred kB, so the whole
        set travels as one message rather than one per frame — and it happens at
        shutdown, entirely outside the measured window.
        """
        if not self.written or not os.path.isdir(self.dir):
            return None
        try:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in sorted(set(self.written)):
                    path = os.path.join(self.dir, name)
                    if os.path.exists(path):
                        archive.write(path, name)
            return buffer.getvalue()
        except Exception as e:
            print(f"[mAP] Warning: packing predictions failed: {e}")
            return None


# ---------------------------------------------------------------------------- #
# Server side — collect, parse, score
# ---------------------------------------------------------------------------- #
def unpack_predictions(payload, destination):
    """Land one device's zip under its CLUSTER's directory.

    Several devices in a cluster each hold part of the frame range, so files
    merge into one directory per cluster: the cluster is the scope the metric is
    reported at, and merging here is what makes it one pooled detection set
    rather than several.
    """
    try:
        os.makedirs(destination, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(destination)
    except Exception as e:
        print(f"[mAP] Warning: unpacking predictions failed: {e}")
        return 0
    return len([n for n in os.listdir(destination) if n.endswith(".txt")])


def load_boxes(directory, image_size=640.0, with_conf=False, max_det=0):
    """``{frame: [(cls, conf, x1, y1, x2, y2)]}`` from ``frame_NNNNNN.txt`` files.

    A malformed line costs its own box and a malformed file its own frame;
    neither is fatal. Ground truth has no confidence column, so ``with_conf``
    stays false there and every label reads as certainty 1.0.

    ``max_det`` keeps only the N most confident detections per frame — COCO's
    own convention (``maxDets=100``), and the difference between a score that
    is comparable with published numbers and one that is not. It is applied
    HERE, on the server, and never when the predictions are written: the
    ``.txt`` files stay complete, so an archived run can be re-scored at a
    different cap without re-running anything.
    """
    out = {}
    if not directory or not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".txt"):
            continue
        try:
            frame = int(os.path.splitext(name)[0].split("_")[-1])
        except ValueError:
            continue
        boxes = []
        try:
            with open(os.path.join(directory, name)) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                        conf = float(parts[5]) if with_conf and len(parts) > 5 else 1.0
                    except ValueError:
                        continue
                    boxes.append((cls, conf,
                                  (cx - bw / 2) * image_size, (cy - bh / 2) * image_size,
                                  (cx + bw / 2) * image_size, (cy + bh / 2) * image_size))
        except Exception:
            continue
        if max_det and len(boxes) > max_det:
            boxes.sort(key=lambda b: -b[1])
            boxes = boxes[:max_det]
        out[frame] = boxes
    return out


def mean_average_precision(preds, gts, iou_thresholds=IOU_THRESHOLDS):
    """COCO-style mAP over one pooled detection set.

    Returns ``(mAP@50:95, mAP@50)``, or ``(None, None)`` when nothing can be
    scored — which is written as an omitted line, never as ``0.0000``. A zero is
    a real accuracy claim and it would be a false one.

    Only frames present in BOTH dicts are scored: a prediction for a frame with
    no label cannot be judged, and a label whose frame never reached this
    cluster would score as a total miss and punish the *partitioning* rather
    than the model. Classes are averaged with equal weight, so one common class
    cannot mask total failure on a rare one.
    """
    frames = sorted(set(preds) & set(gts))
    if not frames:
        return None, None

    gt_by_class, pred_by_class = {}, {}
    for frame in frames:
        for cls, _conf, *box in gts[frame]:
            gt_by_class.setdefault(cls, {}).setdefault(frame, []).append(box)
        for cls, conf, *box in preds.get(frame, []):
            pred_by_class.setdefault(cls, []).append((conf, frame, box))
    if not gt_by_class:
        return None, None

    per_threshold = {thr: [] for thr in iou_thresholds}
    for cls, gt_frames in gt_by_class.items():
        n_gt = sum(len(v) for v in gt_frames.values())
        detections = sorted(pred_by_class.get(cls, []), key=lambda d: -d[0])
        for thr in iou_thresholds:
            per_threshold[thr].append(
                _average_precision(detections, gt_frames, n_gt, thr))

    by_threshold = {thr: sum(v) / len(v) for thr, v in per_threshold.items() if v}
    if not by_threshold:
        return None, None
    overall = sum(by_threshold.values()) / len(by_threshold)
    return overall, by_threshold.get(0.5, overall)


def _average_precision(detections, gt_frames, n_gt, iou_threshold):
    """101-point interpolated AP for one class at one IoU threshold.

    Greedy highest-confidence-first matching with each ground-truth box
    claimable once, so a second box on the same object is a false positive —
    which is what makes duplicate detections cost score rather than being free.
    """
    if n_gt == 0:
        return 0.0
    claimed = {frame: [False] * len(boxes) for frame, boxes in gt_frames.items()}
    tp = [0] * len(detections)
    fp = [0] * len(detections)

    for i, (_conf, frame, box) in enumerate(detections):
        candidates = gt_frames.get(frame)
        if not candidates:
            fp[i] = 1
            continue
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(candidates):
            if claimed[frame][j]:
                continue
            value = _iou(box, gt_box)
            if value > best_iou:
                best_iou, best_j = value, j
        if best_j >= 0 and best_iou >= iou_threshold:
            claimed[frame][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    cum_tp = cum_fp = 0
    precisions, recalls = [], []
    for i in range(len(detections)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / n_gt)
    if not precisions:
        return 0.0

    for i in range(len(precisions) - 2, -1, -1):        # monotone envelope
        precisions[i] = max(precisions[i], precisions[i + 1])
    total, j = 0.0, 0
    for point in RECALL_POINTS:
        while j < len(recalls) and recalls[j] < point:
            j += 1
        total += precisions[j] if j < len(precisions) else 0.0
    return total / len(RECALL_POINTS)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------- #
# The two pipelines
# ---------------------------------------------------------------------------- #
def evaluate_cluster(preds, gts, unit_size, window_batches=DEFAULT_WINDOW_BATCHES):
    """Both pipelines for one cluster.

    ``WINDOW`` slides ``window_batches`` units ONE unit at a time, so window k
    and k+1 cover the same amount of work shifted forward by one and are
    directly comparable — it is the accuracy counterpart of the rolling window
    rate in ``batch_done_ns.log``. Disjoint blocks would not be comparable: a
    step between two of them could be the configuration changing or just a
    different stretch of the workload.

    ``ALL`` is one mAP over every matched frame of the cluster.
    """
    all_50_95, all_50 = mean_average_precision(preds, gts)
    matched = len(set(preds) & set(gts))

    windows = []
    if preds and unit_size > 0:
        n_batches = (max(preds) - 1) // unit_size + 1
        for start in range(0, max(1, n_batches - window_batches + 1)):
            stop = min(start + window_batches, n_batches) - 1
            lo, hi = start * unit_size + 1, (stop + 1) * unit_size
            window_preds = {k: v for k, v in preds.items() if lo <= k <= hi}
            window_gts = {k: v for k, v in gts.items() if lo <= k <= hi}
            value_50_95, value_50 = mean_average_precision(window_preds, window_gts)
            if value_50_95 is None:
                continue          # this stretch of the run has no ground truth
            windows.append({
                "window": len(windows), "batch_lo": start, "batch_hi": stop,
                "frames": len(set(window_preds) & set(window_gts)),
                "map50_95": value_50_95, "map50": value_50,
            })

    return {"all": (all_50_95, all_50), "windows": windows,
            "matched": matched, "gt_frames": len(gts)}
