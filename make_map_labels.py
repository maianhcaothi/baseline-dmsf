"""Generate the ground truth `map.log` scores against — PSEUDO ground truth.

The distributed run reads a video, and a video has no labels. This script makes
the only reference that is actually available here: **the same model, run in one
process at the deepest cut, at a confidence threshold high enough that its
output is worth treating as truth.** It writes one file per frame to
``<log-path>/map/label/frame_NNNNNN.txt`` in the layout ``DmsfMapEval`` reads:

    class_id x_center y_center width height        (normalised to imgsz)

**Say what this measures, every time you quote it.** These labels are not
VisDrone annotations and the numbers in ``map.log`` are not this model's
accuracy. They are *agreement between the configuration under test and the
deepest-cut reference*: run the pipeline at ``split-point: 7`` against labels
made at split point 10 and ``map.log`` answers "what did cutting shallower cost
the output", which is a real DMSF question and the one this test bed can
actually answer. Drop real labels into ``map/label/`` in the same layout and
every number becomes a real mAP with no code change.

Frame numbering is the video's own 1-based frame position, which is exactly what
the pipeline derives from the edge's batch id (``batch_id * batch_size + offset
+ 1``), so the two folders line up frame for frame regardless of batch size.

    python make_map_labels.py                      # whole video, split 10, conf 0.25
    python make_map_labels.py --limit 512           # first 512 frames only
    python make_map_labels.py --split-point 10 --conf 0.30
"""
import argparse
import os
import sys

import cv2
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from models.dmsf_26n import DMSF26n
from utils.metrics import non_max_suppression

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--data', default=None, help='video; default = config data')
    ap.add_argument('--split-point', type=int, default=10,
                    help='reference cut. 10 is the deepest and the least degraded')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='only detections this confident become "truth"')
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--limit', type=int, default=0, help='0 = the whole video')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dmsf = config.get('dmsf', {})
    log_path = config.get('log-path', '.')
    imgsz = int(dmsf.get('imgsz', 640))
    batch = args.batch or int(config['server']['batch-size'])
    data = args.data or config['data']
    device = torch.device(args.device
                          or ('cuda' if torch.cuda.is_available() else 'cpu'))

    out_dir = os.path.join(log_path, 'map', 'label')
    os.makedirs(out_dir, exist_ok=True)

    model = DMSF26n(nc=int(dmsf.get('nc', 10))).to(device)
    ckpt = torch.load(dmsf.get('weights', 'best_dmsf26n.pt'),
                      map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict(),
                          strict=False)
    model.eval()

    print("=" * 72)
    print("  PSEUDO ground truth — the model's own output, not human labels.")
    print(f"  reference cut: split_point={args.split_point}  conf={args.conf}  "
          f"iou={args.iou}")
    print(f"  video: {data}")
    print(f"  -> {out_dir}/frame_NNNNNN.txt   (class cx cy w h, normalised)")
    print("  map.log therefore reads as AGREEMENT WITH THIS REFERENCE,")
    print("  never as VisDrone accuracy. Quote it that way.")
    print("=" * 72)

    cap = cv2.VideoCapture(data)
    if not cap.isOpened():
        sys.exit(f"cannot open {data}")

    frame_no, written, boxes_total, buf = 0, 0, 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        # EXACTLY the edge's preprocessing (plain resize, no letterbox), or the
        # boxes land in a different coordinate space than the predictions.
        frame = cv2.resize(frame, (imgsz, imgsz))
        t = torch.from_numpy(frame[:, :, ::-1].copy()).float() / 255.0
        buf.append((frame_no, t.permute(2, 0, 1)))
        if len(buf) == batch or (args.limit and frame_no >= args.limit):
            written, boxes_total = _flush(model, buf, device, args, imgsz,
                                          out_dir, written, boxes_total)
            buf = []
            print(f"  {written} frame(s), {boxes_total} box(es)", end="\r")
        if args.limit and frame_no >= args.limit:
            break
    if buf:
        written, boxes_total = _flush(model, buf, device, args, imgsz, out_dir,
                                      written, boxes_total)
    cap.release()

    print(f"\n  wrote {written} label file(s), {boxes_total} box(es), "
          f"{boxes_total / max(written, 1):.1f} per frame")
    if boxes_total == 0:
        print("  WARNING: no boxes at all — every mAP would score against an "
              "empty truth. Lower --conf, or supply real labels.")


@torch.no_grad()
def _flush(model, buf, device, args, imgsz, out_dir, written, boxes_total):
    imgs = torch.stack([t for _, t in buf]).to(device)
    out = model(imgs, split_point=args.split_point)
    pred = out[0] if isinstance(out, tuple) else out
    for (frame_no, _), det in zip(buf, non_max_suppression(pred, args.conf, args.iou)):
        rows = det.tolist() if hasattr(det, 'tolist') else list(det)
        with open(os.path.join(out_dir, f"frame_{frame_no:06d}.txt"), 'w') as f:
            for x1, y1, x2, y2, _conf, cls in rows:
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    int(cls), (x1 + x2) / 2 / imgsz, (y1 + y2) / 2 / imgsz,
                    (x2 - x1) / imgsz, (y2 - y1) / imgsz))
        written += 1
        boxes_total += len(rows)
    return written, boxes_total


if __name__ == '__main__':
    main()
