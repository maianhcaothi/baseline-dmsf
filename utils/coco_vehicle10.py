"""
COCO2017 (instances_*.json) -> traffic-only YOLO dataset, native COCO-80 indices.

Keeps the EXACT same 0..79 class indexing as standard COCO/YOLO ("coco.names"
order: person=0, bicycle=1, car=2, motorcycle=3, airplane=4, bus=5, train=6,
truck=7, ...) so predictions line up directly with datasets/groundtruth/*.txt
(used to score split_inference's mAP) -- no remapping needed, nc=80.

Filtering (keeps the dataset small + traffic-focused):
  - An image is KEPT only if it has >=1 annotation among the 5 vehicle
    classes: bicycle, car, motorcycle, bus, truck.
  - "person" is NOT a qualifying class (avoids pulling in selfies / indoor
    photos / sports etc.), but person boxes ARE still labelled if present
    in an already-qualifying (has-a-vehicle) image.
  - Other 74 COCO classes are ignored entirely (never labelled).
  - iscrowd=1 annotations are skipped (no precise per-instance box).

Download (avoids fetching the full ~18GB train2017.zip):
  - Only the qualifying images are fetched, one by one, directly from
    http://images.cocodataset.org/<split>/<file_name>, in parallel.
  - If the image already exists locally under <coco_images_root>/<split>/,
    it is copied instead of re-downloaded.

Output layout (same convention utils/visdrone.py's VisDroneDataset reads):
  <dst_root>/images/{train,val}/  *.jpg
  <dst_root>/labels/{train,val}/  *.txt   (YOLO: cls cx cy w h, normalised)

Usage:
  python utils/coco_vehicle10.py --ann-dir /path/to/coco/annotations --dst data/coco_vehicle10

Then train with:
  python train.py --data data/coco_vehicle10 --nc 80 ...
"""

import argparse
import json
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Standard 80-class COCO order (0-indexed), matches Ultralytics/Darknet coco.names
COCO80_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]
NAME_TO_IDX = {name: i for i, name in enumerate(COCO80_NAMES)}

VEHICLE_CLASSES = {'bicycle', 'car', 'motorcycle', 'bus', 'truck'}  # qualifying classes
LABEL_CLASSES = VEHICLE_CLASSES | {'person'}                       # classes we bother to write boxes for

COCO_IMAGE_BASE_URL = 'http://images.cocodataset.org'


def _fetch_one(url, dst_path):
    if dst_path.exists():
        return True
    try:
        urllib.request.urlretrieve(url, dst_path)
        return True
    except Exception:
        return False


def prepare_coco_vehicle10(ann_dir: str, dst_root: str,
                           coco_images_root: str = None,
                           splits=(('train2017', 'train'), ('val2017', 'val')),
                           max_workers: int = 16):
    """
    ann_dir: directory containing instances_train2017.json / instances_val2017.json
    coco_images_root: optional local dir with <coco_split>/*.jpg already downloaded
                       (if an image exists there, copy instead of re-downloading)
    """
    ann_root = Path(ann_dir)
    dst = Path(dst_root)
    local_root = Path(coco_images_root) if coco_images_root else None

    for coco_split, split in splits:
        ann_path = ann_root / f'instances_{coco_split}.json'
        img_dst = dst / 'images' / split
        lbl_dst = dst / 'labels' / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        with open(ann_path) as f:
            coco = json.load(f)

        cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
        images_by_id = {im['id']: im for im in coco['images']}

        anns_by_image = defaultdict(list)
        has_vehicle = set()
        for ann in coco['annotations']:
            if ann.get('iscrowd', 0):
                continue
            name = cat_id_to_name.get(ann['category_id'])
            if name not in LABEL_CLASSES:
                continue
            anns_by_image[ann['image_id']].append((NAME_TO_IDX[name], ann['bbox']))
            if name in VEHICLE_CLASSES:
                has_vehicle.add(ann['image_id'])

        qualifying_ids = [iid for iid in anns_by_image if iid in has_vehicle]
        print(f"[COCO-vehicle] {split}: {len(qualifying_ids)} images contain a vehicle "
              f"(out of {len(images_by_id)} total in {coco_split})")

        # ---- fetch images: local copy if available, else download ----
        jobs = []
        for image_id in qualifying_ids:
            im = images_by_id[image_id]
            file_name = im['file_name']
            out_path = img_dst / file_name
            if out_path.exists():
                continue
            local_src = (local_root / coco_split / file_name) if local_root else None
            if local_src and local_src.exists():
                shutil.copy(local_src, out_path)
            else:
                url = im.get('coco_url', f'{COCO_IMAGE_BASE_URL}/{coco_split}/{file_name}')
                jobs.append((url, out_path))

        if jobs:
            print(f"  Downloading {len(jobs)} images...")
            ok = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_fetch_one, url, p): p for url, p in jobs}
                for fut in as_completed(futures):
                    if fut.result():
                        ok += 1
            print(f"  Downloaded {ok}/{len(jobs)} images ({len(jobs) - ok} failed)")

        # ---- write labels for images that were actually fetched ----
        kept = 0
        for image_id in qualifying_ids:
            im = images_by_id[image_id]
            file_name = im['file_name']
            if not (img_dst / file_name).exists():
                continue  # download failed, skip label too
            W, H = im['width'], im['height']

            lines = []
            for cls, (x, y, bw, bh) in anns_by_image[image_id]:
                if bw <= 0 or bh <= 0:
                    continue
                cx = (x + bw / 2) / W
                cy = (y + bh / 2) / H
                nw = bw / W
                nh = bh / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            stem = Path(file_name).stem
            with open(lbl_dst / f'{stem}.txt', 'w') as f:
                f.writelines(lines)
            kept += 1

        print(f"[COCO-vehicle] {split}: {kept} images ready in {img_dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ann-dir', required=True, help='Dir with instances_train2017.json / instances_val2017.json')
    p.add_argument('--coco-images-root', default=None, help='Optional local dir with train2017/, val2017/ already downloaded')
    p.add_argument('--dst', default='data/coco_vehicle10', help='Output dataset root (YOLO format)')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    prepare_coco_vehicle10(args.ann_dir, args.dst, args.coco_images_root, max_workers=args.workers)


if __name__ == '__main__':
    main()
