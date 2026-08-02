"""
VisDrone2019-DET dataset loader in YOLO format.

Expected directory layout after running prepare_visdrone():
  data/visdrone/
    images/
      train/  *.jpg
      val/    *.jpg
    labels/
      train/  *.txt   (YOLO: cls cx cy w h, normalised)
      val/    *.txt

VisDrone class mapping (0-indexed, 10 classes total):
  0: pedestrian   1: people         2: bicycle       3: car
  4: van          5: truck          6: tricycle       7: awning-tricycle
  8: bus          9: motor

Usage:
  from utils.visdrone import VisDroneDataset
  ds = VisDroneDataset("data/visdrone/images/train", img_size=640, augment=True)
  loader = DataLoader(ds, batch_size=16, collate_fn=ds.collate_fn)
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Letterbox
# --------------------------------------------------------------------------- #
def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bot = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bot, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _adjust_labels_letterbox(labels, orig_hw, ratio, pad, new_size):
    """Adjust normalised xywh labels to match letterboxed image."""
    if not len(labels):
        return labels
    h0, w0 = orig_hw
    dw, dh = pad
    lbs = labels.copy()
    lbs[:, 1] = (lbs[:, 1] * w0 * ratio + dw) / new_size
    lbs[:, 2] = (lbs[:, 2] * h0 * ratio + dh) / new_size
    lbs[:, 3] = lbs[:, 3] * w0 * ratio / new_size
    lbs[:, 4] = lbs[:, 4] * h0 * ratio / new_size
    return lbs


# --------------------------------------------------------------------------- #
# Augmentation helpers
# --------------------------------------------------------------------------- #
def _random_hsv(img, h=0.015, s=0.7, v=0.4):
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                         cv2.LUT(sat, lut_sat),
                         cv2.LUT(val, lut_val)))
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def _random_flip(img, labels):
    if np.random.random() < 0.5:
        img = np.fliplr(img)
        if labels.size:
            labels[:, 1] = 1 - labels[:, 1]
    return img, labels


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class VisDroneDataset(Dataset):
    NC = 10

    def __init__(self, img_dir: str, img_size: int = 640, augment: bool = False):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment

        self.img_files = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ])
        self.label_files = [
            Path(str(p).replace('/images/', '/labels/').replace(p.suffix, '.txt'))
            for p in self.img_files
        ]

    def __len__(self):
        return len(self.img_files)

    def _load_image(self, idx):
        img = cv2.imread(str(self.img_files[idx]))
        if img is None:
            return np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        return img

    def _load_labels(self, idx):
        lbl_path = self.label_files[idx]
        labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(v) for v in parts])
        return np.array(labels, dtype=np.float32).reshape(-1, 5)

    def _mosaic4(self, idx):
        """
        YOLOv5-style 4-image mosaic augmentation.
        Returns (img [H,W,3] BGR, labels [N,5] normalised xywh).
        """
        s = self.img_size
        xc = int(np.random.uniform(s * 0.5, s * 1.5))
        yc = int(np.random.uniform(s * 0.5, s * 1.5))
        indices = [idx] + [np.random.randint(0, len(self)) for _ in range(3)]
        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        all_labels = []  # (cls, x1, y1, x2, y2) in canvas pixels

        for i, j in enumerate(indices):
            img0 = self._load_image(j)
            h0, w0 = img0.shape[:2]
            r = min(s / h0, s / w0)
            wr = int(round(w0 * r))
            hr = int(round(h0 * r))
            img_r = cv2.resize(img0, (wr, hr))

            if i == 0:   # top-left
                x1c, y1c = max(xc - wr, 0), max(yc - hr, 0)
                x2c, y2c = xc, yc
                x1i = wr - (x2c - x1c)
                y1i = hr - (y2c - y1c)
            elif i == 1: # top-right
                x1c, y1c = xc, max(yc - hr, 0)
                x2c, y2c = min(xc + wr, 2 * s), yc
                x1i = 0
                y1i = hr - (y2c - y1c)
            elif i == 2: # bottom-left
                x1c, y1c = max(xc - wr, 0), yc
                x2c, y2c = xc, min(yc + hr, 2 * s)
                x1i = wr - (x2c - x1c)
                y1i = 0
            else:        # bottom-right
                x1c, y1c = xc, yc
                x2c, y2c = min(xc + wr, 2 * s), min(yc + hr, 2 * s)
                x1i, y1i = 0, 0

            x2i = x1i + (x2c - x1c)
            y2i = y1i + (y2c - y1c)
            canvas[y1c:y2c, x1c:x2c] = img_r[y1i:y2i, x1i:x2i]

            pad_x = x1c - x1i
            pad_y = y1c - y1i

            lbls = self._load_labels(j)
            if len(lbls):
                cx_px = lbls[:, 1] * wr
                cy_px = lbls[:, 2] * hr
                bw_px = lbls[:, 3] * wr
                bh_px = lbls[:, 4] * hr
                x1b = cx_px - bw_px / 2 + pad_x
                y1b = cy_px - bh_px / 2 + pad_y
                x2b = cx_px + bw_px / 2 + pad_x
                y2b = cy_px + bh_px / 2 + pad_y
                for k in range(len(lbls)):
                    all_labels.append([lbls[k, 0], x1b[k], y1b[k], x2b[k], y2b[k]])

        # Resize 2s×2s → s×s
        img_out = cv2.resize(canvas, (s, s))

        if not all_labels:
            return img_out, np.zeros((0, 5), dtype=np.float32)

        lbs = np.array(all_labels, dtype=np.float32)
        lbs[:, 1:] = (lbs[:, 1:] * 0.5).clip(0, s - 1)

        bw = lbs[:, 3] - lbs[:, 1]
        bh = lbs[:, 4] - lbs[:, 2]
        lbs = lbs[(bw > 2) & (bh > 2)]
        if not len(lbs):
            return img_out, np.zeros((0, 5), dtype=np.float32)

        bw = lbs[:, 3] - lbs[:, 1]
        bh = lbs[:, 4] - lbs[:, 2]
        cx = (lbs[:, 1] + bw / 2) / s
        cy = (lbs[:, 2] + bh / 2) / s
        nw = bw / s
        nh = bh / s
        labels_out = np.stack([lbs[:, 0], cx, cy, nw, nh], axis=1).astype(np.float32)
        labels_out = labels_out[(labels_out[:, 3] > 0.001) & (labels_out[:, 4] > 0.001)]
        return img_out, labels_out

    def __getitem__(self, idx):
        if self.augment and np.random.random() < 0.5:
            img, labels = self._mosaic4(idx)
        else:
            img0 = self._load_image(idx)
            h0, w0 = img0.shape[:2]
            labels = self._load_labels(idx)
            img, ratio, pad = _letterbox(img0, self.img_size)
            labels = _adjust_labels_letterbox(labels, (h0, w0), ratio, pad, self.img_size)

        if self.augment:
            _random_hsv(img)
            img, labels = _random_flip(img, labels)

        img = img[:, :, ::-1].copy()
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        labels_t = torch.from_numpy(labels) if len(labels) else torch.zeros((0, 5))
        return img, labels_t, str(self.img_files[idx])

    @staticmethod
    def collate_fn(batch):
        imgs, labels_list, paths = zip(*batch)
        imgs = torch.stack(imgs, 0)
        targets = []
        for i, lbl in enumerate(labels_list):
            if lbl.shape[0]:
                idx = torch.full((lbl.shape[0], 1), i, dtype=torch.float32)
                targets.append(torch.cat([idx, lbl], dim=1))
        targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
        return imgs, targets, paths


# --------------------------------------------------------------------------- #
# VisDrone → YOLO label converter
# --------------------------------------------------------------------------- #
def prepare_visdrone(src_root: str, dst_root: str, splits=('train', 'val')):
    """
    Convert VisDrone2019-DET annotation format to YOLO format.

    VisDrone annotation columns:
      bbox_left, bbox_top, bbox_width, bbox_height, score, category_id, truncation, occlusion
    category_id: 1=pedestrian … 10=motor, 0=ignored region
    Maps to YOLO 0-indexed classes 0..9 (skip ignored=0 and other=11).
    """
    import shutil

    VISDRONE_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
                        6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

    src = Path(src_root)
    dst = Path(dst_root)

    split_map = {'train': 'VisDrone2019-DET-train',
                 'val':   'VisDrone2019-DET-val',
                 'test':  'VisDrone2019-DET-test-dev'}

    for split in splits:
        folder = split_map.get(split, split)
        img_src = src / folder / 'images'
        ann_src = src / folder / 'annotations'

        img_dst = dst / 'images' / split
        lbl_dst = dst / 'labels' / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        img_files = sorted(img_src.glob('*.jpg'))
        from tqdm import tqdm
        from PIL import Image as _PIL
        for img_file in tqdm(img_files, desc=f'  {split}', unit='img'):
            shutil.copy(img_file, img_dst / img_file.name)

            ann_file = ann_src / img_file.with_suffix('.txt').name
            try:
                with _PIL.open(img_file) as _im:
                    W, H = _im.size
            except Exception:
                continue

            lines = []
            if ann_file.exists():
                with open(ann_file) as f:
                    for row in f:
                        cols = row.strip().split(',')
                        if len(cols) < 6:
                            continue
                        x, y, bw, bh = (int(cols[i]) for i in range(4))
                        cat = int(cols[5])
                        if cat not in VISDRONE_TO_YOLO:
                            continue
                        cls = VISDRONE_TO_YOLO[cat]
                        cx = (x + bw / 2) / W
                        cy = (y + bh / 2) / H
                        nw = bw / W
                        nh = bh / H
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            with open(lbl_dst / img_file.with_suffix('.txt').name, 'w') as f:
                f.writelines(lines)

        print(f"[VisDrone] Converted {split}: {len(list(img_dst.glob('*.jpg')))} images")
