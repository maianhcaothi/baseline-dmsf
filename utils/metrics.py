"""
mAP evaluation utilities compatible with YOLOv5 Detect output.

ap_per_class(): standard VOC/COCO mAP computation.
evaluate():     runs model over a DataLoader, returns mAP@0.5 and mAP@0.5:0.95.
"""

import torch
import numpy as np
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# IoU (vectorised, xyxy format)
# --------------------------------------------------------------------------- #
def box_iou(box1, box2):
    """box1: (N,4), box2: (M,4), xyxy. Returns (N,M) IoU matrix."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter = (torch.min(box1[:, None, 2], box2[:, 2]) - torch.max(box1[:, None, 0], box2[:, 0])).clamp(0) * \
            (torch.min(box1[:, None, 3], box2[:, 3]) - torch.max(box1[:, None, 1], box2[:, 1])).clamp(0)
    return inter / (area1[:, None] + area2 - inter + 1e-9)


# --------------------------------------------------------------------------- #
# NMS
# --------------------------------------------------------------------------- #
def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    pred: (B, N, 5+nc) — output of Detect head in eval mode (cat of z).
    Returns list of (M, 6) tensors: [x1,y1,x2,y2,conf,cls] per image.
    """
    bs = pred.shape[0]
    nc = pred.shape[2] - 5
    output = [torch.zeros((0, 6), device=pred.device)] * bs

    for bi in range(bs):
        x = pred[bi]
        x = x[x[:, 4] > conf_thres]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        # xywh → xyxy
        box = x[:, :4].clone()
        box[:, 0] = x[:, 0] - x[:, 2] / 2
        box[:, 1] = x[:, 1] - x[:, 3] / 2
        box[:, 2] = x[:, 0] + x[:, 2] / 2
        box[:, 3] = x[:, 1] + x[:, 3] / 2

        conf, cls = x[:, 5:].max(1, keepdim=True)
        x = torch.cat([box, conf, cls.float()], 1)
        x = x[x[:, 4] > conf_thres]

        # Batched NMS (per-class offset trick)
        c = x[:, 5:6] * 640
        boxes_shifted = x[:, :4] + c
        from torchvision.ops import nms as tv_nms
        keep = tv_nms(boxes_shifted, x[:, 4], iou_thres)[:max_det]
        output[bi] = x[keep]

    return output


# --------------------------------------------------------------------------- #
# AP per class (VOC 11-point interpolation for speed)
# --------------------------------------------------------------------------- #
def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute AP for each class.
    tp       : (N,) bool — whether each prediction is TP
    conf     : (N,) float
    pred_cls : (N,) int
    target_cls: (M,) int — all GT classes
    Returns: (p, r, ap, f1, unique_classes)
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_cls = np.unique(target_cls)
    ap, p, r = [], [], []

    for c in unique_cls:
        mask = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = mask.sum()

        if n_p == 0 or n_gt == 0:
            ap.append(0)
            p.append(0)
            r.append(0)
            continue

        fpc = (1 - tp[mask]).cumsum()
        tpc = tp[mask].cumsum()
        recall = tpc / (n_gt + 1e-9)
        precision = tpc / (tpc + fpc)

        # AP via 11-point interpolation
        ap_val = 0.0
        for thr in np.linspace(0, 1, 11):
            prec_at = precision[recall >= thr]
            ap_val += (prec_at.max() if prec_at.size else 0) / 11.0
        ap.append(ap_val)
        p.append(precision[-1])
        r.append(recall[-1])

    return np.array(p), np.array(r), np.array(ap), np.array(p) * 0, unique_cls


# --------------------------------------------------------------------------- #
# Full evaluation loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, dataloader, device, split_point=None,
             conf_thres=0.001, iou_thres=0.6, img_size=640):
    """
    Evaluate DMSF on a dataloader.
    split_point: fixed split point for evaluation, or None to use split 10 by default.
    Returns dict with map50, map50_95, mp, mr.
    """
    model.eval()
    model.to(device)

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    stats = []
    pbar = tqdm(dataloader, desc="Evaluating")

    for imgs, targets, _ in pbar:
        imgs = imgs.to(device).float()
        targets = targets.to(device)

        # Inference
        sp = split_point if split_point is not None else 10
        out, _ = model(imgs, split_point=sp)

        # NMS
        preds = non_max_suppression(out, conf_thres, iou_thres)

        # Stats per image
        for bi, det in enumerate(preds):
            tgt = targets[targets[:, 0] == bi][:, 1:]  # (M, 5): cls,cx,cy,w,h
            nl = len(tgt)

            if det is None or len(det) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(),
                                  tgt[:, 0].cpu().numpy()))
                continue

            if nl:
                # Convert GT xywh → xyxy (pixel)
                gt_boxes = tgt[:, 1:5].clone()
                gt_boxes[:, 0] = (tgt[:, 1] - tgt[:, 3] / 2) * img_size
                gt_boxes[:, 1] = (tgt[:, 2] - tgt[:, 4] / 2) * img_size
                gt_boxes[:, 2] = (tgt[:, 1] + tgt[:, 3] / 2) * img_size
                gt_boxes[:, 3] = (tgt[:, 2] + tgt[:, 4] / 2) * img_size
                gt_cls = tgt[:, 0]

                correct = torch.zeros(len(det), niou, dtype=torch.bool, device=device)
                iou_mat = box_iou(gt_boxes, det[:, :4])
                for j, iou_thr in enumerate(iouv):
                    x = torch.where((iou_mat >= iou_thr) &
                                    (gt_cls[:, None] == det[:, 5][None, :]))
                    if x[0].shape[0]:
                        matches = torch.cat([torch.stack(x, 1),
                                             iou_mat[x[0], x[1]][:, None]], 1)
                        matches = matches[matches[:, 2].argsort(descending=True)]
                        matches = matches[np.unique(matches[:, 1].cpu(), return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0].cpu(), return_index=True)[1]]
                        correct[matches[:, 1].long(), j] = True
            else:
                correct = torch.zeros(len(det), niou, dtype=torch.bool)

            stats.append((correct.cpu(),
                          det[:, 4].cpu(),
                          det[:, 5].cpu(),
                          tgt[:, 0].cpu().numpy() if nl else np.array([])))

    if not stats:
        return {"map50": 0.0, "map50_95": 0.0, "mp": 0.0, "mr": 0.0}

    stats_np = [np.concatenate(x, 0) for x in zip(*[
        (s[0].numpy(), s[1].numpy(), s[2].numpy(), s[3]) for s in stats
    ])]
    tp_all, conf_all, pred_cls_all, gt_cls_all = stats_np

    _, _, ap, _, _ = ap_per_class(tp_all[:, 0], conf_all, pred_cls_all, gt_cls_all)
    map50 = ap.mean() if len(ap) else 0.0

    # mAP@0.5:0.95 — average across IoU thresholds
    aps_all = []
    for j in range(niou):
        _, _, ap_j, _, _ = ap_per_class(tp_all[:, j], conf_all, pred_cls_all, gt_cls_all)
        aps_all.append(ap_j.mean() if len(ap_j) else 0.0)
    map5095 = float(np.mean(aps_all))

    return {"map50": float(map50), "map50_95": map5095,
            "mp": 0.0, "mr": 0.0}
