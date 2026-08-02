"""
YOLOv5 detection loss — box regression (CIoU) + objectness (BCE) + class (BCE).
Compatible with the Detect head output format in models/common.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --------------------------------------------------------------------------- #
# IoU helpers
# --------------------------------------------------------------------------- #
def bbox_iou(box1, box2, xywh=True, CIoU=False):
    """box1/box2: (N,4) xywh or xyxy."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-9
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-9

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + 1e-9
    iou = inter / union

    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + 1e-9
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        v = (4 / math.pi ** 2) * torch.pow(
            torch.atan(w2 / (h2 + 1e-9)) - torch.atan(w1 / (h1 + 1e-9)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + v) + 1e-9)
        return iou - (rho2 / c2 + v * alpha)
    return iou


# --------------------------------------------------------------------------- #
# ComputeLoss
# --------------------------------------------------------------------------- #
class ComputeLoss:
    def __init__(self, model, nc: int = 80):
        device = next(model.parameters()).device
        self.nc = nc
        self.na = model.detect.na
        self.nl = model.detect.nl
        self.anchors = model.detect.anchors   # (nl, na, 2)
        self.device = device

        self.bce_obj = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean').to(device)

        # Loss weights (YOLOv5s defaults)
        self.box_gain = 0.05
        self.obj_gain = 1.0
        self.cls_gain = 0.5

        # Balance weights per detection scale
        self.balance = [4.0, 1.0, 0.4]

        # Anchor threshold
        self.anchor_t = 4.0

    def __call__(self, preds, targets):
        """
        preds   : list of tensors (nl,) each (bs, na, ny, nx, no)  [training mode]
        targets : (N, 6) — batch_idx, cls, cx, cy, w, h (normalised)
        """
        bs = preds[0].shape[0]
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)

        tcls, tbox, indices, anch = self._build_targets(preds, targets)

        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=self.device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]   # (n, no)

                # Box regression (CIoU)
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat([pxy, pwh], 1)
                iou = bbox_iou(pbox, tbox[i], xywh=True, CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                # Objectness target = iou (detached)
                tobj[b, a, gj, gi] = iou.detach().clamp(0).float()

                # Class loss
                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 5:], device=self.device)
                    t[range(n), tcls[i].long()] = 1.0
                    lcls += self.bce_cls(ps[:, 5:], t)

            lobj += self.bce_obj(pi[..., 4], tobj) * self.balance[i]

        lbox *= self.box_gain
        lobj *= self.obj_gain
        lcls *= self.cls_gain
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat([lbox, lobj, lcls]).detach()

    def _build_targets(self, preds, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets_rep = torch.cat([targets.repeat(na, 1, 1), ai[:, :, None]], 2)

        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                            device=self.device).float() * g

        for i in range(self.nl):
            anchors_i = self.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape, device=self.device)[[3, 2, 3, 2]].float()
            t = targets_rep * gain

            if nt:
                r = t[:, :, 4:6] / anchors_i[:, None]
                j = torch.max(r, 1. / r).max(2).values < self.anchor_t
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j_idx = torch.stack([torch.ones_like(j), j, k, l, m])
                t = t.repeat(5, 1, 1)[j_idx]
                offsets = torch.zeros_like(gxy)[None] + off[:, None]
                offsets = offsets[j_idx]
            else:
                t = targets_rep[0]
                offsets = 0

            bc, gxy, gwh, a = t.chunk(4, 1)
            a = a.long().view(-1)
            b, c = bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            gi = gi.clamp_(0, int(gain[2]) - 1)
            gj = gj.clamp_(0, int(gain[3]) - 1)

            indices.append((b, a, gj, gi))
            tbox.append(torch.cat([gxy - gij, gwh], 1))
            anch.append(anchors_i[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
