"""
DMSF training script.

Training strategy (Section III-D):
  • Random split point sampled each step from {3, 5, 7, 10}
  • Corresponding compress-recover module activated; identity for others
  • All parameters updated jointly with standard YOLOv5 detection loss
  • 600 epochs, batch=32, input 640×640, YOLOv5s on VisDrone2019

Usage:
  # Prepare VisDrone first
  python -c "from utils.visdrone import prepare_visdrone; \
             prepare_visdrone('data/VisDrone', 'data/visdrone')"

  # Train from scratch
  python train.py --data data/visdrone --epochs 600 --batch 32

  # Fine-tune from pretrained YOLOv5s backbone
  python train.py --data data/visdrone --weights yolov5s.pt --epochs 300 --batch 32
"""

import argparse
import os
import random
import time
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from models.dmsf import DMSF
from utils.visdrone import VisDroneDataset
from utils.loss import ComputeLoss
from utils.metrics import evaluate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',    default='data/visdrone',    help='dataset root')
    p.add_argument('--weights', default='',                 help='pretrained YOLOv5s .pt (optional)')
    p.add_argument('--epochs',  type=int, default=600)
    p.add_argument('--batch',   type=int, default=32)
    p.add_argument('--imgsz',   type=int, default=640)
    p.add_argument('--nc',      type=int, default=10,       help='number of classes')
    p.add_argument('--lr',      type=float, default=0.01)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device',  default='',                 help='cuda:0 or cpu')
    p.add_argument('--save-dir',default='runs/train',       help='output directory')
    p.add_argument('--val-freq',type=int, default=10,       help='validate every N epochs')
    p.add_argument('--save-freq',type=int,default=50)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()

    device = (args.device if args.device
              else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    device = torch.device(device)
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DMSF(nc=args.nc).to(device)
    if args.weights and os.path.exists(args.weights):
        model.load_yolov5s_weights(args.weights)

    # Initialise Detect strides properly (requires a real forward pass)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        out = model(dummy, split_point=10)
        if not model.training:
            pass  # eval mode returns (cat_z, preds)
    model.train()

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds = VisDroneDataset(
        os.path.join(args.data, 'images', 'train'),
        img_size=args.imgsz, augment=True)
    val_ds = VisDroneDataset(
        os.path.join(args.data, 'images', 'val'),
        img_size=args.imgsz, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=VisDroneDataset.collate_fn, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch // 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=VisDroneDataset.collate_fn)

    # ── Optimiser + Scheduler ────────────────────────────────────────────────
    # Separate param groups: backbone/neck (weight decay), biases (no decay)
    pg0, pg1, pg2 = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.bias' in name:
            pg2.append(param)
        elif '.weight' in name and '.bn' not in name:
            pg1.append(param)
        else:
            pg0.append(param)

    optimizer = optim.SGD(pg0, lr=args.lr, momentum=0.937, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})
    optimizer.add_param_group({'params': pg2, 'weight_decay': 0.0})

    # Cosine LR annealing
    lf = lambda x: ((1 - x / args.epochs) * (1 - 0.1) + 0.1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    compute_loss = ComputeLoss(model, nc=args.nc)

    # ── Training loop ────────────────────────────────────────────────────────
    split_candidates = model.split_points   # [3, 5, 7, 10]
    best_map50 = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets, _) in enumerate(train_loader):
            imgs = imgs.to(device).float()
            targets = targets.to(device)

            # Random split point for this step
            sp = random.choice(split_candidates)

            optimizer.zero_grad()
            preds = model(imgs, split_point=sp)   # list of (bs,na,ny,nx,no)
            loss, loss_items = compute_loss(preds, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:4d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}  "
              f"time={elapsed:.1f}s")

        # Validation
        if (epoch + 1) % args.val_freq == 0:
            metrics = evaluate(model, val_loader, device,
                               split_point=10,   # evaluate at neck split (best quality)
                               img_size=args.imgsz)
            map50 = metrics['map50']
            print(f"  [Val] mAP@50={map50:.4f}  mAP@50:95={metrics['map50_95']:.4f}")

            if map50 > best_map50:
                best_map50 = map50
                torch.save({'epoch': epoch + 1,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'map50': map50},
                           save_dir / 'best.pt')
                print(f"  ✓ Saved best.pt  (map50={map50:.4f})")
            model.train()

        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({'epoch': epoch + 1,
                        'model': model.state_dict()},
                       save_dir / f'epoch{epoch+1}.pt')

    # Save final
    torch.save({'epoch': args.epochs,
                'model': model.state_dict()},
               save_dir / 'last.pt')
    print(f"\nTraining complete. Best mAP@50: {best_map50:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
