"""Evaluate a checkpoint on a dataset split; print a per-head accuracy table.

    python -m m2a.eval --ckpt runs/smoke/best.pt --data data/smoke --split val
"""

from __future__ import annotations

import argparse

import torch

from .data import MiiAttrDataset, collate_fn
from .model import MultiHeadMiiNet
from .losses import MetricAccumulator
from .schema import HEAD_FIELDS, HEAD_NUM_CLASSES


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("cfg", {})
    m = cfg.get("model", {})
    model = MultiHeadMiiNet(backbone=m.get("backbone", "convnext_tiny"),
                            pretrained=False, head_hidden=m.get("head_hidden", 0),
                            dropout=m.get("dropout", 0.1),
                            head_num_classes=ck.get("head_num_classes"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.ckpt, device)

    ds = MiiAttrDataset(args.data, args.split, args.image_size, train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=collate_fn)
    acc = MetricAccumulator()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            acc.update({k: v.cpu() for k, v in out.items()}, yb)

    head_acc = acc.head_acc()
    print(f"\nsplit={args.split}  n={len(ds)}")
    print(f"{'head':<24}{'classes':>8}{'acc':>9}{'chance':>9}")
    print("-" * 50)
    for h in HEAD_FIELDS:
        c = HEAD_NUM_CLASSES[h]
        print(f"{h:<24}{c:>8}{head_acc[h]:>9.3f}{1.0/c:>9.3f}")
    print("-" * 50)
    print(f"{'MEAN':<24}{'':>8}{acc.mean_acc():>9.3f}")
    print(f"exact-match (all visible heads): {acc.exact_match():.4f}")


if __name__ == "__main__":
    main()
