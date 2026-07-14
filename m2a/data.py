"""Torch Dataset + transforms for the Mii attribute task.

Note: we deliberately do **not** use horizontal flips. Flipping swaps left/right
and would invalidate orientation-bearing labels (eyeRotation, eyebrowRotation,
moleXPosition, flipHair, eyeSpacing semantics). Augmentation is limited to mild
photometric/affine jitter that does not change which discrete attribute is used.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from .schema import HEAD_FIELDS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_manifest(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_transform(image_size: int, train: bool):
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MiiAttrDataset(Dataset):
    def __init__(self, data_root: str, split: str, image_size: int = 224,
                 train: bool = False, manifest_name: str = "manifest.jsonl"):
        self.root = data_root
        rows = load_manifest(os.path.join(data_root, manifest_name))
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r} in {data_root}")
        self.tf = build_transform(image_size, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.root, r["file"])).convert("RGB")
        x = self.tf(img)
        labels = {h: int(r["labels"][h]) for h in HEAD_FIELDS}
        item = (x, labels)
        if "boxes" in r:
            # Normalize boxes based on the original width since transforms resize the image
            # Render width was stored in r["render"]["width"]
            orig_W = r["render"]["width"]
            scaled_boxes = {}
            for k, box in r["boxes"].items():
                scaled_boxes[k] = [c / orig_W for c in box]
            item = (x, labels, scaled_boxes)
        return item


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], 0)
    labels = {h: torch.tensor([b[1][h] for b in batch], dtype=torch.long)
              for h in HEAD_FIELDS}
    if len(batch[0]) > 2:
        boxes_list = [b[2] for b in batch]
        boxes = {}
        for k in boxes_list[0].keys():
            boxes[k] = torch.tensor([box[k] for box in boxes_list], dtype=torch.float)
        return xs, labels, boxes
    return xs, labels


def make_loaders(data_root, image_size, batch_size, num_workers, train_split="train",
                 val_split="val"):
    from torch.utils.data import DataLoader
    tr = MiiAttrDataset(data_root, train_split, image_size, train=True)
    va = MiiAttrDataset(data_root, val_split, image_size, train=False)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    collate_fn=collate_fn, drop_last=True, pin_memory=torch.cuda.is_available())
    vl = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    return tr, va, tl, vl
