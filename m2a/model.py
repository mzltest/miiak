"""Multi-task model: shared CNN backbone + one classification head per attribute.

The backbone is any timm model with its classifier removed (``num_classes=0``,
average pooling) so it emits a shared embedding. We attach ``len(HEAD_FIELDS)``
independent heads, each a (optionally hidden) linear classifier over that
attribute's class count.

Recommended backbones: ``convnext_tiny`` (default) or ``resnet50``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.ops as ops

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None
    _timm_err = e

from .schema import HEAD_FIELDS, HEAD_NUM_CLASSES
from .detection_model import BOX_NAMES

# Mapping of part names to their respective heads
PART_HEADS = {
    "global": ["gender", "favoriteColor", "faceType", "skinColor", "wrinklesType", "makeupType"],
    "hair": ["hairType", "hairColor", "flipHair"],
    "eye": ["eyeType", "eyeColor", "eyeRotation", "eyeSpacing", "eyeYPosition"],
    "eyebrow": ["eyebrowType", "eyebrowColor", "eyebrowRotation", "eyebrowSpacing", "eyebrowYPosition"],
    "nose": ["noseType", "noseYPosition"],
    "mouth": ["mouthType", "mouthColor", "mouthYPosition"],
    "facialHair": ["mustacheType", "beardType", "facialHairColor", "mustacheYPosition"],
    "glasses": ["glassesType", "glassesColor", "glassesYPosition"],
    "mole": ["moleEnabled", "moleXPosition", "moleYPosition"],
}


class MultiHeadMiiNet(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True,
                 head_hidden: int = 0, dropout: float = 0.1,
                 head_num_classes: Optional[Dict[str, int]] = None,
                 two_step: bool = False):
        super().__init__()
        if timm is None:  # pragma: no cover
            raise RuntimeError(f"timm import failed: {_timm_err}")
        self.two_step = two_step
        self.head_num_classes = dict(head_num_classes or HEAD_NUM_CLASSES)
        self.backbone_name = backbone

        # If two step, we create separate backbones/embeddings for each part
        if self.two_step:
            self.backbones = nn.ModuleDict()
            self.feat_dims = {}
            for part in PART_HEADS:
                bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
                self.backbones[part] = bb
                bb.eval()
                with torch.no_grad():
                    input_size = getattr(bb, "default_cfg", {}).get("input_size", (3, 224, 224))
                    feat = int(bb(torch.zeros(1, *input_size)).shape[1])
                self.feat_dims[part] = feat
        else:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
            self.backbone.eval()
            with torch.no_grad():
                input_size = getattr(self.backbone, "default_cfg", {}).get("input_size", (3, 224, 224))
                feat = int(self.backbone(torch.zeros(1, *input_size)).shape[1])
            self.feat_dim = feat

        self.heads = nn.ModuleDict()
        for name, ncls in self.head_num_classes.items():
            feat_dim = self.feat_dims[self._get_part(name)] if self.two_step else self.feat_dim
            if head_hidden and head_hidden > 0:
                self.heads[name] = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(feat_dim, head_hidden),
                    nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, ncls))
            else:
                self.heads[name] = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(feat_dim, ncls))

    def _get_part(self, head_name):
        for part, heads in PART_HEADS.items():
            if head_name in heads:
                return part
        return "global"

    def forward(self, x, boxes=None) -> Dict[str, torch.Tensor]:
        if not self.two_step:
            feat = self.backbone(x)
            return {name: head(feat) for name, head in self.heads.items()}

        # Two-step mode
        out = {}
        B, C, H, W = x.shape

        # Process global first
        global_feat = self.backbones["global"](x)
        for h_name in PART_HEADS["global"]:
            out[h_name] = self.heads[h_name](global_feat)

        # Process each cropped part
        for part in BOX_NAMES:
            if part not in PART_HEADS or not PART_HEADS[part]:
                continue

            # Crop the bounding boxes for this part using ROI align or simple resize.
            # Using simple resize grid_sample per bounding box
            crops = []
            for b_idx in range(B):
                # boxes dict should contain [B, 4] for each part, values are [0, 1] normalized
                box = boxes[part][b_idx]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                # Normalise coordinates for grid_sample (range [-1, 1])
                x_norm1 = x1 * 2 - 1
                y_norm1 = y1 * 2 - 1
                x_norm2 = x2 * 2 - 1
                y_norm2 = y2 * 2 - 1

                # If box is degenerate or inverted, use full image (to avoid NaN gradients)
                if x2 <= x1 or y2 <= y1:
                    x_norm1, y_norm1, x_norm2, y_norm2 = torch.tensor(-1.0, device=x.device), torch.tensor(-1.0, device=x.device), torch.tensor(1.0, device=x.device), torch.tensor(1.0, device=x.device)

                # Create grid for F.grid_sample
                # Resize all crops to 64x64 or the backbone's expected small size
                crop_size = 64

                # We need to maintain gradient flow if boxes were predicted
                # meshgrid doesn't pass gradients from start/end points natively in a simple way
                # Instead, we construct the grid using linspace and scale it
                base_grid_y, base_grid_x = torch.meshgrid(torch.linspace(0, 1, crop_size, device=x.device),
                                                          torch.linspace(0, 1, crop_size, device=x.device), indexing="ij")
                grid_x = x_norm1 + (x_norm2 - x_norm1) * base_grid_x
                grid_y = y_norm1 + (y_norm2 - y_norm1) * base_grid_y

                grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0) # (1, crop_size, crop_size, 2)

                crop = torch.nn.functional.grid_sample(x[b_idx:b_idx+1], grid, align_corners=True)
                crops.append(crop)

            crops = torch.cat(crops, dim=0) # (B, C, crop_size, crop_size)
            part_feat = self.backbones[part](crops)

            for h_name in PART_HEADS[part]:
                out[h_name] = self.heads[h_name](part_feat)

        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict) -> MultiHeadMiiNet:
    m = cfg.get("model", {})
    return MultiHeadMiiNet(
        backbone=m.get("backbone", "convnext_tiny"),
        pretrained=m.get("pretrained", True),
        head_hidden=m.get("head_hidden", 0),
        dropout=m.get("dropout", 0.1),
        two_step=cfg.get("train", {}).get("two_step", False)
    )

if __name__ == "__main__":
    net = MultiHeadMiiNet(backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print("backbone", net.backbone_name, "feat", getattr(net, "feat_dim", "dict"),
          "params", f"{net.num_parameters()/1e6:.1f}M", "heads", len(out))
