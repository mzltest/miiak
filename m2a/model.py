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

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None
    _timm_err = e

from .schema import HEAD_FIELDS, HEAD_NUM_CLASSES


class MultiHeadMiiNet(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True,
                 head_hidden: int = 0, dropout: float = 0.1,
                 head_num_classes: Optional[Dict[str, int]] = None):
        super().__init__()
        if timm is None:  # pragma: no cover
            raise RuntimeError(f"timm import failed: {_timm_err}")
        self.head_num_classes = dict(head_num_classes or HEAD_NUM_CLASSES)
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        # Determine the real embedding width with a dry forward pass. timm's
        # ``num_features`` can disagree with the pooled output dim for some
        # architectures (e.g. MobileNetV3/LCNet have a conv-head expansion), so
        # probing is robust across all backbones.
        self.backbone.eval()
        with torch.no_grad():
            feat = int(self.backbone(torch.zeros(1, 3, 224, 224)).shape[1])
        self.feat_dim = feat

        self.heads = nn.ModuleDict()
        for name, ncls in self.head_num_classes.items():
            if head_hidden and head_hidden > 0:
                self.heads[name] = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(feat, head_hidden),
                    nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, ncls))
            else:
                self.heads[name] = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(feat, ncls))

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {name: head(feat) for name, head in self.heads.items()}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict) -> MultiHeadMiiNet:
    m = cfg.get("model", {})
    return MultiHeadMiiNet(
        backbone=m.get("backbone", "convnext_tiny"),
        pretrained=m.get("pretrained", True),
        head_hidden=m.get("head_hidden", 0),
        dropout=m.get("dropout", 0.1),
    )


if __name__ == "__main__":
    net = MultiHeadMiiNet(backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print("backbone", net.backbone_name, "feat", net.feat_dim,
          "params", f"{net.num_parameters()/1e6:.1f}M", "heads", len(out))
    for k in list(out)[:3]:
        print(" ", k, tuple(out[k].shape))
