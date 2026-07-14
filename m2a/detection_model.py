import torch
import torch.nn as nn
try:
    import timm
except Exception:
    timm = None

# We predict 8 bounding boxes: hair, eye, eyebrow, nose, mouth, facialHair, glasses, mole
# Each bounding box is represented by 4 coordinates (x1, y1, x2, y2)
BOX_NAMES = ["hair", "eye", "eyebrow", "nose", "mouth", "facialHair", "glasses", "mole"]

class BoundingBoxRegressor(nn.Module):
    def __init__(self, backbone="mobilenetv3_large_100", pretrained=True, dropout=0.1):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm import failed")
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")

        self.backbone.eval()
        with torch.no_grad():
            input_size = getattr(self.backbone, "default_cfg", {}).get("input_size", (3, 224, 224))
            feat = int(self.backbone(torch.zeros(1, *input_size)).shape[1])
        self.feat_dim = feat

        # 4 coordinates per box
        self.num_boxes = len(BOX_NAMES)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, self.num_boxes * 4)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        # return a dict of {box_name: (B, 4)}
        boxes = {}
        for i, name in enumerate(BOX_NAMES):
            boxes[name] = out[:, i*4:(i+1)*4]
        return boxes

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

def build_detection_model(cfg: dict) -> BoundingBoxRegressor:
    m = cfg.get("model", {})
    return BoundingBoxRegressor(
        backbone=m.get("backbone", "mobilenetv3_large_100"),
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.1),
    )
