"""Predict Mii attributes from an image, then re-encode + re-render to compare.

    python -m m2a.infer --ckpt runs/smoke/best.pt --image some_mii.png \
        --renderer ffl --out recon.png

Nuisance (size/scale) attributes are not predicted; they are filled with neutral
midpoint values for the reconstruction render.
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from .data import build_transform
from .eval import load_model
from .schema import FIELDS, HEAD_FIELDS, ALL_FIELDS
from .studio import encode_studio
from .renderer import build_renderer, RenderOptions


def predict_fields(model, image: Image.Image, device, image_size=224):
    tf = build_transform(image_size, train=False)
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    fields = {}
    for h in HEAD_FIELDS:
        label = int(out[h].argmax(1).item())
        fields[h] = FIELDS[h].to_value(label)
    # fill nuisance with neutral midpoints
    for name in ALL_FIELDS:
        if name not in fields:
            f = FIELDS[name]
            fields[name] = (f.lo + f.hi) // 2
    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="reconstruction.png")
    ap.add_argument("--renderer", default="ffl", choices=["ffl", "studio", "mock"])
    ap.add_argument("--renderer-url", default=None)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--print-fields", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.ckpt, device)
    image_size = int(cfg.get("train", {}).get("image_size", 224))

    src = Image.open(args.image).convert("RGB")
    fields = predict_fields(model, src, device, image_size)
    if args.print_fields:
        print(json.dumps({h: fields[h] for h in HEAD_FIELDS}, indent=2))

    renderer = build_renderer(args.renderer, base_url=args.renderer_url,
                              **({"width": args.width} if args.renderer == "mock" else {}))
    recon = renderer.render(fields, opt=RenderOptions(width=args.width), randomizer=0)

    W = args.width
    sheet = Image.new("RGB", (2 * W, W), (255, 255, 255))
    sheet.paste(src.resize((W, W)), (0, 0))
    sheet.paste(recon.convert("RGB").resize((W, W)), (W, 0))
    sheet.save(args.out)
    print(f"saved comparison (input | reconstruction) -> {args.out}")


if __name__ == "__main__":
    main()
