import gradio as gr
import torch
import json
from PIL import Image, ImageDraw
import os

from m2a.data import build_transform
from m2a.eval import load_model
from m2a.schema import FIELDS, HEAD_FIELDS, ALL_FIELDS
from m2a.studio import encode_studio, studio_hex
from m2a.renderer import build_renderer, RenderOptions
from m2a.detection_model import BOX_NAMES

CKPT_PATH = "runs/auto/best.pt"
RENDERER_URL = "https://mii-unsecure.ariankordi.net"

def predict_and_render(image):
    if not os.path.exists(CKPT_PATH):
        return None, None, f"Error: Checkpoint not found at {CKPT_PATH}. Please run training first."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, det_model, cfg = load_model(CKPT_PATH, device)
    image_size = int(cfg.get("train", {}).get("image_size", 224))

    # Preprocess image
    tf = build_transform(image_size, train=False)
    # the incoming Gradio image is a PIL Image if we use type="pil"
    img = image.convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    fields = {}
    boxes = None
    with torch.no_grad():
        if det_model is not None:
            boxes = det_model(x)
            out = model(x, boxes)
        elif getattr(model, "two_step", False):
            pass
        else:
            out = model(x)

    for h in HEAD_FIELDS:
        label = int(out[h].argmax(1).item())
        fields[h] = FIELDS[h].to_value(label)

    # fill nuisance with neutral midpoints
    for name in ALL_FIELDS:
        if name not in fields:
            f = FIELDS[name]
            fields[name] = (f.lo + f.hi) // 2

    # Draw bounding boxes if det_model is available
    img_with_boxes = img.copy()
    if boxes is not None:
        draw = ImageDraw.Draw(img_with_boxes)
        w, h = img_with_boxes.size
        for name in BOX_NAMES:
            if name in boxes:
                # The model outputs raw coordinate relative predictions or normalized?
                # According to detection_model.py, it's just a linear layer outputting 4 values per box.
                # In data.py, boxes are scaled by [c / orig_W for c in box].
                # Meaning the targets are in the range [0, 1].
                # So the model predicts values in [0, 1].
                # We need to scale them back to the original image dimensions (w, h).

                box = boxes[name][0].cpu().tolist()
                bx1, by1, bx2, by2 = box

                # Scale from [0, 1] relative coordinates to image size
                bx1, bx2 = bx1 * w, bx2 * w
                by1, by2 = by1 * h, by2 * h

                # don't draw boxes if they are [0,0,0,0] (e.g. absent features)
                if bx2 - bx1 > 1 and by2 - by1 > 1:
                    draw.rectangle([bx1, by1, bx2, by2], outline="red", width=2)
                    draw.text((bx1, by1 - 10), name, fill="red")

    # Generate Chardata
    hex_code = studio_hex(fields, randomizer=0)

    # Render final Mii
    renderer = build_renderer("ffl", base_url=RENDERER_URL)
    # The render option defaults to 256 for width
    try:
        rendered_mii = renderer.render(fields, opt=RenderOptions(width=256), randomizer=0)
    except Exception as e:
        rendered_mii = None
        hex_code += f"\n\nRender failed: {str(e)}"

    return img_with_boxes, rendered_mii, hex_code

demo = gr.Interface(
    fn=predict_and_render,
    inputs=gr.Image(type="pil", label="Upload Mii Image"),
    outputs=[
        gr.Image(type="pil", label="Detected Features"),
        gr.Image(type="pil", label="Rendered Mii"),
        gr.Textbox(label="Chardata (Studio Hex)", interactive=True)
    ],
    title="mii2attr Predictor",
    description="Upload a Mii image to predict its attributes, view bounding boxes, and get the Chardata.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
