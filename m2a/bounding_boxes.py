import torch
from PIL import Image, ImageChops

def _union_bbox(box1, box2):
    if not box1: return box2
    if not box2: return box1
    return [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]

def compute_bounding_boxes_by_diff(fields, opt, randomizer, renderer):
    """
    Computes exact bounding boxes by diffing the original render with renders where
    individual parts are modified/hidden.
    """
    # 1. Base render
    img_base = renderer.render(fields, opt=opt, randomizer=randomizer).convert("RGB")

    boxes = {}
    S = opt.width
    cx = S // 2

    def get_diff_bbox(mod_fields):
        img_mod = renderer.render(mod_fields, opt=opt, randomizer=randomizer).convert("RGB")
        diff = ImageChops.difference(img_base, img_mod)
        box = diff.getbbox()
        if box is None:
            return [0, 0, 0, 0]
        # expand slightly for safety
        pad = 2
        return [max(0, box[0]-pad), max(0, box[1]-pad), min(S, box[2]+pad), min(S, box[3]+pad)]

    # Glasses: hide them
    if fields["glassesType"] > 0:
        f = fields.copy()
        f["glassesType"] = 0
        boxes["glasses"] = get_diff_bbox(f)
    else:
        boxes["glasses"] = [0, 0, 0, 0]

    # Facial hair: hide them
    if fields["mustacheType"] > 0 or fields["beardType"] > 0:
        f = fields.copy()
        f["mustacheType"] = 0
        f["beardType"] = 0
        boxes["facialHair"] = get_diff_bbox(f)
    else:
        boxes["facialHair"] = [0, 0, 0, 0]

    # Mole: hide it
    if fields["moleEnabled"] == 1:
        f = fields.copy()
        f["moleEnabled"] = 0
        boxes["mole"] = get_diff_bbox(f)
    else:
        boxes["mole"] = [0, 0, 0, 0]

    # Eyes: We can't hide them. Change their type to an extreme opposite and scale to max.
    f = fields.copy()
    f["eyeType"] = (fields["eyeType"] + 30) % 60
    f["eyeScale"] = 7
    boxes["eye"] = get_diff_bbox(f)

    # Eyebrow: Change type and scale
    f = fields.copy()
    f["eyebrowType"] = (fields["eyebrowType"] + 12) % 24
    f["eyebrowScale"] = 8
    boxes["eyebrow"] = get_diff_bbox(f)

    # Mouth: Change type and scale
    f = fields.copy()
    f["mouthType"] = (fields["mouthType"] + 17) % 35
    f["mouthScale"] = 8
    boxes["mouth"] = get_diff_bbox(f)

    # Nose: Change type and scale
    f = fields.copy()
    f["noseType"] = (fields["noseType"] + 9) % 17
    f["noseScale"] = 8
    boxes["nose"] = get_diff_bbox(f)

    # Hair: We can't easily isolate hair since removing it (type 0) might not cover the bounding box of the original hair.
    # Instead, we diff with a bald head (type 0). The difference will be the hair itself.
    if fields["hairType"] != 0:
        f = fields.copy()
        f["hairType"] = 0
        boxes["hair"] = get_diff_bbox(f)
    else:
        # If already bald, diff with a big hair
        f = fields.copy()
        f["hairType"] = 20
        boxes["hair"] = get_diff_bbox(f)

    # Validate/cleanup boxes
    for k in boxes:
        if not boxes[k]:
            boxes[k] = [0, 0, 0, 0]

    return boxes
