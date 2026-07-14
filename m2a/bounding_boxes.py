def compute_bounding_boxes(fields, width):
    """
    Computes pseudo-bounding boxes [x1, y1, x2, y2] for parts based on the MockRenderer logic.
    Since Mii renders center the face, we use similar calculations to extract the box.
    """
    S = width
    cx = S // 2
    boxes = {}

    # Face shape dimensions roughly
    fw = int(S * (0.46 + 0.02 * (int(fields["faceType"]) % 4)))
    fh = int(S * 0.60)

    # Hair: roughly the top of the head
    hh = int(S * (0.06 + 0.0014 * int(fields["hairType"])))
    boxes["hair"] = [cx - fw // 2 - 10, int(S * 0.18) - 10, cx + fw // 2 + 10, int(S * 0.18) + hh + 20]

    # Eyes
    ey = int(S * (0.34 + 0.012 * int(fields["eyeYPosition"])))
    ex = int(S * (0.07 + 0.012 * int(fields["eyeSpacing"])))
    # Both eyes combined bounding box
    boxes["eye"] = [cx - ex - 25, ey - 20, cx + ex + 25, ey + 20]

    # Eyebrows
    by = ey - int(20 + (int(fields["eyebrowYPosition"]) - 3))
    boxes["eyebrow"] = [cx - ex - 25, by - 20, cx + ex + 25, by + 20]

    # Nose
    ny = int(S * (0.50 + 0.008 * int(fields["noseYPosition"])))
    boxes["nose"] = [cx - 20, ny - 10, cx + 20, ny + 25]

    # Mouth
    my = int(S * (0.66 + 0.008 * int(fields["mouthYPosition"])))
    boxes["mouth"] = [cx - 30, my - 20, cx + 30, my + 20]

    # Facial hair (mustache and beard)
    fhy1, fhy2 = S, 0
    fhx1, fhx2 = S, 0
    has_fh = False
    if int(fields["mustacheType"]) > 0:
        fhx1 = min(fhx1, cx - 30)
        fhx2 = max(fhx2, cx + 30)
        fhy1 = min(fhy1, my - 20)
        fhy2 = max(fhy2, my + 5)
        has_fh = True
    if int(fields["beardType"]) > 0:
        fhx1 = min(fhx1, cx - fw // 2 - 10)
        fhx2 = max(fhx2, cx + fw // 2 + 10)
        fhy1 = min(fhy1, my - 40)
        fhy2 = max(fhy2, int(S * 0.84) + 10)
        has_fh = True
    if has_fh:
        boxes["facialHair"] = [fhx1, fhy1, fhx2, fhy2]
    else:
        # Invisible bounding box (out of frame or dummy)
        boxes["facialHair"] = [0, 0, 0, 0]

    # Glasses
    if int(fields["glassesType"]) > 0:
        gy = ey + (int(fields["glassesYPosition"]) - 10)
        boxes["glasses"] = [cx - ex - 30, gy - 25, cx + ex + 30, gy + 25]
    else:
        boxes["glasses"] = [0, 0, 0, 0]

    # Mole
    if int(fields["moleEnabled"]) == 1:
        mxp = cx + int((int(fields["moleXPosition"]) - 8) * 4)
        myp = int(S * 0.62) + int((int(fields["moleYPosition"]) - 15) * 2)
        boxes["mole"] = [mxp - 15, myp - 15, mxp + 15, myp + 15]
    else:
        boxes["mole"] = [0, 0, 0, 0]

    # Clip to [0, S]
    for k in boxes:
        b = boxes[k]
        if b != [0, 0, 0, 0]:
            boxes[k] = [max(0, b[0]), max(0, b[1]), min(S, b[2]), min(S, b[3])]
    return boxes
