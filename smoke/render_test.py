"""Validate the Python Studio encoder against the live FFL render server."""
import io, sys, random, urllib.request, urllib.parse
from PIL import Image
sys.path.insert(0, "/agent/workspace/mii2attr")
from m2a.studio import sample_fields, studio_hex, decode_studio, encode_studio

BASE = "https://mii-unsecure.ariankordi.net/miis/image.png"

def render(hexdata, width=256, expression="normal"):
    q = urllib.parse.urlencode({
        "data": hexdata, "width": width, "type": "face",
        "expression": expression, "bgColor": "FFFFFFFF", "clothesColor": "default",
    })
    with urllib.request.urlopen(f"{BASE}?{q}", timeout=40) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGBA")

rng = random.Random(7)
tiles, ok = [], 0
for i in range(6):
    f = sample_fields(rng)
    # decode-reencode determinism check at randomizer 0
    h = studio_hex(f, randomizer=0)
    assert decode_studio(bytes.fromhex(h)) == f, "decode mismatch"
    try:
        im = render(h)
        ok += 1
        print(f"#{i} OK face={f['faceType']:>2} hair={f['hairType']:>3} eye={f['eyeType']:>2} "
              f"glasses={f['glassesType']} beard={f['beardType']} mole={f['moleEnabled']} size={im.size}")
    except Exception as e:
        print(f"#{i} RENDER FAIL: {e}")
        im = Image.new("RGBA", (256, 256), (255, 0, 0, 255))
    tiles.append(im)

# 3x2 montage on white
W = H = 256
sheet = Image.new("RGBA", (3 * W, 2 * H), (255, 255, 255, 255))
for idx, im in enumerate(tiles):
    if im.size != (W, H):
        im = im.resize((W, H))
    sheet.paste(im, ((idx % 3) * W, (idx // 3) * H), im)
sheet.convert("RGB").save("/agent/workspace/mii2attr/smoke/gen_montage.png")
print(f"rendered {ok}/6 ; montage saved")
