"""Render plausible (human-like) Miis; compare FFL vs Nintendo Studio API."""
import io, sys, random, urllib.request, urllib.parse
from PIL import Image
sys.path.insert(0, "/agent/workspace/mii2attr")
from m2a.ffl_random import sample_plausible
from m2a.studio import studio_hex

FFL = "https://mii-unsecure.ariankordi.net/miis/image.png"
NIN = "https://studio.mii.nintendo.com/miis/image.png"

def render(base, hexdata, width=256):
    q = urllib.parse.urlencode({"data": hexdata, "width": width, "type": "face",
                                "expression": "normal", "bgColor": "FFFFFFFF"})
    try:
        with urllib.request.urlopen(f"{base}?{q}", timeout=40) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGBA")
    except Exception as e:  # noqa: BLE001
        print(f"  RENDER FAIL base={base.split('//')[1][:24]} err={e} data={hexdata}")
        return Image.new("RGBA", (width, width), (255, 0, 0, 255))

rng = random.Random(20)
W = 256
sheet = Image.new("RGBA", (4 * W, 2 * W), (255, 255, 255, 255))
for i in range(8):
    f = sample_plausible(rng)
    h = studio_hex(f, randomizer=0)
    im = render(FFL, h)
    if im.size != (W, W):
        im = im.resize((W, W))
    sheet.paste(im, ((i % 4) * W, (i // 4) * W), im)
    print(f"#{i} gender={f['gender']} face={f['faceType']:>2} hair={f['hairType']:>3} "
          f"eye={f['eyeType']:>2} glasses={f['glassesType']} beard={f['beardType']} "
          f"mustache={f['mustacheType']} mole={f['moleEnabled']}")
sheet.convert("RGB").save("/agent/workspace/mii2attr/smoke/plausible_ffl.png")

# cross-renderer check on one Mii (FFL vs Nintendo)
f = sample_plausible(random.Random(3))
h = studio_hex(f, randomizer=0)
cmp = Image.new("RGB", (2 * W, W), (255, 255, 255))
cmp.paste(render(FFL, h).convert("RGB").resize((W, W)), (0, 0))
cmp.paste(render(NIN, h).convert("RGB").resize((W, W)), (W, 0))
cmp.save("/agent/workspace/mii2attr/smoke/renderer_compare.png")
print("saved plausible_ffl.png (8 plausible) and renderer_compare.png (FFL | Nintendo)")
