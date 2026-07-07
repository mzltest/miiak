"""Renderer backends: Mii Studio data -> image.

Backends
--------
* ``ffl``    - HTTP client for an ariankordi FFL-Testing render server.  This is
               the **default / preferred** backend.  Point ``base_url`` at your
               own local instance on the GPU box (``http://localhost:5000`` after
               ``docker compose up`` in the FFL-Testing repo), or at the author's
               public instance for quick tests.
* ``studio`` - HTTP client for Nintendo's Mii Studio render API.  **Fallback
               only, disabled by default.**  Use only if no FFL server is
               reachable.
* ``mock``   - Fully-offline PIL rasteriser.  Not photorealistic, but a
               deterministic function of the *head* attributes, so the whole ML
               pipeline (and "does it learn?") can be exercised with zero
               network / zero GL.

All backends share the Studio query-parameter vocabulary (type, expression,
width, bgColor, clothesColor, camera/character rotation, ...), taken from
mii-js ``studioUrl``.
"""

from __future__ import annotations

import io
import time
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from PIL import Image, ImageDraw

from .studio import encode_studio

# Public FFL render server (same software users self-host). Override for local.
DEFAULT_FFL_URL = "https://mii-unsecure.ariankordi.net"
STUDIO_API_URL = "https://studio.mii.nintendo.com"

VALID_EXPRESSIONS = (
    "normal", "smile", "anger", "sorrow", "surprise", "blink",
    "normal_open_mouth", "smile_open_mouth", "anger_open_mouth",
    "surprise_open_mouth", "sorrow_open_mouth", "blink_open_mouth",
    "wink_left", "wink_right", "wink_left_open_mouth", "wink_right_open_mouth",
    "like_wink_left", "like_wink_right", "frustrated",
)
VALID_TYPES = ("face", "face_only", "all_body")


@dataclass
class RenderOptions:
    """Per-image render parameters (also used as augmentation knobs)."""
    width: int = 256
    type: str = "face"
    expression: str = "normal"
    bg_color: str = "FFFFFFFF"          # 8 hex chars, UPPERCASE (studio rule)
    clothes_color: str = "default"      # 'default' => favoriteColor drives shirt
    character_x_rotate: int = 0
    character_y_rotate: int = 0
    character_z_rotate: int = 0

    def query(self, data_hex: str) -> Dict[str, str]:
        q = {
            "data": data_hex,
            "width": str(self.width),
            "type": self.type if self.type in VALID_TYPES else "face",
            "expression": self.expression if self.expression in VALID_EXPRESSIONS else "normal",
            "bgColor": self.bg_color,
            "clothesColor": self.clothes_color,
        }
        if self.character_x_rotate:
            q["characterXRotate"] = str(self.character_x_rotate % 360)
        if self.character_y_rotate:
            q["characterYRotate"] = str(self.character_y_rotate % 360)
        if self.character_z_rotate:
            q["characterZRotate"] = str(self.character_z_rotate % 360)
        return q


# ---------------------------------------------------------------------------
# Render augmentation: sample RenderOptions that DO NOT change which discrete
# attribute is used (so labels stay valid).  Expression is kept 'normal' by
# default because smile/blink/etc. morph eyes & mouth away from their canonical
# type, which would make eyeType / mouthType ambiguous.
# ---------------------------------------------------------------------------
_BG_PALETTE = ["FFFFFFFF", "E8E8E8FF", "D7E9F7FF", "F7E9D7FF", "E9F7D7FF",
               "EADCF7FF", "202428FF", "F7D7E2FF"]


def sample_render_options(rng: random.Random, width: int = 256,
                          vary_bg: bool = True, vary_pose: bool = False,
                          vary_expression: bool = False) -> RenderOptions:
    opt = RenderOptions(width=width)
    if vary_bg:
        opt.bg_color = rng.choice(_BG_PALETTE)
    if vary_pose:
        # small yaw/pitch only; large rotations hide attributes
        opt.character_y_rotate = rng.choice([0, 0, 10, 350, 20, 340])
        opt.character_x_rotate = rng.choice([0, 0, 8, 352])
    if vary_expression:
        opt.expression = rng.choice(VALID_EXPRESSIONS)
    return opt


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class BaseRenderer:
    def render(self, fields: Dict[str, int], opt: Optional[RenderOptions] = None,
               randomizer: Optional[int] = None) -> Image.Image:
        raise NotImplementedError


class HttpRenderer(BaseRenderer):
    """Shared HTTP logic for the FFL server and the Studio API."""

    def __init__(self, base_url: str, timeout: float = 40.0, retries: int = 3,
                 backoff: float = 1.5):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/miis/image.png"
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        import requests  # local import so 'mock' works without requests
        self._session = requests.Session()

    def render(self, fields, opt=None, randomizer=None):
        opt = opt or RenderOptions()
        data_hex = encode_studio(fields, randomizer=randomizer).hex()
        params = opt.query(data_hex)
        last = None
        for attempt in range(self.retries):
            try:
                r = self._session.get(self.endpoint, params=params, timeout=self.timeout)
                r.raise_for_status()
                if not r.content or r.content[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"non-PNG response ({len(r.content)} bytes)")
                return Image.open(io.BytesIO(r.content)).convert("RGBA")
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < self.retries - 1:
                    time.sleep(self.backoff ** attempt)
        raise RuntimeError(f"render failed after {self.retries} tries: {last}")


class FFLRenderer(HttpRenderer):
    def __init__(self, base_url: str = DEFAULT_FFL_URL, **kw):
        super().__init__(base_url, **kw)


class StudioAPIRenderer(HttpRenderer):
    """Nintendo Mii Studio API. Fallback only -- not used unless selected."""
    def __init__(self, base_url: str = STUDIO_API_URL, **kw):
        super().__init__(base_url, **kw)


# ---------------------------------------------------------------------------
# Offline mock renderer (no network, no GL). Deterministic in head attributes.
# ---------------------------------------------------------------------------
_SKIN = [(255, 222, 188), (244, 207, 170), (231, 180, 139), (201, 142, 100),
         (162, 105, 70), (224, 172, 105), (120, 77, 50)]
_HAIR = [(40, 30, 25), (20, 20, 20), (90, 56, 37), (120, 80, 50), (160, 120, 70),
         (200, 170, 120), (210, 210, 210), (150, 40, 40)]
_FAV = [(200, 40, 40), (230, 120, 40), (240, 210, 60), (120, 200, 80),
        (60, 170, 90), (60, 110, 200), (90, 180, 220), (230, 130, 180),
        (150, 90, 200), (120, 80, 50), (240, 240, 240), (30, 30, 30)]


class MockRenderer(BaseRenderer):
    """Cartoon face that depends deterministically on head attributes."""

    def __init__(self, width: int = 256):
        self.width = width

    def render(self, fields, opt=None, randomizer=None):
        S = opt.width if opt else self.width
        img = Image.new("RGBA", (S, S), (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        cx = S // 2
        skin = _SKIN[int(fields["skinColor"]) % len(_SKIN)]
        # face shape: width varies a touch with faceType
        fw = int(S * (0.46 + 0.02 * (int(fields["faceType"]) % 4)))
        fh = int(S * 0.60)
        d.ellipse([cx - fw // 2, int(S * 0.20), cx + fw // 2, int(S * 0.20) + fh], fill=skin)
        # shirt = favoriteColor
        fav = _FAV[int(fields["favoriteColor"]) % len(_FAV)]
        d.rectangle([cx - S // 3, int(S * 0.86), cx + S // 3, S], fill=fav)
        # hair: a cap whose height encodes hairType bucket
        hair = _HAIR[int(fields["hairColor"]) % len(_HAIR)]
        hh = int(S * (0.06 + 0.0014 * int(fields["hairType"])))
        d.rectangle([cx - fw // 2, int(S * 0.18), cx + fw // 2, int(S * 0.18) + hh], fill=hair)
        # eyes: spacing/Y from attributes; rotation tilts a line
        ey = int(S * (0.34 + 0.012 * int(fields["eyeYPosition"])))
        ex = int(S * (0.07 + 0.012 * int(fields["eyeSpacing"])))
        er = (int(fields["eyeColor"]) * 40) % 255
        eshape = int(fields["eyeType"])
        for sgn in (-1, 1):
            x = cx + sgn * ex
            box = [x - 16, ey - 10, x + 16, ey + 10]
            if eshape % 3 == 0:
                d.ellipse(box, fill=(255, 255, 255), outline=(0, 0, 0))
            elif eshape % 3 == 1:
                d.rectangle(box, fill=(255, 255, 255), outline=(0, 0, 0))
            else:
                d.pieslice(box, 200, 340, fill=(255, 255, 255), outline=(0, 0, 0))
            d.ellipse([x - 5, ey - 5, x + 5, ey + 5], fill=(er, 60, 120))
        # eyebrows
        by = ey - int(20 + (int(fields["eyebrowYPosition"]) - 3))
        ebc = _HAIR[int(fields["eyebrowColor"]) % len(_HAIR)]
        for sgn in (-1, 1):
            x = cx + sgn * ex
            tilt = (int(fields["eyebrowRotation"]) - 6) * sgn
            d.line([x - 16, by + tilt, x + 16, by - tilt], fill=ebc, width=4)
        # nose
        ny = int(S * (0.50 + 0.008 * int(fields["noseYPosition"])))
        d.polygon([(cx, ny), (cx - 8, ny + 14), (cx + 8, ny + 14)],
                  fill=tuple(max(0, c - 30) for c in skin))
        # mouth: color + Y + type
        my = int(S * (0.66 + 0.008 * int(fields["mouthYPosition"])))
        mc = [(190, 90, 90), (210, 120, 120), (200, 70, 70), (170, 60, 60), (150, 50, 50)][int(fields["mouthColor"]) % 5]
        mt = int(fields["mouthType"])
        if mt % 3 == 0:
            d.arc([cx - 22, my - 12, cx + 22, my + 12], 20, 160, fill=mc, width=5)
        elif mt % 3 == 1:
            d.rectangle([cx - 18, my - 4, cx + 18, my + 4], fill=mc)
        else:
            d.ellipse([cx - 16, my - 10, cx + 16, my + 10], fill=mc)
        # facial hair
        if int(fields["mustacheType"]) > 0:
            fc = _HAIR[int(fields["facialHairColor"]) % len(_HAIR)]
            d.rectangle([cx - 20, my - 12, cx + 20, my - 6], fill=fc)
        if int(fields["beardType"]) > 0:
            fc = _HAIR[int(fields["facialHairColor"]) % len(_HAIR)]
            d.arc([cx - fw // 2, my - 30, cx + fw // 2, int(S * 0.84)], 20, 160, fill=fc, width=8)
        # glasses
        if int(fields["glassesType"]) > 0:
            gc = [(60, 60, 60), (90, 60, 40), (160, 40, 40), (40, 80, 160),
                  (40, 140, 90), (140, 90, 200)][int(fields["glassesColor"]) % 6]
            gy = ey + (int(fields["glassesYPosition"]) - 10)
            for sgn in (-1, 1):
                x = cx + sgn * ex
                d.ellipse([x - 18, gy - 14, x + 18, gy + 14], outline=gc, width=3)
        # mole
        if int(fields["moleEnabled"]) == 1:
            mxp = cx + int((int(fields["moleXPosition"]) - 8) * 4)
            myp = int(S * 0.62) + int((int(fields["moleYPosition"]) - 15) * 2)
            d.ellipse([mxp - 3, myp - 3, mxp + 3, myp + 3], fill=(70, 40, 30))
        return img


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_renderer(kind: str = "ffl", base_url: Optional[str] = None,
                   **kw) -> BaseRenderer:
    kind = kind.lower()
    if kind == "ffl":
        return FFLRenderer(base_url or DEFAULT_FFL_URL, **kw)
    if kind == "studio":
        return StudioAPIRenderer(base_url or STUDIO_API_URL, **kw)
    if kind == "mock":
        return MockRenderer(**{k: v for k, v in kw.items() if k == "width"})
    raise ValueError(f"unknown renderer kind {kind!r} (ffl|studio|mock)")
