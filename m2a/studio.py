"""Mii Studio (de)serialisation + random Mii sampling.

``encode_studio`` is a verbatim port of PretendoNetwork/mii-js
``Mii.encodeStudio()`` (``src/mii.ts``).  The produced 47-byte payload, hex
encoded, is exactly what the Mii Studio render API *and* the ariankordi
FFL-Testing render server accept as the ``data`` query parameter.

Cipher (mii-js):
    data[0] = randomizer (0..255)
    next = randomizer
    for each part value v (in STUDIO_ORDER):
        enc = (7 + (v ^ next)) & 0xFF
        next = enc
        append enc

Some fields are remapped to a "studio common colour" space before the cipher
(e.g. ``eyeColor + 8``).  ``decode_studio`` inverts both the cipher and the
colour remaps, recovering the original Mii field values.
"""

from __future__ import annotations

import random as _random
from typing import Dict, List, Optional, Set

from .schema import ALL_FIELDS, FIELDS, HEAD_FIELDS, NUISANCE_FIELDS, assert_valid

STUDIO_LEN = 0x2F  # 47 bytes


# ---------------------------------------------------------------------------
# Forward colour remaps (original field value -> studio-encoded value)
# ---------------------------------------------------------------------------
def _map_facial_hair_color(v: int) -> int:
    return 8 if v == 0 else v


def _map_eyebrow_color(v: int) -> int:
    return 8 if v == 0 else v


def _map_hair_color(v: int) -> int:
    return 8 if v == 0 else v


def _map_eye_color(v: int) -> int:
    return v + 8


def _map_glasses_color(v: int) -> int:
    if v == 0:
        return 8
    elif v < 6:
        return v + 13
    return 0


def _map_mouth_color(v: int) -> int:
    return v + 19 if v < 4 else 0


# ---------------------------------------------------------------------------
# STUDIO_ORDER: the 46 parts in the exact order mii-js writes them.
# Each entry is (field_name, transform_or_None).
# ---------------------------------------------------------------------------
STUDIO_ORDER = [
    ("facialHairColor", _map_facial_hair_color),
    ("beardType", None),
    ("build", None),
    ("eyeVerticalStretch", None),
    ("eyeColor", _map_eye_color),
    ("eyeRotation", None),
    ("eyeScale", None),
    ("eyeType", None),
    ("eyeSpacing", None),
    ("eyeYPosition", None),
    ("eyebrowVerticalStretch", None),
    ("eyebrowColor", _map_eyebrow_color),
    ("eyebrowRotation", None),
    ("eyebrowScale", None),
    ("eyebrowType", None),
    ("eyebrowSpacing", None),
    ("eyebrowYPosition", None),
    ("skinColor", None),
    ("makeupType", None),
    ("faceType", None),
    ("wrinklesType", None),
    ("favoriteColor", None),
    ("gender", None),
    ("glassesColor", _map_glasses_color),
    ("glassesScale", None),
    ("glassesType", None),
    ("glassesYPosition", None),
    ("hairColor", _map_hair_color),
    ("flipHair", None),
    ("hairType", None),
    ("height", None),
    ("moleScale", None),
    ("moleEnabled", None),
    ("moleXPosition", None),
    ("moleYPosition", None),
    ("mouthHorizontalStretch", None),
    ("mouthColor", _map_mouth_color),
    ("mouthScale", None),
    ("mouthType", None),
    ("mouthYPosition", None),
    ("mustacheScale", None),
    ("mustacheType", None),
    ("mustacheYPosition", None),
    ("noseScale", None),
    ("noseType", None),
    ("noseYPosition", None),
]
assert len(STUDIO_ORDER) == 46, len(STUDIO_ORDER)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_studio(fields: Dict[str, int], randomizer: Optional[int] = None,
                  rng: Optional[_random.Random] = None) -> bytes:
    """Serialise Mii ``fields`` to 47 raw Studio bytes.

    ``randomizer`` may be pinned (0..255) for deterministic output; otherwise a
    random seed byte is drawn (matching mii-js behaviour).
    """
    assert_valid(fields)
    if randomizer is None:
        r = rng.randint(0, 255) if rng is not None else _random.randint(0, 255)
    else:
        r = int(randomizer) & 0xFF

    out = bytearray(STUDIO_LEN)
    out[0] = r
    nxt = r
    pos = 1
    for name, transform in STUDIO_ORDER:
        v = int(fields[name])
        if transform is not None:
            v = transform(v)
        enc = (7 + (v ^ nxt)) & 0xFF
        nxt = enc
        out[pos] = enc
        pos += 1
    return bytes(out)


def studio_hex(fields: Dict[str, int], randomizer: Optional[int] = None,
               rng: Optional[_random.Random] = None) -> str:
    return encode_studio(fields, randomizer=randomizer, rng=rng).hex()


# ---------------------------------------------------------------------------
# Decoding (inverse) -- used for the round-trip self-test and for sanity checks.
# ---------------------------------------------------------------------------
def _inv_facial_hair_color(v: int) -> int:
    return 0 if v == 8 else v


def _inv_eyebrow_color(v: int) -> int:
    return 0 if v == 8 else v


def _inv_hair_color(v: int) -> int:
    return 0 if v == 8 else v


def _inv_eye_color(v: int) -> int:
    return v - 8


def _inv_glasses_color(v: int) -> int:
    if v == 8:
        return 0
    if 14 <= v <= 18:
        return v - 13
    return 0  # unreachable for valid glassesColor (0..5)


def _inv_mouth_color(v: int) -> int:
    if v == 0:
        return 4
    if 19 <= v <= 22:
        return v - 19
    return 0


_INVERSE = {
    "facialHairColor": _inv_facial_hair_color,
    "eyebrowColor": _inv_eyebrow_color,
    "hairColor": _inv_hair_color,
    "eyeColor": _inv_eye_color,
    "glassesColor": _inv_glasses_color,
    "mouthColor": _inv_mouth_color,
}


def decode_studio(data: bytes) -> Dict[str, int]:
    """Inverse of :func:`encode_studio`. Returns original Mii field values."""
    if len(data) != STUDIO_LEN:
        raise ValueError(f"studio data must be {STUDIO_LEN} bytes, got {len(data)}")
    nxt = data[0]
    fields: Dict[str, int] = {}
    pos = 1
    for name, _transform in STUDIO_ORDER:
        enc = data[pos]
        pos += 1
        studio_val = ((enc - 7) & 0xFF) ^ nxt
        nxt = enc
        inv = _INVERSE.get(name)
        fields[name] = inv(studio_val) if inv is not None else studio_val
    return fields


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------
def _neutral(name: str) -> int:
    f = FIELDS[name]
    return (f.lo + f.hi) // 2


def sample_fields(rng: _random.Random, randomize_nuisance: bool = True) -> Dict[str, int]:
    """Draw a uniformly-random valid Mii.

    Head fields are always uniform over their range.  Nuisance (size/scale)
    fields are uniform when ``randomize_nuisance`` (the default, for robustness)
    or set to a neutral midpoint otherwise.
    """
    fields: Dict[str, int] = {}
    for name in ALL_FIELDS:
        f = FIELDS[name]
        if f.role == "nuisance" and not randomize_nuisance:
            fields[name] = _neutral(name)
        else:
            fields[name] = rng.randint(f.lo, f.hi)
    return fields


# ---------------------------------------------------------------------------
# Relevance masking: heads whose attribute is invisible for a given Mii.
# Their labels should be set to ignore_index so they don't pollute the loss.
# ---------------------------------------------------------------------------
def irrelevant_heads(fields: Dict[str, int]) -> Set[str]:
    """Head fields that have no visible effect on this particular Mii.

    Conservative: only clearly-invisible cases.

    * mole off            -> mole X/Y position irrelevant
    * no glasses (type 0) -> glasses colour + Y position irrelevant
    * no mustache (type 0)-> mustache Y position irrelevant
    * no facial hair      -> facial-hair colour irrelevant
    * hat hair (34, 57)   -> hair colour + flip irrelevant (hats use clothes colour)
    """
    out: Set[str] = set()
    if int(fields["moleEnabled"]) == 0:
        out.update(("moleXPosition", "moleYPosition"))
    if int(fields["glassesType"]) == 0:
        out.update(("glassesColor", "glassesYPosition"))
    if int(fields["mustacheType"]) == 0:
        out.add("mustacheYPosition")
    if int(fields["mustacheType"]) == 0 and int(fields["beardType"]) == 0:
        out.add("facialHairColor")
    if int(fields["hairType"]) in (34, 57):
        out.update(("hairColor", "flipHair"))
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest(n: int = 5000) -> None:
    rng = _random.Random(0)
    bad = 0
    for _ in range(n):
        f = sample_fields(rng)
        for r in (None, 0, 255, rng.randint(0, 255)):
            data = encode_studio(f, randomizer=r)
            assert len(data) == STUDIO_LEN
            back = decode_studio(data)
            for name in ALL_FIELDS:
                if back[name] != f[name]:
                    bad += 1
                    if bad < 10:
                        print(f"MISMATCH {name}: in={f[name]} out={back[name]} (rand={r})")
    if bad:
        raise SystemExit(f"round-trip FAILED: {bad} mismatches")
    print(f"round-trip OK over {n} miis x 4 randomizers ({len(ALL_FIELDS)} fields each)")


if __name__ == "__main__":
    _selftest()
    # show one example payload
    rng = _random.Random(1)
    f = sample_fields(rng)
    print("example studio hex (randomizer=0):", studio_hex(f, randomizer=0))
