"""Plausible (human-like) Mii sampling.

Ports FFL's ``FFLiDatabaseRandom::Get`` (via AnonymousUser98/mii-creator) — the
algorithm consoles use for the "random Mii" / "look-alike" feature. It picks
part *types* from gender/age/race-conditioned eligibility tables, sets sizes and
positions to sensible defaults, and correlates colours (eyebrow = hair,
facial-hair = hair). The result looks like a real person's Mii rather than the
garbled output of fully-uniform sampling.

Two samplers:
* ``sample_ffl``        - faithful port; sizes/positions are the FFL defaults
                          (low variance), so several heads become near-constant.
* ``sample_plausible``  - ``sample_ffl`` + bounded jitter on sizes / positions /
                          rotations + occasional mole / colour decorrelation.
                          This keeps Miis human-like while making every head a
                          real (non-degenerate) prediction target and giving the
                          size robustness the spec requires. **Default sampler.**
"""

from __future__ import annotations

import random as _random
from typing import Dict, Optional

from . import _ffl_tables as T
from .schema import FIELDS

# enums (match FFL)
GENDER_MALE, GENDER_FEMALE = 0, 1
AGE_CHILD, AGE_ADULT, AGE_ELDER = 0, 1, 2
RACE_BLACK, RACE_WHITE, RACE_ASIAN = 0, 1, 2


def _eye_rot_offset(t: int) -> int:
    return 32 - T.ROTATE_EYE[t]


def _eyebrow_rot_offset(t: int) -> int:
    return 32 - T.ROTATE_EYEBROW[t]


def _pick(rng: _random.Random, entry):
    """entry == [count, [list]]; choose uniformly from the first `count`."""
    count, lst = entry[0], entry[1]
    return int(lst[rng.randrange(count)])


def _determine(rng, gender, age, race):
    if gender is None:
        gender = GENDER_MALE if rng.randrange(2) == 0 else GENDER_FEMALE
    if age is None:
        r = rng.randrange(10)
        age = AGE_CHILD if r < 4 else AGE_ADULT if r < 8 else AGE_ELDER
    if race is None:
        r = rng.randrange(10)
        race = RACE_ASIAN if r < 4 else RACE_WHITE if r < 8 else RACE_BLACK
    return gender, age, race


def _glass_type(rng, age):
    target = rng.randrange(100)
    row = T.RANDOM_GLASS_TYPE[age]
    typ = 0
    while target >= row[typ]:
        typ += 1
    return typ


def sample_ffl(rng: _random.Random, gender: Optional[int] = None,
               age: Optional[int] = None, race: Optional[int] = None,
               favorite_color: Optional[int] = None, hair_color: Optional[int] = None,
               eye_color: Optional[int] = None) -> Dict[str, int]:
    gender, age, race = _determine(rng, gender, age, race)
    f: Dict[str, int] = {}
    base = rng.randrange(3) if (gender == GENDER_FEMALE or age == AGE_CHILD) else 0

    f["faceType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_FACE_TYPE[gender][age][race])
    f["skinColor"] = _pick(rng, T.RANDOM_PARTS_ARRAY_FACELINE_COLOR[gender][race])
    f["wrinklesType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_FACE_LINE[gender][age][race])
    f["makeupType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_FACE_MAKEUP[gender][age][race])
    f["hairType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_HAIR_TYPE[gender][age][race])
    f["hairColor"] = hair_color if hair_color is not None else \
        _pick(rng, T.RANDOM_PARTS_ARRAY_HAIR_COLOR[race][age])
    f["flipHair"] = rng.randrange(2)

    f["eyeType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_EYE_TYPE[gender][age][race])
    f["eyeColor"] = eye_color if eye_color is not None else \
        _pick(rng, T.RANDOM_PARTS_ARRAY_EYE_COLOR[race])
    f["eyeScale"] = 4
    f["eyeVerticalStretch"] = 3
    if gender == GENDER_MALE:
        eye_rot, tgt = 4, _eye_rot_offset(2)
    else:
        eye_rot, tgt = 3, _eye_rot_offset(4)
    f["eyeSpacing"] = 2
    f["eyeYPosition"] = base + 12
    f["eyeRotation"] = eye_rot + tgt - _eye_rot_offset(f["eyeType"])

    f["eyebrowType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_EYEBROW_TYPE[gender][age][race])
    f["eyebrowColor"] = f["hairColor"]
    f["eyebrowScale"] = 4
    f["eyebrowVerticalStretch"] = 3
    if race == RACE_ASIAN:
        f["eyebrowYPosition"], eb_tgt = base + 9, _eyebrow_rot_offset(6)
    else:
        f["eyebrowYPosition"], eb_tgt = base + 10, _eyebrow_rot_offset(0)
    f["eyebrowSpacing"] = 2
    f["eyebrowRotation"] = 6 + eb_tgt - _eyebrow_rot_offset(f["eyebrowType"])

    f["noseType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_NOSE_TYPE[gender][age][race])
    f["noseScale"] = 4 if gender == GENDER_MALE else 3
    f["noseYPosition"] = base + 9

    f["mouthType"] = _pick(rng, T.RANDOM_PARTS_ARRAY_MOUTH_TYPE[gender][age][race])
    f["mouthColor"] = 0 if gender == GENDER_MALE else rng.randrange(5)
    f["mouthScale"] = 4
    f["mouthHorizontalStretch"] = 3
    f["mouthYPosition"] = base + 13

    # facial hair: only adult/elder males, ~20% chance
    if gender == GENDER_MALE and age in (AGE_ADULT, AGE_ELDER) and rng.randrange(10) < 2:
        mustache_type, random_beard = 0, False
        r = rng.randrange(3)
        if r == 0:
            random_beard = True
        elif r == 1:
            mustache_type = rng.randrange(5) + 1
        else:  # r == 2: both (TS fall-through)
            random_beard = True
            mustache_type = rng.randrange(5) + 1
        beard_type = rng.randrange(5) + 1 if random_beard else 0
        mustache_y = 10
    else:
        mustache_type, beard_type, mustache_y = 0, 0, base + 10
    f["mustacheType"] = mustache_type
    f["beardType"] = beard_type
    f["facialHairColor"] = f["hairColor"]
    f["mustacheScale"] = 4
    f["mustacheYPosition"] = mustache_y

    f["glassesType"] = _glass_type(rng, age)
    f["glassesColor"] = 0
    f["glassesScale"] = 4
    f["glassesYPosition"] = base + 10

    f["moleEnabled"] = 0
    f["moleScale"] = 4
    f["moleXPosition"] = 2
    f["moleYPosition"] = 20

    f["height"] = 64
    f["build"] = 64
    f["gender"] = gender
    f["favoriteColor"] = favorite_color if favorite_color is not None else rng.randrange(12)
    return f


# ---------------------------------------------------------------------------
# jitter bands (applied around FFL defaults to make heads non-degenerate)
# ---------------------------------------------------------------------------
_JITTER_POS = {
    "eyeYPosition": 5, "eyebrowYPosition": 4, "noseYPosition": 5, "mouthYPosition": 4,
    "glassesYPosition": 5, "mustacheYPosition": 4, "eyeSpacing": 4, "eyebrowSpacing": 4,
    "eyeRotation": 2, "eyebrowRotation": 2,
}
_JITTER_SIZE = {
    "eyeScale": 3, "eyebrowScale": 3, "noseScale": 3, "mouthScale": 3,
    "glassesScale": 3, "mustacheScale": 3, "moleScale": 3,
    "eyeVerticalStretch": 2, "eyebrowVerticalStretch": 2, "mouthHorizontalStretch": 2,
}
_JITTER_SCALE = {"none": 0.0, "light": 0.5, "medium": 1.0, "heavy": 1.6}


def _jit(rng, f, name, d):
    if d <= 0:
        return
    fld = FIELDS[name]
    f[name] = max(fld.lo, min(fld.hi, f[name] + rng.randint(-d, d)))


def sample_plausible(rng: _random.Random, jitter: str = "medium",
                     mole_prob: float = 0.25, decorrelate_prob: float = 0.15,
                     gender: Optional[int] = None, age: Optional[int] = None,
                     race: Optional[int] = None) -> Dict[str, int]:
    f = sample_ffl(rng, gender=gender, age=age, race=race)
    scale = _JITTER_SCALE.get(jitter, 1.0)
    if scale <= 0:
        return f
    for name, d in {**_JITTER_POS, **_JITTER_SIZE}.items():
        _jit(rng, f, name, max(1, round(d * scale)))
    _jit(rng, f, "height", max(1, int(40 * scale)))
    _jit(rng, f, "build", max(1, int(40 * scale)))
    # enable a mole sometimes so the mole heads see positive examples
    if rng.random() < mole_prob:
        f["moleEnabled"] = 1
        f["moleXPosition"] = rng.randint(0, 16)
        f["moleYPosition"] = rng.randint(8, 28)
    # occasional colour decorrelation for head learnability
    if rng.random() < decorrelate_prob:
        f["eyebrowColor"] = rng.randint(0, 7)
    if rng.random() < decorrelate_prob:
        f["facialHairColor"] = rng.randint(0, 7)
    if f["glassesType"] != 0:                 # glasses colour only matters with glasses
        f["glassesColor"] = rng.randint(0, 5)
    return f


if __name__ == "__main__":
    from .schema import assert_valid, HEAD_FIELDS
    rng = _random.Random(0)
    seen = {h: set() for h in HEAD_FIELDS}
    for _ in range(4000):
        f = sample_plausible(rng)
        assert_valid(f)
        for h in HEAD_FIELDS:
            seen[h].add(f[h])
    print("validity OK over 4000 plausible miis")
    print("distinct values seen per head (head: count):")
    for h in HEAD_FIELDS:
        print(f"  {h:<22}{len(seen[h]):>4}  e.g. {sorted(list(seen[h]))[:8]}")
