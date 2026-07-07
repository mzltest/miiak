"""Canonical Mii attribute schema.

Field ranges are taken verbatim from PretendoNetwork/mii-js ``src/mii.ts``
(the ``validate()`` method), which is the authoritative, production-tested
description of Ver3 / Mii-Studio attribute ranges.

Two roles:

* ``head``     - a discrete attribute we PREDICT (one classification head each).
* ``nuisance`` - a discrete attribute we RANDOMISE during data generation
                 (for robustness) but do NOT predict.  Per the project spec,
                 all *size / scale / stretch / proportion* parameters are
                 nuisance: their level count is large/unhelpful and the user
                 explicitly asked to skip predicting them while still varying
                 them in the training set.

Every field is an integer in the inclusive range ``[min, max]``.  A head's
class count is ``max - min + 1`` and its training label is ``value - min``
(so e.g. ``eyebrowYPosition`` 3..18 maps to classes 0..15).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Field:
    name: str
    lo: int
    hi: int
    role: str  # "head" | "nuisance"

    @property
    def num_classes(self) -> int:
        return self.hi - self.lo + 1

    def to_label(self, value: int) -> int:
        return int(value) - self.lo

    def to_value(self, label: int) -> int:
        return int(label) + self.lo


# ---------------------------------------------------------------------------
# Full field table.  Order here is purely declarative; the Studio serialisation
# order lives in ``studio.py`` (STUDIO_ORDER) and is independent of this list.
# ---------------------------------------------------------------------------
_FIELDS: List[Field] = [
    # global -----------------------------------------------------------------
    Field("gender", 0, 1, "head"),
    Field("favoriteColor", 0, 11, "head"),
    Field("height", 0, 127, "nuisance"),   # body scale -> nuisance
    Field("build", 0, 127, "nuisance"),    # body scale -> nuisance
    # faceline ---------------------------------------------------------------
    Field("faceType", 0, 11, "head"),
    Field("skinColor", 0, 6, "head"),
    Field("wrinklesType", 0, 11, "head"),
    Field("makeupType", 0, 11, "head"),
    # hair -------------------------------------------------------------------
    Field("hairType", 0, 131, "head"),
    Field("hairColor", 0, 7, "head"),
    Field("flipHair", 0, 1, "head"),
    # eyes -------------------------------------------------------------------
    Field("eyeType", 0, 59, "head"),
    Field("eyeColor", 0, 5, "head"),
    Field("eyeScale", 0, 7, "nuisance"),
    Field("eyeVerticalStretch", 0, 6, "nuisance"),
    Field("eyeRotation", 0, 7, "head"),
    Field("eyeSpacing", 0, 12, "head"),
    Field("eyeYPosition", 0, 18, "head"),
    # eyebrows ---------------------------------------------------------------
    Field("eyebrowType", 0, 24, "head"),
    Field("eyebrowColor", 0, 7, "head"),
    Field("eyebrowScale", 0, 8, "nuisance"),
    Field("eyebrowVerticalStretch", 0, 6, "nuisance"),
    Field("eyebrowRotation", 0, 11, "head"),
    Field("eyebrowSpacing", 0, 12, "head"),
    Field("eyebrowYPosition", 3, 18, "head"),
    # nose -------------------------------------------------------------------
    Field("noseType", 0, 17, "head"),
    Field("noseScale", 0, 8, "nuisance"),
    Field("noseYPosition", 0, 18, "head"),
    # mouth ------------------------------------------------------------------
    Field("mouthType", 0, 35, "head"),
    Field("mouthColor", 0, 4, "head"),
    Field("mouthScale", 0, 8, "nuisance"),
    Field("mouthHorizontalStretch", 0, 6, "nuisance"),
    Field("mouthYPosition", 0, 18, "head"),
    # facial hair ------------------------------------------------------------
    Field("mustacheType", 0, 5, "head"),
    Field("beardType", 0, 5, "head"),
    Field("facialHairColor", 0, 7, "head"),
    Field("mustacheScale", 0, 8, "nuisance"),
    Field("mustacheYPosition", 0, 16, "head"),
    # glasses ----------------------------------------------------------------
    Field("glassesType", 0, 8, "head"),
    Field("glassesColor", 0, 5, "head"),
    Field("glassesScale", 0, 7, "nuisance"),
    Field("glassesYPosition", 0, 20, "head"),
    # mole -------------------------------------------------------------------
    Field("moleEnabled", 0, 1, "head"),
    Field("moleScale", 0, 8, "nuisance"),
    Field("moleXPosition", 0, 16, "head"),
    Field("moleYPosition", 0, 30, "head"),
]

FIELDS: Dict[str, Field] = {f.name: f for f in _FIELDS}

# Ordered names so head order is deterministic everywhere (model, losses, infer).
HEAD_FIELDS: List[str] = [f.name for f in _FIELDS if f.role == "head"]
NUISANCE_FIELDS: List[str] = [f.name for f in _FIELDS if f.role == "nuisance"]
ALL_FIELDS: List[str] = [f.name for f in _FIELDS]

# {head_name: num_classes} -- consumed by the model to build heads.
HEAD_NUM_CLASSES: Dict[str, int] = {n: FIELDS[n].num_classes for n in HEAD_FIELDS}


def assert_valid(fields: Dict[str, int]) -> None:
    """Raise if any field is missing or out of its declared range."""
    for name, f in FIELDS.items():
        if name not in fields:
            raise ValueError(f"missing field {name!r}")
        v = fields[name]
        if not (f.lo <= int(v) <= f.hi):
            raise ValueError(f"{name}={v} out of range [{f.lo},{f.hi}]")


if __name__ == "__main__":
    print(f"total fields : {len(ALL_FIELDS)}")
    print(f"head fields  : {len(HEAD_FIELDS)}")
    print(f"nuisance     : {len(NUISANCE_FIELDS)}")
    total_logits = sum(HEAD_NUM_CLASSES.values())
    print(f"sum of head classes (total output logits): {total_logits}")
    for n in HEAD_FIELDS:
        print(f"  head {n:<24} classes={HEAD_NUM_CLASSES[n]}")
