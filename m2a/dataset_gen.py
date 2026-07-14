"""Generate a synthetic (image, attribute-label) dataset.

For each sample we draw a uniformly-random valid Mii (nuisance size/scale
params randomised for robustness), encode it to Mii Studio data, render it via
the chosen backend, and write:

* ``<out>/images/<idx>.png``
* one JSON line in ``<out>/manifest.jsonl`` with the head labels (``-100`` where
  a head is irrelevant for that Mii), the full field dict and render options.
* ``<out>/meta.json`` describing the schema + generation parameters.

Usage (CPU smoke test against the public FFL server)::

    python -m m2a.dataset_gen --n 200 --out data/smoke --renderer ffl --workers 8

On the GPU box, point at your own local renderer::

    python -m m2a.dataset_gen --n 200000 --out data/train \
        --renderer ffl --renderer-url http://localhost:5000 --workers 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from .schema import FIELDS, HEAD_FIELDS, HEAD_NUM_CLASSES, NUISANCE_FIELDS
from .studio import sample_fields, irrelevant_heads, encode_studio
from .ffl_random import sample_ffl, sample_plausible
from .renderer import build_renderer, sample_render_options
from .bounding_boxes import compute_bounding_boxes_by_diff

IGNORE = -100

_thread_local = threading.local()


def _get_renderer(kind, base_url, width):
    r = getattr(_thread_local, "renderer", None)
    if r is None:
        kw = {"width": width} if kind == "mock" else {}
        r = build_renderer(kind, base_url=base_url, **kw)
        _thread_local.renderer = r
    return r


def _labels_for(fields: Dict[str, int]) -> Dict[str, int]:
    skip = irrelevant_heads(fields)
    out = {}
    for h in HEAD_FIELDS:
        out[h] = IGNORE if h in skip else FIELDS[h].to_label(fields[h])
    return out


def _sample(rng, args):
    if args.sampler == "uniform":
        return sample_fields(rng, randomize_nuisance=not args.no_randomize_nuisance)
    if args.sampler == "ffl":
        return sample_ffl(rng)
    return sample_plausible(rng, jitter=args.jitter)


def _gen_one(idx, args, split):
    rng = random.Random((args.seed * 1_000_003) ^ idx)
    fields = _sample(rng, args)
    opt = sample_render_options(rng, width=args.width, vary_bg=args.vary_bg,
                                vary_pose=args.vary_pose, vary_expression=args.vary_expression)
    randomizer = rng.randint(0, 255)
    renderer = _get_renderer(args.renderer, args.renderer_url, args.width)
    img = renderer.render(fields, opt=opt, randomizer=randomizer)
    rel = os.path.join("images", f"{idx:07d}.png")
    img.convert("RGB").save(os.path.join(args.out, rel))

    # Compute bounding boxes for the step 2 object detection using image diffing
    boxes = compute_bounding_boxes_by_diff(fields, opt, randomizer, renderer)

    return {
        "file": rel,
        "split": split,
        "labels": _labels_for(fields),
        "boxes": boxes,
        "fields": fields,
        "render": {"width": opt.width, "type": opt.type, "expression": opt.expression,
                   "bgColor": opt.bg_color, "clothesColor": opt.clothes_color,
                   "cYaw": opt.character_y_rotate, "cPitch": opt.character_x_rotate},
        "randomizer": randomizer,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate Mii image->attribute dataset")
    ap.add_argument("--n", type=int, required=True, help="number of samples")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--renderer", default="ffl", choices=["ffl", "studio", "mock"])
    ap.add_argument("--renderer-url", default=None, help="base url for ffl/studio backend")
    ap.add_argument("--sampler", default="plausible", choices=["plausible", "ffl", "uniform"],
                    help="plausible=human-like FFL random+jitter (default); "
                         "ffl=strict FFL random; uniform=full-range random")
    ap.add_argument("--jitter", default="medium", choices=["none", "light", "medium", "heavy"],
                    help="jitter band for --sampler plausible")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--vary-bg", action="store_true", default=True)
    ap.add_argument("--no-vary-bg", dest="vary_bg", action="store_false")
    ap.add_argument("--vary-pose", action="store_true", default=False)
    ap.add_argument("--vary-expression", action="store_true", default=False)
    ap.add_argument("--no-randomize-nuisance", action="store_true", default=False)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    # deterministic split assignment
    idxs = list(range(args.n))
    split_rng = random.Random(args.seed)
    split_rng.shuffle(idxs)
    n_val = int(round(args.n * args.val_frac))
    val_set = set(idxs[:n_val])
    split_of = {i: ("val" if i in val_set else "train") for i in range(args.n)}

    meta = {
        "n": args.n, "renderer": args.renderer, "renderer_url": args.renderer_url,
        "sampler": args.sampler, "jitter": args.jitter,
        "width": args.width, "seed": args.seed, "val_frac": args.val_frac,
        "vary_bg": args.vary_bg, "vary_pose": args.vary_pose,
        "vary_expression": args.vary_expression,
        "randomize_nuisance": not args.no_randomize_nuisance,
        "head_fields": HEAD_FIELDS, "head_num_classes": HEAD_NUM_CLASSES,
        "nuisance_fields": NUISANCE_FIELDS, "ignore_index": IGNORE,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    manifest_path = os.path.join(args.out, "manifest.jsonl")
    t0 = time.time()
    done = fail = 0
    lock = threading.Lock()
    with open(manifest_path, "w") as mf:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_gen_one, i, args, split_of[i]): i for i in range(args.n)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    row = fut.result()
                    with lock:
                        mf.write(json.dumps(row) + "\n")
                        mf.flush()
                    done += 1
                except Exception as e:  # noqa: BLE001
                    fail += 1
                    if fail <= 10:
                        print(f"[warn] sample {i} failed: {e}")
                if (done + fail) % 50 == 0 or (done + fail) == args.n:
                    rate = (done + fail) / max(1e-6, time.time() - t0)
                    print(f"  {done+fail}/{args.n}  ok={done} fail={fail}  {rate:.1f}/s")

    print(f"done: {done} ok, {fail} failed -> {manifest_path}  ({time.time()-t0:.1f}s)")
    if done == 0:
        raise SystemExit("no samples generated; check renderer connectivity / URL")


if __name__ == "__main__":
    main()
