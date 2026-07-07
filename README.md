# mii2attr — Mii image → Mii attributes (multi-task CNN)

Given a **rendered Mii image**, predict the **discrete Mii attributes** that
produced it (face/hair/eyes/eyebrows/nose/mouth/facial-hair/glasses/mole types,
colours, rotations and positions). A shared CNN backbone (ConvNeXt-Tiny or
ResNet-50) feeds **34 parallel classification heads**, one per attribute.

This is a *closed-loop synthetic* task: we generate unlimited training data by
sampling random Miis, rendering them with the authentic **FFL** renderer, and
learning to invert the renderer.

## Status — CPU pipeline validated end-to-end
Everything below was smoke-tested on a CPU-only box (no GPU, no GL, no compiler):

* **Studio encoder** ported verbatim from `mii-js`; round-trip verified over
  5000 Miis × 4 randomizers; generated Miis render correctly on the live FFL server.
* **Dataset generator** produced 300 real FFL renders (~22 img/s) with labels.
* **Model / loss / metrics** run; backward works; 11.5M-param ResNet-18 forward
  ~0.35 s/batch(8) on 2 CPU cores.
* **Training loop** (AMP, cosine+warmup, checkpoint/resume, CSV log, DDP-ready)
  runs; loss decreased 85.0→82.6.
* **Overfit sanity test**: on 32 samples, loss 84→0.05 and train accuracy →
  **1.000**, exact-match → **1.000** (proves the architecture learns the mapping).
* **eval** (per-head accuracy table) and **infer** (predict → re-encode → re-render
  comparison) CLIs work.

Real accuracy comes from scaling data + epochs on a GPU (instructions below).

## Quickstart — one command
```bash
bash start.sh
```
Detects GPU/CUDA, installs deps (CUDA or CPU PyTorch), asks how to render
(local FFL via Docker with **automatic resource download + build**, the public
FFL instance, Nintendo Studio API, or offline mock), auto-writes a config matched
to your hardware, generates the dataset, and starts training — streaming loss and
per-epoch validation accuracy. Re-run to resume/reuse.

It **auto-switches pipeline by hardware**: with a GPU it trains ConvNeXt-Tiny
@224 + AMP; with **CPU only** it switches to a lightweight MobileNetV3 @160 +
`channels_last` (≈10x faster on CPU). Pick a CPU profile with `CPU_PROFILE`.

Unattended example:
```bash
AUTO=1 RENDERER=ffl_docker N_SAMPLES=200000 EPOCHS=30 bash start.sh        # GPU box
AUTO=1 RENDERER=ffl_public CPU_PROFILE=balanced N_SAMPLES=40000 bash start.sh   # CPU box
```
Knobs (env vars): `AUTO, RENDERER, RENDERER_URL, SAMPLER, JITTER, N_SAMPLES,
EPOCHS, BATCH, BACKBONE, PRETRAINED, WORKERS, WIDTH, USE_VENV, USE_DDP,
SKIP_DATA, SKIP_TRAIN`. Prefer the manual steps below for full control.

## Architecture
```
image ─► timm backbone (classifier removed, global avg pool) ─► shared embedding
                                                                  │
        ┌────────────┬────────────┬─────────── … 34 heads ───────┤
     Linear(eyeType) Linear(hairType) Linear(mouthColor) …   (one per attribute)
        │            │            │
     CE(60)       CE(132)      CE(5)            loss = Σ per-head cross-entropy
```
* Heads whose attribute is **invisible** for a given Mii (mole off, no glasses,
  no facial hair, hat hair) get label `-100` and are ignored by the loss.
* **Size / scale / stretch / height / build are NOT predicted** (12 "nuisance"
  fields). They ARE randomised in the training set so the model is robust to
  them — exactly per spec. (Flipping any nuisance field to a head in
  `m2a/schema.py` is a one-line change if you later want ~46 heads.)

## Repo layout
```
m2a/
  schema.py       # field ranges + head/nuisance split (single source of truth)
  studio.py       # Mii Studio encode/decode (mii-js port) + uniform sampler + masking
  ffl_random.py   # plausible (human-like) sampler: FFL DatabaseRandom port + jitter
  _ffl_tables.py  # AUTO-GENERATED FFL random part tables (tools/extract_ffl_tables.py)
  renderer.py     # FFL HTTP backend (default) | Studio API (fallback) | offline mock
  dataset_gen.py  # sample → render → (image, labels) + manifest.jsonl
  data.py         # torch Dataset + transforms (no hflip) + collate
  model.py        # timm backbone + ModuleDict of per-attribute heads
  losses.py       # masked multi-task CE + per-head/exact-match metrics
  train.py        # config-driven trainer (AMP, ckpt/resume, DDP via torchrun)
  eval.py         # per-head accuracy table
  infer.py        # image → predicted attrs → re-render comparison
configs/          # smoke_cpu.yaml, cpu.yaml (MobileNetV3), gpu_default.yaml
scripts/          # setup_renderer_gpu.sh (bring up the local FFL renderer)
assets/           # FFLResHigh.dat (staged for the local renderer)
smoke/            # validation scripts + sample outputs
# vendored references: FFL-Testing/ mii-js/ MiiJS/ MiiDataFiles/
```

## Install
```bash
# CPU:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
# GPU: install the CUDA torch build for your CUDA version, then:
pip install -r requirements.txt
```

## 1) Stand up the local FFL renderer (preferred)
The renderer is the FFL render server (`ariankordi/FFL-Testing`), the same
software whether self-hosted or via the author's public instance.
```bash
bash scripts/setup_renderer_gpu.sh      # docker compose up → http://localhost:5000
```
The Nintendo Mii Studio API is supported as a **fallback only** (`--renderer studio`)
and is off by default.

## 2) Generate data
```bash
# Local renderer (human-like Miis, the default sampler):
python -m m2a.dataset_gen --n 200000 --out data/train \
    --renderer ffl --renderer-url http://localhost:5000 --workers 64 --width 224
# Quick test against the public instance (no setup):
python -m m2a.dataset_gen --n 300 --out data/smoke --renderer ffl --workers 6
# Fully offline (no renderer) — cartoon mock, for plumbing tests only:
python -m m2a.dataset_gen --n 300 --out data/mock --renderer mock
```

### Sampling realism (`--sampler`)
Fully-uniform random Miis look garbled and inhuman, so the **default sampler is
`plausible`**: a port of FFL's `FFLiDatabaseRandom` (the console "random Mii" /
"look-alike" generator) that picks part *types* from gender/age/race-conditioned
eligibility tables and correlates colours (eyebrow = hair), **plus** bounded
jitter on sizes/positions/rotations so every head stays a real prediction target
and the model gets size robustness.
* `--sampler plausible` (default) — human-like; `--jitter none|light|medium|heavy`
  (`medium` default; `none` = strict FFL with fixed sizes/positions).
* `--sampler ffl` — strict FFL DatabaseRandom (most realistic, low size/pos variance).
* `--sampler uniform` — full-range random (max coverage, inhuman faces).

Render robustness knobs: `--vary-bg` (on), `--vary-pose`, `--vary-expression`.

### Renderer backends (`--renderer`)
* `ffl` (default) — the FFL render server (local or public); authentic Wii-U/3DS look.
* `studio` — Nintendo's Mii Studio API (`https://studio.mii.nintendo.com`),
  **verified working** with the same Studio encoding; a real fallback (off by
  default per spec). Has retry/backoff for the occasional transient 400.
* `mock` — offline PIL cartoon (no network/GL), for plumbing tests only.
The same `data` (Studio hex) renders identically on both `ffl` and `studio`.

## 3) Train
```bash
# CPU smoke (what was validated here):
python -m m2a.train --config configs/smoke_cpu.yaml          # full 3 epochs
python -m m2a.train --config configs/smoke_cpu.yaml --max-steps 40

# Single GPU (ConvNeXt-Tiny, pretrained):
python -m m2a.train --config configs/gpu_default.yaml --data data/train

# Multi-GPU:
torchrun --nproc_per_node=4 -m m2a.train --config configs/gpu_default.yaml --data data/train
```
Checkpoints (`last.pt`, `best.pt`) and `log.csv` land in `out/`. Resume with
`--resume runs/.../last.pt`.

## 4) Evaluate / infer
```bash
python -m m2a.eval --ckpt runs/convnext_tiny/best.pt --data data/train --split val
python -m m2a.infer --ckpt runs/convnext_tiny/best.pt --image mii.png \
    --renderer ffl --renderer-url http://localhost:5000 --out recon.png --print-fields
```

## CPU training (no GPU)
No GPU? The CPU pipeline uses a small **MobileNetV3** backbone at **160px** with
`channels_last` — roughly **10x faster** than ConvNeXt@224 on CPU, while still
having enough capacity to learn the task. `bash start.sh` selects it
automatically; `CPU_PROFILE` picks the speed/quality point:

| `CPU_PROFILE` | backbone | input | notes |
|---|---|---|---|
| `fast`     | mobilenetv3_small_100 (2.1M) | 128px | fastest; lower ceiling on hard heads |
| `balanced` (default) | mobilenetv3_large_100 (5.0M) | 160px | best speed/accuracy trade-off |
| `quality`  | resnet50 (24.7M) | 192px | slowest, highest capacity |

Measured CPU **training** throughput (this repo's heads, 160px, batch 32, with
`channels_last`; numbers are for **2 cores** — scale ~1.6x for 4 cores):

| backbone | params | img/s (2 cores) |
|---|---|---|
| mobilenetv3_small_100 | 2.1M | ~155 |
| mobilenetv3_large_100 | 5.0M | ~59 |
| resnet18 | 11.5M | ~35 |
| resnet50 | 24.7M | ~10 |
| convnext_tiny @224 | 28.3M | ~5 |

So on a **4-core CPU**, `mobilenetv3_large_100 @160` does **~95 img/s** → an
overnight run (~8–10 h) sees **~2.5–3.5M images**, comparable to a 2-hour T4
session. Manual run:
```bash
python -m m2a.train --config configs/cpu.yaml --data data/train
```
Expect easy heads (gender, colours, glasses/mole presence) to reach ~90%+, with
the hardest many-class heads (hairType 132-way, eyeType 60-way) lower — a useful
model, just below what a GPU + ConvNeXt reaches.

## Attribute schema (34 heads)
gender, favoriteColor, faceType, skinColor, wrinklesType, makeupType, hairType,
hairColor, flipHair, eyeType, eyeColor, eyeRotation, eyeSpacing, eyeYPosition,
eyebrowType, eyebrowColor, eyebrowRotation, eyebrowSpacing, eyebrowYPosition,
noseType, noseYPosition, mouthType, mouthColor, mouthYPosition, mustacheType,
beardType, facialHairColor, mustacheYPosition, glassesType, glassesColor,
glassesYPosition, moleEnabled, moleXPosition, moleYPosition.

Nuisance (randomised, not predicted): height, build, eyeScale,
eyeVerticalStretch, eyebrowScale, eyebrowVerticalStretch, noseScale, mouthScale,
mouthHorizontalStretch, mustacheScale, glassesScale, moleScale.

Ranges and the Studio serialisation are ported verbatim from
`PretendoNetwork/mii-js` (`src/mii.ts`).

## Credits
FFL renderer: `ariankordi/FFL-Testing` (built on AboodXD's FFL decomp + RIO).
Mii data format: `PretendoNetwork/mii-js`, `HEYimHeroic/MiiDataFiles`.
The FFL resource file is a Nintendo asset; provide your own copy.
