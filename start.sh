#!/usr/bin/env bash
# =============================================================================
# mii2attr one-shot launcher.
#
#   bash start.sh
#
# Detects GPU/CUDA, installs deps, sets up a renderer (local FFL via Docker with
# auto resource download + build, or public/Nintendo/mock), auto-writes a config
# matched to your hardware, generates a dataset, and starts training (printing
# loss + per-epoch validation accuracy).
#
# Fully interactive by default. For unattended runs set AUTO=1 and/or any of:
#   RENDERER=ffl_docker|ffl_native|ffl_public|studio|mock   RENDERER_URL=...
#   SAMPLER=plausible|ffl|uniform   JITTER=none|light|medium|heavy
#   N_SAMPLES, EPOCHS, BATCH, BACKBONE, PRETRAINED, WORKERS, WIDTH, USE_VENV,
#   USE_DDP, SKIP_TRAIN, SKIP_DATA
# e.g.:  AUTO=1 RENDERER=ffl_public N_SAMPLES=2000 bash start.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

c_cy=$'\033[1;36m'; c_ye=$'\033[1;33m'; c_rd=$'\033[1;31m'; c_mg=$'\033[1;35m'; c_0=$'\033[0m'
log()  { printf "%s[start]%s %s\n" "$c_cy" "$c_0" "$*"; }
warn() { printf "%s[warn]%s %s\n"  "$c_ye" "$c_0" "$*"; }
err()  { printf "%s[error]%s %s\n" "$c_rd" "$c_0" "$*" >&2; }
hr()   { printf '%s\n' "------------------------------------------------------------"; }

AUTO="${AUTO:-0}"
EXAMPLE_DATA="005057676b565c6278819697bbc3cecad3e6edf301080a122e303a381c235f4a52595c4e51494f585c5f667d848b96"

ask() { # ask VAR "prompt" "default"
  local var="$1" prompt="$2" def="$3" ans envval="${!1-}"
  if [ -n "$envval" ]; then printf '%s[?]%s %s = %s (env)\n' "$c_mg" "$c_0" "$prompt" "$envval"; return; fi
  if [ "$AUTO" = "1" ] || [ ! -t 0 ]; then printf -v "$var" '%s' "$def"
    printf '%s[?]%s %s = %s (default)\n' "$c_mg" "$c_0" "$prompt" "$def"; return; fi
  read -r -p "$(printf '%s[?]%s %s [%s]: ' "$c_mg" "$c_0" "$prompt" "$def")" ans || ans=""
  printf -v "$var" '%s' "${ans:-$def}"
}
SUDO() { if [ "$(id -u)" = 0 ]; then "$@"; elif command -v sudo >/dev/null 2>&1; then sudo "$@";
         else err "need root/sudo to run: $*"; return 1; fi; }

# ---------------------------------------------------------------------------
log "mii2attr launcher  (repo: $ROOT)"; hr

# 1) hardware -----------------------------------------------------------------
HAS_GPU=0; NGPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1; NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
HAS_DOCKER=0; DC="docker compose"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  HAS_DOCKER=1
  docker compose version >/dev/null 2>&1 || DC="docker-compose"
fi
NCORES=$( (nproc 2>/dev/null || echo 4) )
if [ "$HAS_GPU" = 1 ]; then log "GPU: yes (${NGPU}x)  docker: $([ $HAS_DOCKER = 1 ] && echo yes || echo no)  cores: $NCORES"
else log "GPU: none (CPU mode)  docker: $([ $HAS_DOCKER = 1 ] && echo yes || echo no)  cores: $NCORES"; fi

# 2) python + deps ------------------------------------------------------------
USE_VENV="${USE_VENV:-1}"
PYBIN="python3"
if [ "$USE_VENV" = "1" ]; then
  log "using virtualenv .venv"
  [ -d .venv ] || "$PYBIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYBIN="python"
fi
if "$PYBIN" -c "import torch" >/dev/null 2>&1; then
  log "torch already installed ($("$PYBIN" -c 'import torch;print(torch.__version__)'))"
else
  "$PYBIN" -m pip install --upgrade pip >/dev/null 2>&1 || true
  if [ "$HAS_GPU" = 1 ]; then
    log "installing CUDA PyTorch"; "$PYBIN" -m pip install torch torchvision
  else
    log "installing CPU PyTorch"; "$PYBIN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  fi
fi
"$PYBIN" -c "import timm,PIL,requests,yaml" >/dev/null 2>&1 || {
  log "installing timm/pillow/requests/pyyaml"; "$PYBIN" -m pip install timm pillow requests pyyaml; }
log "python env ready"; hr

# 3) renderer -----------------------------------------------------------------
DEF_RENDERER="ffl_public"; [ "$HAS_DOCKER" = 1 ] && DEF_RENDERER="ffl_docker"
ask RENDERER "Renderer (ffl_docker/ffl_native/ffl_public/studio/mock)" "$DEF_RENDERER"

ensure_resource() {
  if [ -f assets/FFLResHigh.dat ]; then log "resource present (assets/FFLResHigh.dat)"; return; fi
  log "downloading FFL resource (Miitomo AFLResHigh) from archive.org ..."
  mkdir -p assets
  curl -fSL --connect-timeout 30 -o assets/_aflres.zip \
    "https://web.archive.org/web/20180502054513/http://download-cdn.miitomo.com/native/20180125111639/android/v2/asset_model_character_mii_AFLResHigh_2_3_dat.zip"
  "$PYBIN" - <<'PY'
import zipfile, glob, os, shutil
z = "assets/_aflres.zip"
zf = zipfile.ZipFile(z); zf.extractall("assets/_aflres")
dat = glob.glob("assets/_aflres/**/*.dat", recursive=True)[0]
shutil.copy(dat, "assets/FFLResHigh.dat")
print("resource ->", os.path.getsize("assets/FFLResHigh.dat"), "bytes")
PY
}
ensure_ffl_repo() {
  if [ ! -d FFL-Testing ]; then
    log "cloning FFL-Testing (renderer-server-prototype, recursive) ..."
    git clone --recursive -b renderer-server-prototype https://github.com/ariankordi/FFL-Testing.git
  else
    git -C FFL-Testing submodule update --init --recursive >/dev/null 2>&1 || true
  fi
}
wait_for_render() { # wait_for_render URL
  local url="$1" i=0 tries=90
  log "waiting for renderer at $url ..."
  while [ $i -lt $tries ]; do
    if curl -fsS -o /dev/null --max-time 5 \
       "${url}/miis/image.png?data=${EXAMPLE_DATA}&width=64&type=face&expression=normal&bgColor=FFFFFFFF" 2>/dev/null; then
      log "renderer is up"; return 0; fi
    i=$((i+1)); sleep 2
  done
  return 1
}

RENDER_KIND="ffl"; RENDER_URL=""
case "$RENDERER" in
  ffl_docker)
    [ "$HAS_DOCKER" = 1 ] || { err "docker not available; choose ffl_public or install docker"; exit 1; }
    ensure_ffl_repo; ensure_resource
    cp -f assets/FFLResHigh.dat FFL-Testing/FFLResHigh.dat
    log "building + starting renderer containers ($DC up -d --build) ..."
    ( cd FFL-Testing && $DC up -d --build )
    RENDER_KIND="ffl"; RENDER_URL="http://localhost:5000"
    wait_for_render "$RENDER_URL" || { err "renderer didn't come up; see: cd FFL-Testing && $DC logs"; exit 1; }
    ;;
  ffl_native)
    ensure_ffl_repo; ensure_resource
    cp -f assets/FFLResHigh.dat FFL-Testing/FFLResHigh.dat
    warn "native build is experimental; Docker (ffl_docker) is more reliable."
    if command -v apt-get >/dev/null 2>&1; then
      SUDO apt-get update && SUDO apt-get install -y xvfb git g++ cmake pkg-config libglfw3-dev zlib1g-dev libgl1-mesa-dev libosmesa6-dev
    elif command -v dnf >/dev/null 2>&1; then
      SUDO dnf install -y git gcc-c++ cmake pkgconfig glfw-devel zlib-devel mesa-libGL-devel mesa-libOSMesa-devel
    else warn "unknown package manager; install glfw3+zlib+mesa-GL+g+++cmake yourself"; fi
    ( cd FFL-Testing && cmake -S . -B build -DRIO_NO_CLIP_CONTROL=ON -DRIO_USE_HEADLESS_GLFW=ON \
        -DCMAKE_CXX_FLAGS="-DNDEBUG -O3" && cmake --build build -j"$NCORES" )
    log "starting renderer + web server in background (logs in /tmp/ffl_*.log)"
    ( cd FFL-Testing && nohup xvfb-run -a ./build/ffl_testing_2 --server >/tmp/ffl_render.log 2>&1 & )
    if [ -d FFL-Testing/server-impl ] && command -v go >/dev/null 2>&1; then
      ( cd FFL-Testing/server-impl && nohup go run . >/tmp/ffl_web.log 2>&1 & )
    else warn "Go web server not started (need Go in server-impl). See FFL-Testing README."; fi
    RENDER_KIND="ffl"; RENDER_URL="http://localhost:5000"
    wait_for_render "$RENDER_URL" || { err "native renderer not reachable; check /tmp/ffl_*.log. Try ffl_docker or ffl_public."; exit 1; }
    ;;
  ffl_public)
    RENDER_KIND="ffl"; RENDER_URL="https://mii-unsecure.ariankordi.net"
    warn "using PUBLIC FFL instance (shared/rate-limited; fine for small datasets)."
    wait_for_render "$RENDER_URL" || warn "public renderer slow/unreachable; will retry during generation."
    ;;
  studio)
    RENDER_KIND="studio"; RENDER_URL="https://studio.mii.nintendo.com"
    warn "using Nintendo Studio API (fallback)."
    ;;
  mock)
    RENDER_KIND="mock"; RENDER_URL=""; warn "using OFFLINE mock renderer (not photorealistic)." ;;
  *) err "unknown RENDERER=$RENDERER"; exit 1 ;;
esac
log "renderer: kind=$RENDER_KIND url=${RENDER_URL:-<default>}"; hr

# 4) sampling + sizes + config -----------------------------------------------
ask SAMPLER "Sampler (plausible/ffl/uniform)" "plausible"
ask JITTER  "Jitter for plausible (none/light/medium/heavy)" "medium"
CL="${CHANNELS_LAST:-true}"   # NHWC: ~40% faster conv on CPU (oneDNN) and on GPU
if [ "$HAS_GPU" = 1 ]; then
  # GPU pipeline: big backbone, full res, AMP
  DEF_N=50000; DEF_EPOCHS=20; DEF_BATCH=256; DEF_BACKBONE="vit_base_patch16_siglip_256"; DEF_PRE="true"
  DEF_WIDTH=256; DEF_WORKERS=$(( NCORES < 16 ? NCORES : 16 )); DEF_AMP="true"; DEF_LR="0.0008"; DEF_WARM=1000; DEF_LOG=50
  log "pipeline: GPU (backbone=$DEF_BACKBONE @${DEF_WIDTH}px, AMP)"
else
  # CPU pipeline: lightweight mobile backbone + low res (much faster on CPU).
  # CPU_PROFILE: fast (mnv3-small@128) | balanced (mnv3-large@160) | quality (resnet50@192)
  CPU_PROFILE="${CPU_PROFILE:-balanced}"
  case "$CPU_PROFILE" in
    fast)    DEF_BACKBONE="mobilenetv3_small_100"; DEF_WIDTH=128; DEF_BATCH=96; DEF_EPOCHS=25 ;;
    quality) DEF_BACKBONE="resnet50";              DEF_WIDTH=192; DEF_BATCH=32; DEF_EPOCHS=12 ;;
    *)       DEF_BACKBONE="mobilenetv3_large_100"; DEF_WIDTH=160; DEF_BATCH=64; DEF_EPOCHS=18 ;;
  esac
  DEF_N=30000; DEF_PRE="true"; DEF_WORKERS=$(( NCORES < 6 ? NCORES : 6 )); DEF_AMP="false"
  DEF_LR="0.0015"; DEF_WARM=50; DEF_LOG=20
  log "pipeline: CPU (profile=$CPU_PROFILE, backbone=$DEF_BACKBONE @${DEF_WIDTH}px, channels_last=$CL)"
fi
ask N_SAMPLES "Dataset size (#images)" "$DEF_N"
ask EPOCHS    "Training epochs" "$DEF_EPOCHS"
ask BACKBONE  "Backbone (cpu: mobilenetv3_large_100/mobilenetv3_small_100; gpu: convnext_tiny/resnet50)" "$DEF_BACKBONE"
ask PRETRAINED "Pretrained backbone (true/false; needs internet for weights)" "$DEF_PRE"
ask BATCH     "Batch size" "$DEF_BATCH"
ask WORKERS   "Dataloader workers" "$DEF_WORKERS"
ask WIDTH     "Render/image size (px)" "$DEF_WIDTH"

mkdir -p configs
cat > configs/auto.yaml <<EOF
# AUTO-GENERATED by start.sh ($(date -u +%FT%TZ))
data: data/auto
out: runs/auto
model:
  backbone: $BACKBONE
  pretrained: $PRETRAINED
  head_hidden: 0
  dropout: 0.1
train:
  image_size: $WIDTH
  batch_size: $BATCH
  epochs: $EPOCHS
  lr: $DEF_LR
  weight_decay: 0.05
  warmup_steps: $DEF_WARM
  num_workers: $WORKERS
  cpu_threads: $NCORES
  amp: $DEF_AMP
  channels_last: $CL
  grad_clip: 5.0
  label_smoothing: 0.05
  log_interval_steps: $DEF_LOG
  seed: 0
EOF
log "wrote configs/auto.yaml (backbone=$BACKBONE pretrained=$PRETRAINED batch=$BATCH epochs=$EPOCHS)"; hr

# 5) dataset ------------------------------------------------------------------
SKIP_DATA="${SKIP_DATA:-0}"
if [ -f data/auto/manifest.jsonl ] && [ "$SKIP_DATA" != 1 ]; then
  have=$(wc -l < data/auto/manifest.jsonl | tr -d ' ')
  ask REGEN "Found data/auto with $have samples. Regenerate? (y/n)" "n"
  [ "$REGEN" = "y" ] || SKIP_DATA=1
fi
if [ "$SKIP_DATA" = 1 ]; then
  log "reusing existing data/auto"
else
  url_arg=(); [ -n "$RENDER_URL" ] && url_arg=(--renderer-url "$RENDER_URL")
  log "generating $N_SAMPLES images (renderer=$RENDER_KIND sampler=$SAMPLER jitter=$JITTER) ..."
  "$PYBIN" -m m2a.dataset_gen --n "$N_SAMPLES" --out data/auto \
    --renderer "$RENDER_KIND" ${url_arg[@]+"${url_arg[@]}"} --sampler "$SAMPLER" --jitter "$JITTER" \
    --width "$WIDTH" --workers "$WORKERS"
fi
hr

# 6) train --------------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" = 1 ]; then log "SKIP_TRAIN=1, done."; exit 0; fi
USE_DDP="${USE_DDP:-0}"
if [ "$HAS_GPU" = 1 ] && [ "${NGPU:-1}" -gt 1 ]; then
  ask USE_DDP "Use all $NGPU GPUs (DistributedDataParallel)? (1/0)" "1"
fi
log "starting training (loss + per-epoch val accuracy will stream below)"; hr
if [ "$USE_DDP" = 1 ] && [ "${NGPU:-1}" -gt 1 ]; then
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$NGPU" -m m2a.train --config configs/auto.yaml
  else
    "$PYBIN" -m torch.distributed.run --nproc_per_node="$NGPU" -m m2a.train --config configs/auto.yaml
  fi
else
  "$PYBIN" -m m2a.train --config configs/auto.yaml
fi
hr
log "done. checkpoints: runs/auto/{last,best}.pt   log: runs/auto/log.csv"
log "evaluate: $PYBIN -m m2a.eval --ckpt runs/auto/best.pt --data data/auto --split val"
log "infer:    $PYBIN -m m2a.infer --ckpt runs/auto/best.pt --image IMG.png --renderer $RENDER_KIND ${RENDER_URL:+--renderer-url $RENDER_URL} --out recon.png"
