#!/usr/bin/env bash
# Bring up a LOCAL FFL Mii render server (ariankordi/FFL-Testing) on your box.
# The data generator (m2a.dataset_gen --renderer ffl --renderer-url http://localhost:5000)
# then renders against this local instance instead of any public endpoint.
#
# Requires the Nintendo FFL resource file `FFLResHigh.dat` (this repo already
# staged it at assets/FFLResHigh.dat from the Miitomo AFLResHigh asset).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFL_DIR="${HERE}/FFL-Testing"
RES="${HERE}/assets/FFLResHigh.dat"

if [ ! -d "${FFL_DIR}" ]; then
  echo "Cloning FFL-Testing (renderer-server-prototype)..."
  git clone --recursive -b renderer-server-prototype \
    https://github.com/ariankordi/FFL-Testing.git "${FFL_DIR}"
else
  echo "FFL-Testing present; ensuring submodules..."
  git -C "${FFL_DIR}" submodule update --init --recursive
fi

if [ ! -f "${RES}" ]; then
  echo "ERROR: ${RES} not found. Provide the FFL resource file first." >&2
  exit 1
fi
cp -f "${RES}" "${FFL_DIR}/FFLResHigh.dat"
echo "Placed FFLResHigh.dat into ${FFL_DIR}"

# ---- Option A: Docker (simplest) ------------------------------------------
if command -v docker >/dev/null 2>&1; then
  echo "Starting renderer via docker compose (port 5000)..."
  ( cd "${FFL_DIR}" && docker compose up -d )
  echo "Renderer should be at http://localhost:5000"
  echo "Test: curl 'http://localhost:5000/miis/image.png?data=005057676b565c6278819697bbc3cecad3e6edf301080a122e303a381c235f4a52595c4e51494f585c5f667d848b96&width=256' -o test.png"
  exit 0
fi

# ---- Option B: native build (headless GLFW + software/HW GL) ---------------
cat <<'EOF'
Docker not found. Native build instructions:

  # deps (Debian/Ubuntu):
  sudo apt install -y git g++ cmake pkg-config libglfw3-dev zlib1g-dev libgl1-mesa-dev
  # (Fedora/RHEL: sudo dnf install glfw-devel zlib-devel mesa-libGL-devel gcc-c++ cmake)

  cd FFL-Testing
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -march=native" -DRIO_USE_HEADLESS_GLFW=ON
  cmake --build build -j"$(nproc)"

  # run the renderer (headless) + the bundled web server:
  ./build/ffl_testing_2 --server &
  # then a web server from server-impl/ (Go: `cd server-impl && go run .`)
  # listening on :5000 and proxying /miis/image.png to the renderer.

On a headless server with a GPU, EGL/HW acceleration is fastest. On CPU-only
hosts use Mesa software GL (LIBGL_ALWAYS_SOFTWARE=1) — slower but works.
EOF
