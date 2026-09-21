#!/bin/bash

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Linux equivalent of install.bat: creates a conda environment, installs
# torch/core/TRELLIS dependencies, patches in TRELLIS via setup_trellis.sh,
# and downloads the required models. Mirrors install.bat's steps 1:1 except
# for the pieces that are inherently Windows-only (Blender addon folder copy,
# CHAT_TO_3D_PATH env var, NIM services under WSL) - see README for details.
#
# Usage: bash install_linux.sh
#
# All paths below default to living under the repo directory rather than
# $HOME, so a machine with a small home/conda partition (common when conda
# was installed on the OS disk but the repo lives on a bigger data disk)
# doesn't run out of space. Override any of them if you want different
# locations (e.g. pointing HF_HOME at a separately mounted large volume):
#   ENV_PREFIX, CONDA_PKGS_DIRS, PIP_CACHE_DIR, HF_HOME, CUDA_HOME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_PREFIX="${ENV_PREFIX:-$SCRIPT_DIR/.conda_env}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRIPT_DIR/.conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRIPT_DIR/.pip_cache}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
# huggingface_hub's "xet" fast-download backend hit repeated DNS/timeout
# errors on one test network and was measured ~5.6x slower than the plain
# HTTP downloader for the same file there (20m54s vs 3m43s, both eventually
# succeeded - not a hang, just much slower under that network's retries).
# Disabled by default since the plain downloader was strictly better in that
# test; set to 0 to opt back into xet if your network doesn't have this issue.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "Starting installation process..."
echo "  Conda env prefix: $ENV_PREFIX"
echo "  HF_HOME:          $HF_HOME"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Conda is not installed or not in PATH. Please install Conda first."
    exit 1
fi

if [ ! -f requirements-torch.txt ]; then
    echo "[ERROR] requirements-torch.txt not found!"
    exit 1
fi
if [ ! -f requirements.txt ]; then
    echo "[ERROR] requirements.txt not found!"
    exit 1
fi
if [ ! -f requirements-trellis.txt ]; then
    echo "[ERROR] requirements-trellis.txt not found!"
    exit 1
fi

if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
        echo "  Auto-detected CUDA_HOME: $CUDA_HOME"
    else
        echo "[WARN] CUDA_HOME not set and nvcc not found on PATH."
        echo "       The CUDA extensions built from source later in this script will fail."
        echo "       Set CUDA_HOME to your CUDA toolkit install and re-run."
    fi
fi

# Accept Anaconda Terms of Service (no-op / harmless if channels aren't used)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if [ -d "$ENV_PREFIX" ]; then
    echo "Conda environment already exists at $ENV_PREFIX"
else
    echo "Creating conda environment at $ENV_PREFIX..."
    conda create --prefix "$ENV_PREFIX" python=3.11.9 -y
fi

run() {
    conda run --prefix "$ENV_PREFIX" --no-capture-output "$@"
}

echo "Updating pip and installing build tools..."
run python -m pip install --upgrade pip wheel
run python -m pip install setuptools==75.8.2

echo "Installing torch requirements..."
run pip install -r requirements-torch.txt

# The CUDA extensions built from source below fail hard if the toolkit's
# major.minor CUDA version doesn't match what torch itself was compiled
# with (PyTorch's cpp_extension enforces this strictly - e.g. 13.2 vs 12.8
# errors even though the driver supports both). Cloud GPU images (vast.ai,
# etc.) commonly ship a newer system CUDA toolkit than whatever a pinned
# torch build used, so check now and self-heal by installing a matching
# toolkit into the conda env, rather than failing partway through the
# extension builds below.
TORCH_CUDA_VERSION="$(run python -c 'import torch; print(torch.version.cuda)' 2>/dev/null | tail -1)"
NVCC_CUDA_VERSION=""
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    NVCC_CUDA_VERSION="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || true)"
fi
if [ -n "$TORCH_CUDA_VERSION" ] && [ -n "$NVCC_CUDA_VERSION" ] && [ "$TORCH_CUDA_VERSION" != "$NVCC_CUDA_VERSION" ]; then
    echo "[WARN] System CUDA toolkit ($NVCC_CUDA_VERSION) doesn't match the CUDA version torch was built with ($TORCH_CUDA_VERSION)."
    echo "       Installing a matching CUDA $TORCH_CUDA_VERSION toolkit into the conda env instead of using the system one..."
    if conda install -y --prefix "$ENV_PREFIX" -c "nvidia/label/cuda-${TORCH_CUDA_VERSION}.0" cuda-toolkit; then
        export CUDA_HOME="$ENV_PREFIX"
        export PATH="$CUDA_HOME/bin:$PATH"
        echo "  Now using CUDA_HOME=$CUDA_HOME"
    else
        echo "[WARN] Failed to install a matching CUDA toolkit automatically."
        echo "       The CUDA extension builds below will likely fail until CUDA_HOME points at a $TORCH_CUDA_VERSION.x toolkit."
    fi
fi

echo "Installing core requirements..."
run pip install -r requirements.txt

echo "Installing native TRELLIS requirements..."
run pip install -r requirements-trellis.txt

# CUDA extensions must be installed with --no-build-isolation so their build
# scripts can see the already-installed torch and detect the GPU's CUDA
# version/architecture. Pinned to the same commits install_dependencies.py
# uses on Windows, for the same reproducibility reasons - except gsplat,
# which uses the Linux JIT-build path instead of the bundled Windows wheel.
echo "Installing TRELLIS CUDA extensions (building from source, may take several minutes)..."

# vox2seq's source lives in a subdirectory of a HuggingFace Space. Cloned
# manually (rather than via `pip install git+...`) because HF's git backend
# doesn't support the partial clone (--filter=blob:none) pip uses by default
# for git installs - that fails with "the remote end hung up unexpectedly"
# even though a plain full clone of the same repo works fine.
VOX2SEQ_CLONE_DIR="$(mktemp -d)"
trap 'rm -rf "$VOX2SEQ_CLONE_DIR"' EXIT
git clone --quiet https://huggingface.co/spaces/dkatz2391/TRELLIS_TextTo3D_Try2 "$VOX2SEQ_CLONE_DIR"
git -C "$VOX2SEQ_CLONE_DIR" checkout --quiet f29eac513da77cf6e2c185f52180bb245df11c8a
run pip install --no-build-isolation "$VOX2SEQ_CLONE_DIR/extensions/vox2seq"

run pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"

run pip install --no-build-isolation \
    "git+https://github.com/JeffreyXiang/diffoctreerast.git@b09c20b84ec3aace4729e6e18a613112320eca3a"

if ! run pip install gsplat==1.4.0; then
    echo "No prebuilt gsplat wheel for this platform/CUDA combo, building from source..."
    run pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0"
fi

echo "Setting up TRELLIS source and patches..."
bash setup_trellis.sh

echo "Downloading required models..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[WARN] HF_TOKEN is not set - gated models (e.g. Llama-3.1) will fail to download."
    echo "       export HF_TOKEN=<your token> and re-run download_models.py if needed."
fi
run python download_models.py

# Write a small run script that pins the same HF_HOME/env choices made above.
# Without this, `python app.py` run later in a plain shell falls back to the
# default ~/.cache/huggingface (empty) and silently re-downloads everything
# into the wrong place instead of reusing what was just downloaded here.
cat > run_app.sh <<RUNEOF
#!/bin/bash
# Generated by install_linux.sh - runs app.py with the same env this install used.
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$SCRIPT_DIR"
export HF_HOME="$HF_HOME"
export HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
exec conda run --prefix "$ENV_PREFIX" --no-capture-output python app.py
RUNEOF
chmod +x run_app.sh

echo
echo "Installation completed successfully!"
echo
echo "To run the application:"
echo "  ./run_app.sh"
echo "Then open http://127.0.0.1:7860/?__theme=light"
