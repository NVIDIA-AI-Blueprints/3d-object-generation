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

# Linux equivalent of setup_trellis.bat: fetch TRELLIS and apply patches that
# remove the kaolin/easydict/INRIA-licensed dependencies.
#
# Unlike the .bat, this does a plain full clone rather than a sparse checkout:
# the sparse checkout exists upstream to avoid redistributing the INRIA-licensed
# files even transiently, which only matters when redistributing this repo's
# git history. It has no functional effect on the running app, so a full clone
# is simpler and more robust here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRELLIS_REPO_URL="https://github.com/microsoft/TRELLIS.git"

if [ ! -d "trellis/.git" ]; then
    if [ -d "trellis" ]; then
        echo "Removing incomplete trellis directory..."
        rm -rf trellis
    fi
    echo "Cloning TRELLIS..."
    git clone "$TRELLIS_REPO_URL" trellis
else
    echo "TRELLIS already cloned, applying patches..."
fi

# Copy MIT/Apache-2.0 licensed patch files (removes easydict and kaolin dependencies)
cp trellis_patch/renderers/gaussian_render.py trellis/trellis/renderers/
cp trellis_patch/renderers/mesh_renderer.py trellis/trellis/renderers/
cp trellis_patch/representations/gaussian/gaussian_model.py trellis/trellis/representations/gaussian/
cp trellis_patch/representations/mesh/cube2mesh.py trellis/trellis/representations/mesh/
cp trellis_patch/pipelines/samplers/flow_euler.py trellis/trellis/pipelines/samplers/
cp trellis_patch/utils/render_utils.py trellis/trellis/utils/
cp trellis_patch/utils/postprocessing_utils.py trellis/trellis/utils/

# Create flexicubes directory and copy patched files (replaces FlexiCubes submodule entirely)
mkdir -p trellis/trellis/representations/mesh/flexicubes
cp trellis_patch/representations/mesh/flexicubes/flexicubes.py trellis/trellis/representations/mesh/flexicubes/
cp trellis_patch/representations/mesh/flexicubes/tables.py trellis/trellis/representations/mesh/flexicubes/

echo "TRELLIS is set up and patched."
