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

# Linux equivalent of stop_services.bat, used by the Blender addon's
# Start/Stop toggle (NV_Trellis_Addon.py) to clean up orphaned services from
# a previous session. The addon terminates the process it spawned itself via
# TrellisManager().stop_services() right after calling this script - this
# script's job is only to catch anything left running outside that tracking
# (e.g. a previous Blender session that crashed).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="trellis"
if [ -f config.py ]; then
    RAW_VALUE=$(grep -m1 "^CONDA_ENV_NAME" config.py | cut -d'=' -f2 | tr -d '"'"'"' []')
    [ -n "$RAW_VALUE" ] && ENV_NAME="$RAW_VALUE"
fi

USE_NATIVE_LLM="True"
if [ -f config.py ]; then
    RAW_VALUE=$(grep -m1 "^USE_NATIVE_LLM" config.py | cut -d'=' -f2 | cut -d'#' -f1 | tr -d ' []')
    [ -n "$RAW_VALUE" ] && USE_NATIVE_LLM="$RAW_VALUE"
fi

USE_NATIVE_TRELLIS="True"
if [ -f config.py ]; then
    RAW_VALUE=$(grep -m1 "^USE_NATIVE_TRELLIS" config.py | cut -d'=' -f2 | cut -d'#' -f1 | tr -d ' []')
    [ -n "$RAW_VALUE" ] && USE_NATIVE_TRELLIS="$RAW_VALUE"
fi

echo "Stopping services..."

if [ "$USE_NATIVE_LLM" = "False" ]; then
    echo "Stopping LLM container..."
    conda run -n "$ENV_NAME" --no-capture-output python -c "from nim_llm.manager import stop_container; stop_container()" \
        || echo "[WARNING] Failed to stop LLM container gracefully"
else
    echo "Skipping LLM container (using native LLM)"
fi

if [ "$USE_NATIVE_TRELLIS" = "False" ]; then
    echo "Stopping Trellis container..."
    conda run -n "$ENV_NAME" --no-capture-output python -c "from nim_trellis.manager import stop_container; stop_container()" \
        || echo "[WARNING] Failed to stop Trellis container gracefully"
else
    echo "Skipping Trellis container (using native TRELLIS)"
fi

# Kill any remaining app.py processes (equivalent of the .bat's blunt
# `taskkill /f /im python.exe`, but scoped to this app instead of every
# Python process on the machine).
echo "Stopping app.py processes..."
if pkill -f "[a]pp\.py"; then
    echo "app.py processes stopped"
else
    echo "No app.py processes found to stop"
fi

echo
echo "Services stopped!"
echo
