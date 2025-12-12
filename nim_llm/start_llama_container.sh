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

# Get the NGC API key from the environment variable
#echo "Using NGC API Key: $NGC_API_KEY"

# Ensure NGC_API_KEY is set before running podman
if [ -z "$NGC_API_KEY" ]; then
  echo "NGC API Key is missing!"
  exit 1
fi

# Require container name via environment
if [ -z "$CONTAINER_NAME" ]; then
  echo "Container name (CONTAINER_NAME) is missing!"
  exit 1
fi

# Login to nvcr.io
echo "$NGC_API_KEY" | podman login nvcr.io -u '$oauthtoken' --password-stdin

# Export env vars
export LOCAL_NIM_CACHE=~/.cache/nim

# Setup cache dir
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"

# Run container
podman run --name "$CONTAINER_NAME" -it --rm \
    --device nvidia.com/gpu=all \
    --shm-size=8GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -e NIM_RELAX_MEM_CONSTRAINTS=1 \
    -u $(id -u) \
    -p 19002:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:1.10
