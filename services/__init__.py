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

"""Business logic services for Chat-to-3D application."""

import config

# Always available
from .image_generation_service import ImageGenerationService

# Conditional import: AgentService
# Use native PyTorch LLM or NIM-based LLM based on config
if config.USE_NATIVE_LLM:
    from .native_agent_service import NativeAgentService as AgentService
else:
    from .agent_service import AgentService

# Conditional import: Model3DService
# Use native TRELLIS or NIM-based Trellis based on config
if config.USE_NATIVE_TRELLIS:
    from .trellis_service import Model3DService
else:
    from .model_3d_service import Model3DService

__all__ = [
    'AgentService',
    'ImageGenerationService',
    'Model3DService',
] 