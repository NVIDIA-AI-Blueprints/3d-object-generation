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

"""Simple configuration for Chat-to-3D application.

This configuration file centralizes file paths and basic variables
without changing core functionality or UI elements.
"""

import os
from pathlib import Path

# =============================================================================
# Conda Environment Settings
# =============================================================================
CONDA_ENV_NAME = "trellis"

# =============================================================================
# Backend Selection Flags
# =============================================================================
# USE_NATIVE_LLM: Controls which LLM backend to use
#   - True:  Use native PyTorch model (no NIM, no griptape required)
#   - False: Use LLM NIM service (requires griptape package)
USE_NATIVE_LLM = True

# USE_NATIVE_TRELLIS: Controls which 3D generation backend to use
#   - True:  Use native PyTorch TRELLIS (no NIM required, runs locally)
#   - False: Use Trellis NIM service (requires running NIM container)
USE_NATIVE_TRELLIS = True


# #############################################################################
#                          NIM CONFIGURATION
# #############################################################################
# Settings used when USE_NATIVE_LLM = False or USE_NATIVE_TRELLIS = False
# Requires: griptape[all] package and running NIM containers
# =============================================================================

# NIM LLM Agent Settings
AGENT_MODEL = "meta/llama-3.1-8b-instruct"
AGENT_BASE_URL = "http://localhost:19002/v1"

# NIM TRELLIS Settings
TRELLIS_BASE_URL = "http://localhost:8000/v1"


# #############################################################################
#                        NATIVE CONFIGURATION
# #############################################################################
# Settings used when USE_NATIVE_LLM = True or USE_NATIVE_TRELLIS = True
# No NIM required, runs models directly on GPU
# =============================================================================

# -----------------------------------------------------------------------------
# Native LLM Settings (used when USE_NATIVE_LLM = True)
# -----------------------------------------------------------------------------
# Model name from HuggingFace
# Qwen3-4B: "Qwen/Qwen3-4B" (use precision: bfloat16) - 4B params
# Llama-3.1-8B: "meta-llama/Llama-3.1-8B-Instruct" (use precision: bfloat16) - 8B params
# For GPTQ INT4: "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" (use precision: int4)
NATIVE_LLM_MODEL = "Qwen/Qwen3-4B"

# Precision options: "float16", "bfloat16", "float32", "int4"
# Use "int4" when loading GPTQ quantized models
NATIVE_LLM_PRECISION = "bfloat16"

NATIVE_LLM_MAX_NEW_TOKENS = 1024  # Maximum tokens to generate
NATIVE_LLM_DEVICE = "cuda:0"  # Device to load model on

# -----------------------------------------------------------------------------
# Native TRELLIS Settings (used when USE_NATIVE_TRELLIS = True)
# -----------------------------------------------------------------------------
# Native TRELLIS model name (from HuggingFace)
NATIVE_TRELLIS_MODEL = "microsoft/TRELLIS-image-large"

# GPU Memory Settings
# Prevent PyTorch from using shared memory (system RAM) - only use dedicated VRAM
# Set to a value < 1.0 to limit GPU memory fraction (e.g., 0.95 = 95% of VRAM)
# Set to 1.0 to use all available VRAM (but still prevent shared memory)
# Set to None to disable memory limit (allows shared memory - NOT recommended)
TRELLIS_MAX_GPU_MEMORY_FRACTION = 0.95

# Clear GPU cache between batch image processing to prevent memory buildup
TRELLIS_CLEAR_CACHE_BETWEEN_IMAGES = True

# #############################################################################
#                        COMMON CONFIGURATION
# #############################################################################
# Settings shared by both NIM and Native backends
# =============================================================================

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------
# Get the base directory of the project
BASE_DIR = Path(__file__).parent

# Use user's home directory for data storage
HOME_DIR = Path.home()
TRELLIS_DIR = HOME_DIR / ".trellis"  # Hidden directory
ASSETS_DIR = TRELLIS_DIR / "assets"
PROMPTS_DIR = TRELLIS_DIR / "prompts"
SCENE_DIR = TRELLIS_DIR / "scene"

# Create directories
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
SCENE_DIR.mkdir(parents=True, exist_ok=True)

# Define file paths
OUTPUT_DIR = ASSETS_DIR
PROMPTS_FILE = PROMPTS_DIR / "prompts.json"

# Static asset paths
STATIC_DIR = BASE_DIR / "static"
CSS_DIR = STATIC_DIR / "css"
JS_DIR = STATIC_DIR / "js"
IMAGES_DIR = STATIC_DIR / "images"
ASSETS_APP_DIR = BASE_DIR / "assets"
GENERATED_IMAGES_DIR = ASSETS_APP_DIR / "images"
MODELS_DIR = ASSETS_APP_DIR / "models"

# Create application directories
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File paths for static assets
CUSTOM_CSS_FILE = CSS_DIR / "custom.css"
CUSTOM_JS_FILE = JS_DIR / "custom.js"
NVIDIA_LOGO_FILE = IMAGES_DIR / "nvidia_logo.png"
GENERATING_PLACEHOLDER_FILE = IMAGES_DIR / "generating.svg"

# -----------------------------------------------------------------------------
# LLM Shared Settings (used by both backends)
# -----------------------------------------------------------------------------
LLM_TEMPERATURE = 0.4  # Controls randomness in LLM responses (0.0 = deterministic, 1.0 = very random)
LLM_RANDOM_SEED_ENABLED = True  # Enable random seed for object generation
TWO_D_PROMPT_LENGTH = 30

# -----------------------------------------------------------------------------
# UI Settings
# -----------------------------------------------------------------------------
NUM_OF_OBJECTS = 20
MAX_CARDS = 20  # Maximum number of cards in gallery
CARDS_PER_ROW = 4  # Number of cards per row in gallery
INITIAL_MESSAGE = "Hello! I'm your helpful scene planning assistant. Please describe the scene you'd like to create."

# -----------------------------------------------------------------------------
# VRAM Thresholds
# -----------------------------------------------------------------------------
VRAM_THRESHOLD_SANA = 16  # if GPU has <= VRAM_THRESHOLD_SANA, SANA Pipeline will be stopped
VRAM_THRESHOLD_DISABLE_IMAGE_GEN_DURING_3D_GENERATION = 16  # Disable image gen during 3D gen if GPU <= this
VRAM_THRESHOLD_LLM = 31  # if GPU has > VRAM_THRESHOLD_LLM, LLM Agent will not be stopped

# -----------------------------------------------------------------------------
# Model Defaults
# -----------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_SPARSE_STEPS = 25
DEFAULT_SLAT_STEPS = 25
DEFAULT_CFG_STRENGTH = 7.5
MAX_PROMPT_LENGTH = 50

# TRELLIS pipeline settings (for NIM)
TRELLIS_TEXT_LARGE_MODEL = "JeffreyXiang/TRELLIS-text-large"
TRELLIS_TEXT_BASE_MODEL = "JeffreyXiang/TRELLIS-text-base"
TRELLIS_IMAGE_LARGE_MODEL = "microsoft/TRELLIS-image-large"

# Model configuration
TRELLIS_MODEL_NAME_MAP = {
    "TRELLIS-text-large": TRELLIS_TEXT_LARGE_MODEL,
    "TRELLIS-text-base": TRELLIS_TEXT_BASE_MODEL
}
DEFAULT_TRELLIS_MODEL = "TRELLIS-text-large"

# Algorithm configuration
SPCONV_ALGO = "spconv2"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
VERBOSE = False  # Enable detailed logging for timing and VRAM usage


# #############################################################################
#                          HELPER FUNCTIONS
# #############################################################################

def get_static_paths():
    """Get static asset paths."""
    return {
        "css": CUSTOM_CSS_FILE,
        "js": CUSTOM_JS_FILE,
        "nvidia_logo": NVIDIA_LOGO_FILE,
        "generated_images": GENERATED_IMAGES_DIR,
        "models": MODELS_DIR
    }

def get_output_paths():
    """Get output paths."""
    return {
        "generated_images": GENERATED_IMAGES_DIR,
        "models": MODELS_DIR,
        "assets": ASSETS_DIR,
        "prompts": PROMPTS_DIR,
        "scene": SCENE_DIR
    }
