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

# Get the base directory of the project
BASE_DIR = Path(__file__).parent

# Use user's home directory
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

NUM_OF_OBJECTS = 20

# Basic configuration settings
MAX_CARDS = 20  # Maximum number of cards in gallery
CARDS_PER_ROW = 4  # Number of cards per row in gallery
VRAM_THRESHOLD_SANA = 16 # if GPU has <= VRAM_THRESHOLD_SANA, SANA Pipeline will be stopped
VRAM_THRESHOLD_DISABLE_IMAGE_GEN_DURING_3D_GENERATION = 16 # if GPU has <= VRAM_THRESHOLD_DISABLE_IMAGE_GEN_DURING_3D_GENERATION, image generation will be disabled during 3D generation
VRAM_THRESHOLD_LLM = 31  # if GPU has > VRAM_THRESHOLD_LLM, LLM Agent will not be stopped
DEFAULT_SEED = 42
DEFAULT_SPARSE_STEPS = 25
DEFAULT_SLAT_STEPS = 25
DEFAULT_CFG_STRENGTH = 7.5
MAX_PROMPT_LENGTH = 50
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# TRELLIS pipeline settings
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

# INITIAL MESSAGE
INITIAL_MESSAGE = "Hello! I'm your helpful scene planning assistant. Please describe the scene you'd like to create."

# =============================================================================
# LLM Backend Selection
# =============================================================================
# USE_NATIVE_LLM: Controls which LLM backend to use
#   - True:  Use native PyTorch model (no NIM, no griptape required)
#   - False: Use LLM NIM service (requires griptape package)
USE_NATIVE_LLM = True

# =============================================================================
# NIM Agent Settings (used when USE_NATIVE_LLM = False)
# Requires: griptape[all] package
# =============================================================================
AGENT_MODEL = "meta/llama-3.1-8b-instruct"
AGENT_BASE_URL = "http://localhost:19002/v1"
TRELLIS_BASE_URL = "http://localhost:8000/v1"
TWO_D_PROMPT_LENGTH = 30

# =============================================================================
# LLM Shared Settings (used by both backends)
# =============================================================================
LLM_TEMPERATURE = 0.4  # Controls randomness in LLM responses (0.0 = deterministic, 1.0 = very random)
LLM_RANDOM_SEED_ENABLED = True  # Enable random seed for object generation

# =============================================================================
# Native LLM Settings (used when USE_NATIVE_LLM = True)
# No NIM required, runs model directly on GPU
# =============================================================================
# Model name from HuggingFace
# For full precision: "meta-llama/Llama-3.1-8B-Instruct" (use precision: bfloat16)
# For GPTQ INT4:      "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" (use precision: int4)
NATIVE_LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

# Precision options: "float16", "bfloat16", "float32", "int4"
# Use "int4" when loading GPTQ quantized models
NATIVE_LLM_PRECISION = "int4"

NATIVE_LLM_MAX_NEW_TOKENS = 1024  # Maximum tokens to generate
NATIVE_LLM_DEVICE = "cuda:0"  # Device to load model on

# =============================================================================
# Verbose Logging
# =============================================================================
VERBOSE = True  # Enable detailed logging for timing and VRAM usage

# Simple helper functions
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