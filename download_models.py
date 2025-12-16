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

"""
Download all required models for the 3D Object Generation application.

This script downloads:
- Sana Sprint model (image generation)
- NSFW Prompt Detector (guardrail)
- Native LLM model (if USE_NATIVE_LLM = True)
- Native TRELLIS model (if USE_NATIVE_TRELLIS = True)
"""

import torch
import gc
import sys
from pathlib import Path

from diffusers import SanaSprintPipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config settings directly
import config


def download_sana_model():
    """Download the Sana Sprint image generation model."""
    logger.info("Downloading Sana Sprint model...")
    try:
        sana_model = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16
        )
        del sana_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("✓ Sana Sprint model downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error downloading Sana Sprint model: {e}")
        return False


def download_guardrail_model():
    """Download the NSFW Prompt Detector guardrail model."""
    logger.info("Downloading NSFW Prompt Detector model...")
    try:
        guardrail_pipe = pipeline("text-classification", model="ezb/NSFW-Prompt-Detector")
        del guardrail_pipe
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("✓ NSFW Prompt Detector model downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error downloading NSFW Prompt Detector model: {e}")
        return False


def download_native_llm_model():
    """Download the native LLM model for text generation."""
    if not config.USE_NATIVE_LLM:
        logger.info("Skipping native LLM download (USE_NATIVE_LLM = False)")
        return True
    
    logger.info(f"Downloading native LLM model: {config.NATIVE_LLM_MODEL}...")
    try:
        # Download tokenizer
        logger.info("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.NATIVE_LLM_MODEL)
        del tokenizer
        
        # Download model weights (don't load to GPU, just cache)
        logger.info("  Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            config.NATIVE_LLM_MODEL,
            device_map="cpu",  # Download to CPU only
            torch_dtype=torch.float16,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"✓ Native LLM model downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error downloading native LLM model: {e}")
        logger.error("  Make sure you have accepted the model license on HuggingFace")
        logger.error("  and set HF_TOKEN environment variable if required.")
        return False


def download_native_trellis_model():
    """Download the native TRELLIS 3D generation model."""
    if not config.USE_NATIVE_TRELLIS:
        logger.info("Skipping native TRELLIS download (USE_NATIVE_TRELLIS = False)")
        return True
    
    # Add trellis submodule to path
    trellis_path = Path(__file__).parent / "trellis"
    if str(trellis_path) not in sys.path:
        sys.path.insert(0, str(trellis_path))
    
    logger.info(f"Downloading native TRELLIS model: {config.NATIVE_TRELLIS_MODEL}...")
    try:
        # Import TRELLIS pipeline
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        # Download by loading the pipeline (will cache to HuggingFace cache)
        logger.info("  Downloading TRELLIS pipeline (this may take a while)...")
        pipeline_trellis = TrellisImageTo3DPipeline.from_pretrained(config.NATIVE_TRELLIS_MODEL)
        del pipeline_trellis
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("✓ Native TRELLIS model downloaded successfully!")
        return True
    except ImportError as e:
        logger.warning(f"Could not import TRELLIS pipeline: {e}")
        logger.warning("  TRELLIS dependencies may not be installed yet.")
        logger.warning("  The model will be downloaded when first used.")
        return True  # Not a fatal error
    except Exception as e:
        logger.error(f"✗ Error downloading TRELLIS model: {e}")
        return False


def download_models():
    """Download all required models."""
    logger.info("=" * 60)
    logger.info("Starting model downloads...")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  USE_NATIVE_LLM: {config.USE_NATIVE_LLM}")
    logger.info(f"  USE_NATIVE_TRELLIS: {config.USE_NATIVE_TRELLIS}")
    if config.USE_NATIVE_LLM:
        logger.info(f"  NATIVE_LLM_MODEL: {config.NATIVE_LLM_MODEL}")
    if config.USE_NATIVE_TRELLIS:
        logger.info(f"  NATIVE_TRELLIS_MODEL: {config.NATIVE_TRELLIS_MODEL}")
    logger.info("")
    
    all_success = True
    
    # Core models (always needed)
    if not download_sana_model():
        all_success = False
    
    if not download_guardrail_model():
        all_success = False
    
    # Native LLM model (if enabled)
    if not download_native_llm_model():
        all_success = False
    
    # Native TRELLIS model (if enabled)
    if not download_native_trellis_model():
        all_success = False
    
    logger.info("")
    logger.info("=" * 60)
    if all_success:
        logger.info("✓ All models downloaded successfully!")
    else:
        logger.warning("⚠ Some models failed to download. Check the logs above.")
    logger.info("=" * 60)
    
    return all_success


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1) 