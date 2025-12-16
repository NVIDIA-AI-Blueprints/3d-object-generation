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

"""Native TRELLIS 3D model generation service - drop-in replacement for Model3DService."""

import os
import sys
import time
import logging
import datetime
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from PIL import Image

# Add trellis submodule to Python path
TRELLIS_PATH = Path(__file__).parent.parent / "trellis"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

# Set TRELLIS environment variables before importing
os.environ['SPCONV_ALGO'] = 'native'  # Faster for single runs

logger = logging.getLogger(__name__)

# Import config for settings
try:
    import config
    VERBOSE = getattr(config, 'VERBOSE', True)
    MAX_GPU_MEMORY_FRACTION = getattr(config, 'TRELLIS_MAX_GPU_MEMORY_FRACTION', 0.95)
    CLEAR_CACHE_BETWEEN_IMAGES = getattr(config, 'TRELLIS_CLEAR_CACHE_BETWEEN_IMAGES', True)
except ImportError:
    VERBOSE = True
    MAX_GPU_MEMORY_FRACTION = 0.95
    CLEAR_CACHE_BETWEEN_IMAGES = True


def setup_gpu_memory_limit(device_id: int = 0):
    """Set GPU memory limit to prevent using shared memory (system RAM).
    
    This forces PyTorch to only use dedicated VRAM. If memory is exceeded,
    an OOM error is raised instead of spilling to slower shared memory.
    
    Args:
        device_id: CUDA device index
    """
    if not torch.cuda.is_available():
        return
    
    if MAX_GPU_MEMORY_FRACTION is None:
        logger.warning("GPU memory limit disabled - shared memory may be used!")
        return
    
    try:
        # Set memory fraction limit (prevents shared memory usage)
        torch.cuda.set_per_process_memory_fraction(MAX_GPU_MEMORY_FRACTION, device_id)
        
        # Get total memory for logging
        total_mem = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        limited_mem = total_mem * MAX_GPU_MEMORY_FRACTION
        
        if VERBOSE:
            logger.info(f"GPU memory limit set: {limited_mem:.2f} GB ({MAX_GPU_MEMORY_FRACTION*100:.0f}% of {total_mem:.2f} GB)")
            logger.info("Shared memory (system RAM) usage is DISABLED")
    except Exception as e:
        logger.warning(f"Failed to set GPU memory limit: {e}")


def clear_gpu_cache():
    """Clear GPU cache to free unused memory."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if VERBOSE:
            logger.debug("GPU cache cleared")


# =============================================================================
# VRAM and Timing Utilities
# =============================================================================
def get_gpu_memory_info(device_id: int = 0) -> Optional[dict]:
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
        free = total - reserved
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(free, 2)
        }
    return None


def log_gpu_memory(prefix: str = "", device_id: int = 0) -> Optional[dict]:
    """Log GPU memory usage."""
    info = get_gpu_memory_info(device_id)
    if info and VERBOSE:
        logger.info(f"{prefix}VRAM - Allocated: {info['allocated_gb']:.2f} GB, "
                   f"Reserved: {info['reserved_gb']:.2f} GB, "
                   f"Free: {info['free_gb']:.2f} GB, "
                   f"Total: {info['total_gb']:.2f} GB")
    return info


class Model3DService:
    """Native TRELLIS service for generating 3D models from images.
    
    This class has the same interface as the REST API-based Model3DService,
    allowing it to be used as a drop-in replacement.
    """
    
    # Default model (can be overridden by config.NATIVE_TRELLIS_MODEL)
    DEFAULT_MODEL = "microsoft/TRELLIS-image-large"
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the 3D model service.
        
        Args:
            base_url: Ignored - kept for API compatibility with REST-based Model3DService
        """
        # base_url is ignored but kept for interface compatibility
        self.base_url = base_url
        
        # Get model name from config if available
        try:
            import config
            self.model_name = getattr(config, 'NATIVE_TRELLIS_MODEL', self.DEFAULT_MODEL)
        except ImportError:
            self.model_name = self.DEFAULT_MODEL
        self.pipeline = None
        self.timeout = 300  # 5 minutes timeout for 3D generation
        self._is_loaded = False
        self._load_time = None
        self._vram_after_load = None
        
        # Lazy import flag
        self._imports_done = False
        
    def _lazy_import(self):
        """Lazily import TRELLIS modules to avoid import errors at startup."""
        if self._imports_done:
            return
            
        global TrellisImageTo3DPipeline, postprocessing_utils
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils
        
        self._imports_done = True
        
    def _ensure_pipeline_loaded(self) -> bool:
        """Ensure the TRELLIS pipeline is loaded.
        
        Note: Models are pre-loaded at startup via GPUMemoryManager.preload_all_models().
        This method is kept for backwards compatibility and edge cases.
        
        Returns:
            True if pipeline is loaded, False otherwise
        """
        # Fast path: already loaded (normal case after pre-loading)
        if self._is_loaded and self.pipeline is not None:
            return True
        
        # Load pipeline if not already loaded
        try:
            self._lazy_import()
            
            if VERBOSE:
                logger.info("=" * 60)
                logger.info("TRELLIS Pipeline - Loading")
                logger.info("=" * 60)
                log_gpu_memory("Before pipeline load - ")
            
            # Set GPU memory limit BEFORE loading the model
            # This prevents PyTorch from using shared memory (system RAM)
            setup_gpu_memory_limit(device_id=0)
            
            load_start = time.time()
            
            # Load pipeline from HuggingFace
            logger.info(f"Loading TRELLIS model: {self.model_name}")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_name)
            self.pipeline.cuda()
            
            self._load_time = time.time() - load_start
            self._is_loaded = True
            
            if VERBOSE:
                self._vram_after_load = log_gpu_memory("After pipeline load - ")
                logger.info(f">>> Pipeline loaded in {self._load_time:.2f} seconds")
                if self._vram_after_load:
                    logger.info(f">>> VRAM used: {self._vram_after_load['allocated_gb']:.2f} GB")
            
            return True
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory while loading TRELLIS pipeline: {e}")
            logger.error("Try reducing TRELLIS_MAX_GPU_MEMORY_FRACTION in config.py or freeing GPU memory")
            self._is_loaded = False
            return False
        except Exception as e:
            logger.error(f"Failed to load TRELLIS pipeline: {e}")
            self._is_loaded = False
            return False
    
    def generate_3d_model(self, image_path: str, output_dir: str = "assets/models") -> Tuple[bool, str, Optional[str]]:
        """Generate a 3D model from an image file.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save the generated GLB file
            
        Returns:
            Tuple of (success, message, glb_file_path)
        """
        try:
            # Ensure pipeline is loaded
            if not self._ensure_pipeline_loaded():
                return False, "Failed to load TRELLIS pipeline", None
            
            # Validate image exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False, f"Image file not found: {image_path}", None
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load image
            image = Image.open(image_path)
            logger.info(f"Loaded image: {image_path}")
            
            if VERBOSE:
                logger.info("=" * 60)
                logger.info("TRELLIS Pipeline - Running Inference")
                logger.info("=" * 60)
                vram_before = log_gpu_memory("Before inference - ")
            
            run_start = time.time()
            
            # Run the pipeline
            outputs = self.pipeline.run(image, seed=1)
            
            run_time = time.time() - run_start
            
            if VERBOSE:
                vram_after = log_gpu_memory("After inference - ")
                logger.info(f">>> Inference completed in {run_time:.2f} seconds")
                if vram_after:
                    logger.info(f">>> Peak VRAM: {vram_after['reserved_gb']:.2f} GB")
            
            # Generate filename based on original image name
            image_name = Path(image_path).stem
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            glb_filename = f"{image_name}_{timestamp}.glb"
            glb_path = os.path.join(output_dir, glb_filename)
            
            # Export GLB
            export_start = time.time()
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(glb_path)
            
            if VERBOSE:
                logger.info(f">>> GLB exported in {time.time() - export_start:.2f} seconds")
            
            logger.info(f"Saved 3D model to: {glb_path}")
            
            # Clear intermediate tensors to free memory
            del outputs
            clear_gpu_cache()
            
            return True, "Successfully generated 3D model", glb_path
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory during 3D generation: {e}")
            clear_gpu_cache()
            return False, "GPU out of memory - try processing fewer images or reduce model settings", None
        except Exception as e:
            logger.error(f"Error in generate_3d_model: {e}")
            clear_gpu_cache()
            return False, f"Error generating 3D model: {str(e)}", None
    
    def check_service_health(self) -> bool:
        """Check if the 3D generation service is available.
        
        Returns:
            True if service is healthy (pipeline loaded or can be loaded), False otherwise
        """
        # If pipeline is already loaded, it's healthy
        if self._is_loaded and self.pipeline is not None:
            return True
        
        # Try to import TRELLIS modules to check if dependencies are available
        try:
            self._lazy_import()
            return True
        except Exception as e:
            logger.warning(f"TRELLIS service not healthy: {e}")
            return False
    
    def wait_for_service_ready(self, timeout: int = 300, poll_interval: int = 5) -> bool:
        """Wait for the TRELLIS service to be ready.
        
        For native TRELLIS, this just ensures the pipeline can be loaded.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            poll_interval: Time between checks in seconds (default: 5)
            
        Returns:
            True if service is ready, False if timeout reached
        """
        start_time = time.time()
        logger.info(f"Checking TRELLIS service readiness (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            if self.check_service_health():
                elapsed = time.time() - start_time
                logger.info(f"TRELLIS service is ready (checked in {elapsed:.1f}s)")
                return True
            
            elapsed = time.time() - start_time
            logger.info(f"TRELLIS not ready yet, waiting... ({elapsed:.0f}s / {timeout}s)")
            time.sleep(poll_interval)
        
        logger.error(f"TRELLIS service not ready after {timeout}s timeout")
        return False
    
    def batch_generate_models(self, image_paths: list, output_dir: str = "assets/models") -> Dict[str, Any]:
        """Generate 3D models for multiple images sequentially.
        
        Images are processed one at a time with memory cleanup between each
        to prevent VRAM overflow and shared memory usage.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save generated GLB files
            
        Returns:
            Dictionary with results for each image
        """
        results = {
            "successful": [],
            "failed": [],
            "total": len(image_paths)
        }
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Clear GPU cache before each image (except first) to prevent memory buildup
            if CLEAR_CACHE_BETWEEN_IMAGES and i > 0:
                if VERBOSE:
                    logger.info("Clearing GPU cache before next image...")
                clear_gpu_cache()
                log_gpu_memory("Memory after cleanup - ")
            
            success, message, glb_path = self.generate_3d_model(image_path, output_dir)
            
            if success:
                results["successful"].append({
                    "image_path": image_path,
                    "glb_path": glb_path,
                    "message": message
                })
            else:
                results["failed"].append({
                    "image_path": image_path,
                    "error": message
                })
        
        # Final cleanup after batch processing
        clear_gpu_cache()
        
        logger.info(f"Batch processing complete: {len(results['successful'])}/{results['total']} successful")
        return results
    
    def unload_pipeline(self):
        """Unload the pipeline to free GPU memory."""
        if self.pipeline is not None:
            if VERBOSE:
                log_gpu_memory("Before unload - ")
            
            del self.pipeline
            self.pipeline = None
            self._is_loaded = False
            gc.collect()
            torch.cuda.empty_cache()
            
            if VERBOSE:
                log_gpu_memory("After unload - ")
                logger.info("TRELLIS pipeline unloaded")
    
    def move_to_cpu(self):
        """Move pipeline to CPU to free GPU memory."""
        if self.pipeline is not None:
            self.pipeline.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("TRELLIS pipeline moved to CPU")
    
    def move_to_gpu(self):
        """Move pipeline back to GPU."""
        if self.pipeline is not None:
            self.pipeline.cuda()
            logger.info("TRELLIS pipeline moved to GPU")


# =============================================================================
# Standalone usage example
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create service (same interface as Model3DService)
    model_service = Model3DService()
    
    # Check health
    if model_service.check_service_health():
        print("✅ TRELLIS service is healthy")
    else:
        print("❌ TRELLIS service is not available")
        exit(1)
    
    # Example: Single image generation
    image_path = "assets/example_image/test.png"
    success, message, glb_path = model_service.generate_3d_model(image_path)
    
    if success:
        print(f"✅ Success: {message}")
        print(f"📁 GLB file saved to: {glb_path}")
    else:
        print(f"❌ Failed: {message}")
    
    # Cleanup
    model_service.unload_pipeline()
