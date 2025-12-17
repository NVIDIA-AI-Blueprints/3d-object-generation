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

"""GPU Memory Manager - Coordinates GPU memory usage across multiple models.

This manager ensures only the active model uses GPU memory by moving 
inactive models to CPU. The workflow is:
1. Initialization: Load all models, move TRELLIS+SANA to CPU, LLM stays on GPU
2. After prompt: LLM runs → move LLM to CPU
3. Image generation: Move SANA to GPU → generate → move SANA to CPU  
4. 3D generation: Move TRELLIS to GPU → generate → TRELLIS stays on GPU
"""

import logging
import gc
import time
import torch
import config

logger = logging.getLogger(__name__)

VERBOSE = getattr(config, 'VERBOSE', False)


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        # Note: Removed synchronize() - not needed for memory queries
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
    return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0}


def log_gpu_memory(prefix: str = ""):
    """Log GPU memory usage."""
    if VERBOSE:
        info = get_gpu_memory_info()
        logger.info(f"{prefix}VRAM - Allocated: {info['allocated_gb']:.2f} GB, "
                   f"Reserved: {info['reserved_gb']:.2f} GB, "
                   f"Free: {info['free_gb']:.2f} GB")


class GPUMemoryManager:
    """Manages GPU memory across multiple models (LLM, SANA, TRELLIS).
    
    This is a singleton-like manager that coordinates model placement
    to ensure efficient GPU memory usage on a single GPU system.
    """
    
    def __init__(self):
        self.llm_service = None
        self.sana_service = None
        self.trellis_service = None
        
        # Track which model is currently on GPU
        self._current_gpu_model = None  # 'llm', 'sana', 'trellis', or None
        
    def register_llm_service(self, service):
        """Register the LLM agent service."""
        self.llm_service = service
        logger.info("GPUMemoryManager: LLM service registered")
        
    def register_sana_service(self, service):
        """Register the SANA image generation service."""
        self.sana_service = service
        logger.info("GPUMemoryManager: SANA service registered")
        
    def register_trellis_service(self, service):
        """Register the TRELLIS 3D generation service."""
        self.trellis_service = service
        logger.info("GPUMemoryManager: TRELLIS service registered")
    
    def _clear_gpu_cache(self):
        """Clear GPU cache after moving models."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Note: Removed synchronize() - it blocks system-wide
    
    def _warmup_trellis(self):
        """Run a warmup inference on TRELLIS to compile CUDA kernels.
        
        This creates a small dummy image and runs the pipeline,
        ensuring all kernels are compiled for faster subsequent runs.
        """
        from PIL import Image
        import numpy as np
        
        # Create a small dummy image (64x64 is enough for warmup)
        # Using a simple colored image instead of noise for consistency
        dummy_size = 64
        dummy_array = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)
        dummy_array[:, :] = [128, 128, 128]  # Gray image
        dummy_image = Image.fromarray(dummy_array, 'RGB')
        
        # Run pipeline with minimal settings
        if self.trellis_service and self.trellis_service.pipeline is not None:
            logger.info("    Running warmup with 64x64 dummy image...")
            
            # Run the pipeline (this compiles CUDA kernels)
            _ = self.trellis_service.pipeline.run(
                dummy_image,
                seed=42
            )
            
            # Clear the warmup outputs
            self._clear_gpu_cache()
            logger.info("    CUDA kernels compiled and cached")
    
    def prepare_for_llm(self):
        """Prepare GPU for LLM inference.
        
        Move TRELLIS and SANA to CPU, ensure LLM is on GPU.
        """
        if self._current_gpu_model == 'llm':
            return  # Already ready
            
        start_time = time.time()
        logger.info("GPUMemoryManager: Preparing GPU for LLM inference...")
        
        if VERBOSE:
            log_gpu_memory("Before LLM prep - ")
        
        # Move TRELLIS to CPU if loaded (but don't move if it's the current model - user wants it to stay)
        # Actually, for LLM we need the space, so move TRELLIS to CPU
        if self.trellis_service and hasattr(self.trellis_service, 'pipeline'):
            if self.trellis_service.pipeline is not None:
                logger.info("  Moving TRELLIS to CPU...")
                self.trellis_service.move_to_cpu()
        
        # Move SANA to CPU if loaded  
        if self.sana_service and self.sana_service.is_loaded:
            logger.info("  Moving SANA to CPU...")
            self.sana_service.move_sana_pipeline_to_cpu()
        
        # Ensure LLM is on GPU
        if self.llm_service and hasattr(self.llm_service, 'agent') and self.llm_service.agent:
            if hasattr(self.llm_service.agent, 'device') and self.llm_service.agent.device != config.NATIVE_LLM_DEVICE:
                logger.info("  Moving LLM to GPU...")
                self.llm_service.move_agent_to_gpu()
        
        self._clear_gpu_cache()
        self._current_gpu_model = 'llm'
        
        if VERBOSE:
            log_gpu_memory("After LLM prep - ")
            logger.info(f"GPUMemoryManager: LLM prep complete in {time.time() - start_time:.2f}s")
    
    def prepare_for_sana(self):
        """Prepare GPU for SANA image generation.
        
        Move LLM to CPU, move SANA to GPU.
        TRELLIS should already be on CPU.
        """
        if self._current_gpu_model == 'sana':
            return  # Already ready
            
        start_time = time.time()
        logger.info("GPUMemoryManager: Preparing GPU for SANA image generation...")
        
        if VERBOSE:
            log_gpu_memory("Before SANA prep - ")
        
        # Move LLM to CPU
        if self.llm_service and hasattr(self.llm_service, 'agent'):
            if self.llm_service.agent and hasattr(self.llm_service.agent, 'device'):
                if self.llm_service.agent.device != "cpu":
                    logger.info("  Moving LLM to CPU...")
                    self.llm_service.move_agent_to_cpu()
        
        # Move TRELLIS to CPU if loaded (shouldn't be on GPU at this point)
        if self.trellis_service and hasattr(self.trellis_service, 'pipeline'):
            if self.trellis_service.pipeline is not None:
                logger.info("  Ensuring TRELLIS is on CPU...")
                self.trellis_service.move_to_cpu()
        
        self._clear_gpu_cache()
        
        # Move SANA to GPU
        if self.sana_service:
            logger.info("  Moving SANA to GPU...")
            self.sana_service.move_sana_pipeline_to_gpu()
        
        self._current_gpu_model = 'sana'
        
        if VERBOSE:
            log_gpu_memory("After SANA prep - ")
            logger.info(f"GPUMemoryManager: SANA prep complete in {time.time() - start_time:.2f}s")
    
    def prepare_for_trellis(self):
        """Prepare GPU for TRELLIS 3D generation.
        
        Move LLM and SANA to CPU, move TRELLIS to GPU.
        TRELLIS will stay on GPU after this (for subsequent 3D generations).
        """
        if self._current_gpu_model == 'trellis':
            return  # Already ready
            
        start_time = time.time()
        logger.info("GPUMemoryManager: Preparing GPU for TRELLIS 3D generation...")
        
        if VERBOSE:
            log_gpu_memory("Before TRELLIS prep - ")
        
        # Move LLM to CPU
        if self.llm_service and hasattr(self.llm_service, 'agent'):
            if self.llm_service.agent and hasattr(self.llm_service.agent, 'device'):
                if self.llm_service.agent.device != "cpu":
                    logger.info("  Moving LLM to CPU...")
                    self.llm_service.move_agent_to_cpu()
        
        # Move SANA to CPU
        if self.sana_service and self.sana_service.is_loaded:
            logger.info("  Moving SANA to CPU...")
            self.sana_service.move_sana_pipeline_to_cpu()
        
        self._clear_gpu_cache()
        
        # Move TRELLIS to GPU - it will STAY on GPU
        if self.trellis_service:
            logger.info("  Moving TRELLIS to GPU (will stay on GPU)...")
            if hasattr(self.trellis_service, 'pipeline') and self.trellis_service.pipeline is not None:
                self.trellis_service.move_to_gpu()
        
        self._current_gpu_model = 'trellis'
        
        if VERBOSE:
            log_gpu_memory("After TRELLIS prep - ")
            logger.info(f"GPUMemoryManager: TRELLIS prep complete in {time.time() - start_time:.2f}s")
    
    def preload_all_models(self):
        """Pre-load all models at startup.
        
        1. Load TRELLIS → warmup → move to CPU
        2. Load SANA → move to CPU  
        3. Load LLM → stays on GPU (ready for chat)
        
        Returns:
            dict: Status of each model load
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("PRE-LOADING ALL MODELS AT STARTUP")
        logger.info("=" * 60)
        
        status = {
            "llm_loaded": False,
            "sana_loaded": False,
            "trellis_loaded": False,
            "total_time": 0
        }
        
        if VERBOSE:
            log_gpu_memory("Before pre-loading - ")
        
        # Step 1: Load TRELLIS (largest model, ~8GB)
        if self.trellis_service and config.USE_NATIVE_TRELLIS:
            try:
                logger.info("\n[1/3] Loading TRELLIS model...")
                trellis_start = time.time()
                
                # Load the pipeline (this will load to GPU)
                if hasattr(self.trellis_service, '_ensure_pipeline_loaded'):
                    self.trellis_service._ensure_pipeline_loaded()
                    status["trellis_loaded"] = self.trellis_service._is_loaded
                
                # Warmup: Run a dummy inference to compile CUDA kernels
                if status["trellis_loaded"]:
                    logger.info("  Running TRELLIS warmup inference...")
                    warmup_start = time.time()
                    try:
                        self._warmup_trellis()
                        logger.info(f"  Warmup completed in {time.time() - warmup_start:.2f}s")
                    except Exception as e:
                        logger.warning(f"  Warmup failed (non-critical): {e}")
                
                # Move to CPU to free GPU for other models
                if status["trellis_loaded"]:
                    logger.info("  Moving TRELLIS to CPU...")
                    self.trellis_service.move_to_cpu()
                    self._clear_gpu_cache()
                
                logger.info(f"  TRELLIS loaded in {time.time() - trellis_start:.2f}s")
                if VERBOSE:
                    log_gpu_memory("  After TRELLIS - ")
            except Exception as e:
                logger.error(f"  Failed to load TRELLIS: {e}")
        else:
            logger.info("[1/3] TRELLIS: Skipped (USE_NATIVE_TRELLIS=False or service not registered)")
        
        # Step 2: Load SANA (image generation, ~5GB)
        if self.sana_service:
            try:
                logger.info("\n[2/3] Loading SANA model...")
                sana_start = time.time()
                
                # Load the model
                self.sana_service.load_sana_model(device="cuda:0")
                status["sana_loaded"] = self.sana_service.is_loaded
                
                # Move to CPU to free GPU for LLM
                if status["sana_loaded"]:
                    logger.info("  Moving SANA to CPU...")
                    self.sana_service.move_sana_pipeline_to_cpu()
                    self._clear_gpu_cache()
                
                logger.info(f"  SANA loaded in {time.time() - sana_start:.2f}s")
                if VERBOSE:
                    log_gpu_memory("  After SANA - ")
            except Exception as e:
                logger.error(f"  Failed to load SANA: {e}")
        else:
            logger.info("[2/3] SANA: Skipped (service not registered)")
        
        # Step 3: Load LLM (stays on GPU for chat)
        if self.llm_service and config.USE_NATIVE_LLM:
            try:
                logger.info("\n[3/3] Loading LLM model...")
                llm_start = time.time()
                
                # Load the agent and model
                if hasattr(self.llm_service, '_ensure_agent_loaded'):
                    # Create agent wrapper first (without loading model)
                    self.llm_service._ensure_agent_loaded(load_model=False)
                    # Now load the model (this is where the actual GPU memory is used)
                    if hasattr(self.llm_service.agent, 'ensure_model_loaded'):
                        self.llm_service.agent.ensure_model_loaded()
                    status["llm_loaded"] = self.llm_service.agent is not None and self.llm_service.agent.is_loaded
                
                logger.info(f"  LLM loaded in {time.time() - llm_start:.2f}s")
                if VERBOSE:
                    log_gpu_memory("  After LLM (on GPU) - ")
            except Exception as e:
                logger.error(f"  Failed to load LLM: {e}")
        else:
            logger.info("[3/3] LLM: Skipped (USE_NATIVE_LLM=False or service not registered)")
        
        self._current_gpu_model = 'llm'  # LLM is on GPU, ready for chat
        
        status["total_time"] = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("PRE-LOADING COMPLETE")
        logger.info(f"  TRELLIS: {'✓ Loaded (on CPU)' if status['trellis_loaded'] else '✗ Not loaded'}")
        logger.info(f"  SANA: {'✓ Loaded (on CPU)' if status['sana_loaded'] else '✗ Not loaded'}")
        logger.info(f"  LLM: {'✓ Loaded (on GPU)' if status['llm_loaded'] else '✗ Not loaded'}")
        logger.info(f"  Total time: {status['total_time']:.2f}s")
        logger.info("=" * 60)
        
        if VERBOSE:
            log_gpu_memory("Final memory state - ")
        
        return status
    
    def get_status(self) -> dict:
        """Get status of all registered services and GPU memory."""
        return {
            "current_gpu_model": self._current_gpu_model,
            "llm_registered": self.llm_service is not None,
            "sana_registered": self.sana_service is not None,
            "trellis_registered": self.trellis_service is not None,
            "gpu_memory": get_gpu_memory_info()
        }


# Global singleton instance
_gpu_manager = None


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get the global GPU memory manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUMemoryManager()
    return _gpu_manager
