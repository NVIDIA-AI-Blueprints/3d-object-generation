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

"""Image generation service using SANA model."""

import os
import logging
import datetime
import torch
import gc
from diffusers import SanaSprintPipeline
import time
import config
from services.guardrail_service import GuardrailService
from utils import clear_image_generation_failure_flags, check_gpu_vram_capacity

# Set environment variables for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

logger = logging.getLogger(__name__)

class ImageGenerationService:
    def __init__(self):
        self.sana_pipeline = None
        self.is_loaded = False
        self.model_path = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
        self.device = None
        self.guardrail_service = GuardrailService()
        
    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent fragmentation."""
        try:
            if torch.cuda.is_available():
                # More aggressive memory clearing
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Could not clear GPU memory: {e}")
    
    def move_sana_pipeline_to_device(self, device="cuda:0"):
        """Move SANA pipeline and guardrail model to specified device (GPU or CPU)."""
        try:
            if self.sana_pipeline is None:
                logger.warning("SANA pipeline is not loaded")
                return False
            
            logger.info(f"Moving SANA pipeline to {device}")

            if self.device == device:
                logger.info(f"SANA pipeline is already on {device}")
            else:
                # Move pipeline to specified device
                self.sana_pipeline.to(device)
                self.device = device
                logger.info(f"Successfully moved SANA pipeline to {device}")
            
            # Also move guardrail model (used alongside SANA for content safety)
            if self.guardrail_service.is_loaded:
                self.guardrail_service.move_to_device(device)
            
            # Clear GPU memory after moving to CPU
            if device == "cpu":
                self._clear_gpu_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error moving SANA pipeline to {device}: {e}")
            return False
    
    def move_sana_pipeline_to_gpu(self):
        """Move SANA pipeline and guardrail to GPU."""
        return self.move_sana_pipeline_to_device("cuda:0")
    
    def move_sana_pipeline_to_cpu(self):
        """Move SANA pipeline and guardrail to CPU."""
        return self.move_sana_pipeline_to_device("cpu")
    
    def unload_sana_model(self):
        """Completely unload SANA pipeline to free all memory (GPU and CPU).
        
        Use this in RAM_RESTRICTED mode to save system memory.
        The model will need to be reloaded before next use.
        """
        if self.sana_pipeline is not None:
            logger.info("Unloading SANA pipeline completely...")
            del self.sana_pipeline
            self.sana_pipeline = None
            self.is_loaded = False
            
            # Also unload guardrail if present
            if self.guardrail_service:
                self.guardrail_service.unload_model()
            
            # Clear memory
            self._clear_gpu_memory()
            logger.info("SANA pipeline unloaded")
            return True
        return False
    
    def load_sana_model(self, device="cuda:0", force_reload=False):
        """Load the SANA model for image generation with optimizations."""
        try:
            print(f"Timestamp before load_sana_model: {time.time()}")
            if self.is_loaded and self.sana_pipeline is not None and not force_reload:
                logger.info("SANA model already loaded")
                self.move_sana_pipeline_to_device(device)
                return True
            
            print(f"Timestamp after load_sana_model: {time.time()}")
            # Clear GPU memory before loading
            self._clear_gpu_memory() 
            print(f"Timestamp after clear_gpu_memory: {time.time()}")
            
            logger.info("Loading SANA model...")
            
            initial_time = time.time()
            # Load model with optimizations and directly to GPU
            self.sana_pipeline = SanaSprintPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
            print(f"Timestamp after load_sana_model: {time.time()}")
            print(f"Time taken to load SANA model: {time.time() - initial_time} seconds")
            
            initial_time = time.time()
            # Move to GPU with memory optimization
            self.move_sana_pipeline_to_device("cuda:0")
            
            print(f"Time taken to move SANA model to GPU: {time.time() - initial_time} seconds")
            print(f"Timestamp after move_sana_model_to_gpu: {time.time()}")
        
            self.is_loaded = True
            logger.info("Successfully loaded SANA model")   
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading SANA model: {e}")
            self.is_loaded = False
            self.sana_pipeline = None
            return False
        
    def cleanup_sana_pipeline(self):
        """Clean up the current model"""
        if self.sana_pipeline is not None:
            try:
                # Move to CPU first
                if hasattr(self.sana_pipeline, 'cuda'):
                    self.sana_pipeline.cpu()

                # Clear internal tensors if any
                if hasattr(self.sana_pipeline, '__dict__'):
                    for k, v in list(vars(self.sana_pipeline).items()):
                        if torch.is_tensor(v):
                            setattr(self.sana_pipeline, k, None)
                            del v
                # Delete the model reference
                del self.sana_pipeline
                self.sana_pipeline = None
                self.is_loaded = False
                self.device = None

                self._clear_gpu_memory()
                logger.info("Successfully cleaned up SANA pipeline")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


        

    def if_sana_pipeline_movement_required(self, vram_threshold=config.VRAM_THRESHOLD_SANA):
        """Check if the SANA pipeline needs to be moved to GPU or CPU."""
        return check_gpu_vram_capacity(vram_threshold)
        
    
    def generate_image_from_prompt(self, object_name, prompt, output_dir, seed=42):
        """Generate a single image from a prompt using SANA model."""
        try:
            # First, check content safety using guardrail
            logger.info(f"Checking content safety for prompt: {prompt[:100]}...")
            is_safe, safety_message = self.guardrail_service.check_prompt_safety(prompt)
            
            if not is_safe:
                logger.warning(f"2D prompt flagged as inappropriate for {object_name}: {safety_message}")
                # Return a special flag to indicate 2D prompt content was filtered
                return False, "PROMPT_CONTENT_FILTERED", None
            
            if not self.load_sana_model():
                return False, "Failed to load SANA model", None
            
            # Format object name: lowercase and replace spaces with underscores
            formatted_object_name = object_name.lower().replace(" ", "_")
            
            # Generate image - SCM requires num_inference_steps=2
            with torch.no_grad():  # Reduce memory usage during inference
                image = self.sana_pipeline(
                    prompt=prompt,
                    num_inference_steps=2,  # SCM requirement
                    generator=torch.Generator("cuda").manual_seed(seed)
                ).images[0]
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename using convention: objectname_seed_timestamp
            image_path = os.path.join(output_dir, f"{formatted_object_name}_{seed}_{timestamp}.png")
            
            # Save the image
            image.save(image_path)
            
            logger.info(f"Generated image: {image_path}")
            
            return True, f"Successfully generated image for {object_name}", image_path
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return False, f"Error generating image: {str(e)}", None
    
    def generate_images_for_objects(self, objects_data, output_dir="static/images/generated"):
        """Generate images for all objects in the gallery data."""
        try:
            if not self.load_sana_model():
                return False, "Failed to load SANA model", {}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            generated_images = {}
            content_filtered_objects = []
            
            for obj in objects_data:
                object_name = obj["title"]
                prompt = obj["description"]
                
                logger.info(f"Generating image for: {object_name}")
                success, message, image_path = self.generate_image_from_prompt(
                    object_name, prompt, output_dir
                )
                
                if success and image_path:
                    generated_images[object_name] = image_path
                    # Clear any previous failure flags since image generation succeeded
                    obj = clear_image_generation_failure_flags(obj)
                    logger.info(f"Generated image for {object_name}: {image_path}")
                elif message == "PROMPT_CONTENT_FILTERED":
                    # Mark object as 2D prompt content filtered
                    obj["path"] = "static/images/content_filtered.svg"
                    obj["prompt_content_filtered"] = True
                    obj["prompt_content_filtered_timestamp"] = datetime.datetime.now().isoformat()
                    content_filtered_objects.append(object_name)
                    logger.warning(f"2D prompt content filtered for {object_name}")
                else:
                    # Mark object as having failed image generation
                    obj["image_generation_failed"] = True
                    obj["image_generation_error"] = message
                    logger.error(f"Failed to generate image for {object_name}: {message}")
            
            # Log summary
            if content_filtered_objects:
                logger.warning(f"Content filtered objects: {content_filtered_objects}")
            
            # Ensure all CUDA operations are complete before returning to Gradio
            # Clear GPU cache (removed synchronize - it blocks system-wide)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return True, f"Generated {len(generated_images)} images, {len(content_filtered_objects)} content filtered", generated_images
            
        except Exception as e:
            logger.error(f"Error generating images for objects: {e}")
            # Clear cache on error path (removed synchronize - it blocks system-wide)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, f"Error generating images: {str(e)}", {}
    
    def generate_images_for_objects_with_progress(self, objects_data, output_dir="static/images/generated"):
        """Generate images for all objects with progress updates (generator).
        
        Yields after each image is generated so the UI can update progressively.
        During generation, keeps image_generating=True so buttons stay disabled.
        Only sets image_generating=False on final yield when all images are complete.
        
        Yields:
            tuple: (current_idx, total, object_name, updated_objects_data, is_complete)
        """
        try:
            if not self.load_sana_model():
                yield (0, len(objects_data), "Failed to load model", objects_data, True)
                return
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            total = len(objects_data)
            content_filtered_objects = []
            
            for idx, obj in enumerate(objects_data):
                object_name = obj["title"]
                prompt = obj["description"]
                
                logger.info(f"Generating image {idx+1}/{total}: {object_name}")
                success, message, image_path = self.generate_image_from_prompt(
                    object_name, prompt, output_dir
                )
                
                if success and image_path:
                    objects_data[idx]["path"] = image_path
                    # Keep image_generating=True during progress - buttons stay disabled
                    # Clear any previous failure flags
                    objects_data[idx] = clear_image_generation_failure_flags(objects_data[idx])
                    logger.info(f"Generated image for {object_name}: {image_path}")
                elif message == "PROMPT_CONTENT_FILTERED":
                    objects_data[idx]["path"] = "static/images/content_filtered.svg"
                    objects_data[idx]["prompt_content_filtered"] = True
                    objects_data[idx]["prompt_content_filtered_timestamp"] = datetime.datetime.now().isoformat()
                    content_filtered_objects.append(object_name)
                    logger.warning(f"2D prompt content filtered for {object_name}")
                else:
                    objects_data[idx]["image_generation_failed"] = True
                    objects_data[idx]["image_generation_error"] = message
                    logger.error(f"Failed to generate image for {object_name}: {message}")
                
                # Yield progress after each image (buttons still disabled)
                yield (idx + 1, total, object_name, objects_data.copy(), False)
            
            # Final yield - NOW set image_generating=False for all to enable buttons
            for obj in objects_data:
                obj["image_generating"] = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            yield (total, total, "Complete", objects_data, True)
            
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            # On error, also clear the generating flag
            for obj in objects_data:
                obj["image_generating"] = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield (0, len(objects_data), f"Error: {str(e)}", objects_data, True) 