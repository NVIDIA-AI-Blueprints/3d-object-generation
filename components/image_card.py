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


import gradio as gr
import random
import os
from services.image_generation_service import ImageGenerationService
# Model3DService is passed as parameter to handlers, not imported directly
# This allows app.py to control which implementation is used (NIM vs Native TRELLIS)
import datetime
import config
from utils import clear_image_generation_failure_flags, should_disable_buttons_during_3d_generation

# Only import GPU manager for native models
if config.USE_NATIVE_LLM or config.USE_NATIVE_TRELLIS:
    from services.gpu_memory_manager import get_gpu_memory_manager

def invalidate_3d_model(gallery_data, card_idx, object_name, context="image change"):
    """Invalidate any existing 3D model for a card when the image has changed."""
    updated_data = gallery_data.copy()
    
    # Invalidate any existing 3D model since the image has changed
    if "glb_path" in updated_data[card_idx]:
        print(f"Invalidating existing 3D model for '{object_name}' due to {context}")
        del updated_data[card_idx]["glb_path"]
    
    if "3d_generated" in updated_data[card_idx]:
        del updated_data[card_idx]["3d_generated"]
    
    if "3d_timestamp" in updated_data[card_idx]:
        del updated_data[card_idx]["3d_timestamp"]
    
    if "content_filtered" in updated_data[card_idx]:
        print(f"Clearing 3D content filtered state for '{object_name}' due to {context}")
        del updated_data[card_idx]["content_filtered"]
    
    if "content_filtered_timestamp" in updated_data[card_idx]:
        del updated_data[card_idx]["content_filtered_timestamp"]
    
    # Reset 3D generation state to allow new generation
    updated_data[card_idx]["3d_generating"] = False
    
    # Clear batch processing flag if it was set
    if "batch_processing" in updated_data[card_idx]:
        del updated_data[card_idx]["batch_processing"]
    
    print(f"3D model invalidated - '→ 3D' button re-enabled")
    return updated_data

def create_convert_all_3d_handler(model_3d_service):
    """Create a handler that converts all unconverted images to 3D models."""
    
    def disable_all_buttons(gallery_data):
        """First stage: disable all buttons by marking all items as batch processing."""
        if not gallery_data:
            return gallery_data
        
        updated_data = gallery_data.copy()
        
        # Mark all items as being processed in batch mode to disable all buttons
        for idx, obj in enumerate(updated_data):
            updated_data[idx]["batch_processing"] = True
            
            # Also mark for global 3D generation if VRAM threshold is met
            if should_disable_buttons_during_3d_generation():
                updated_data[idx]["3d_generation_global"] = True
        
        print(f"Disabled all buttons for {len(gallery_data)} items during batch 3D conversion")
        return updated_data
    
    def perform_batch_3d_conversion(gallery_data):
        """Second stage: perform the actual 3D conversion."""
        try:
            if not gallery_data:
                print("No gallery data to process")
                return gallery_data
            
            updated_data = gallery_data.copy()
            converted_count = 0
            total_unconverted = 0
            
            print(f"Starting batch 3D conversion for {len(gallery_data)} items...")
            
            # First pass: identify unconverted items and mark them as generating
            for idx, obj in enumerate(updated_data):
                if not obj.get("glb_path") and not obj.get("3d_generating", False) and not obj.get("content_filtered", False):
                    updated_data[idx]["3d_generating"] = True
                    total_unconverted += 1
                    print(f" Queued '{obj['title']}' for 3D conversion")
            
            if total_unconverted == 0:
                print("All items already have 3D models or are being generated")
                # Clear batch_processing flag for all items
                for idx in range(len(updated_data)):
                    updated_data[idx]["batch_processing"] = False
                    # Also clear global 3D generation flag
                    if "3d_generation_global" in updated_data[idx]:
                        del updated_data[idx]["3d_generation_global"]
                return updated_data
            
            print(f"Converting {total_unconverted} items to 3D...")
            
            # Prepare GPU for TRELLIS (moves LLM and SANA to CPU)
            if config.USE_NATIVE_TRELLIS:
                gpu_manager = get_gpu_memory_manager()
                gpu_manager.prepare_for_trellis()
            
            # Second pass: generate 3D models for each unconverted item
            for idx, obj in enumerate(updated_data):
                if obj.get("3d_generating", False):
                    object_name = obj["title"]
                    image_path = obj["path"]
                    
                    print(f"  Converting '{object_name}' to 3D...")
                    
                    # Set output directory for generated 3D models
                    output_dir = config.MODELS_DIR
                    
                    # Generate 3D model using Model3DService
                    success, message, glb_path = model_3d_service.generate_3d_model(
                        image_path=image_path,
                        output_dir=output_dir
                    )
                    
                    if success and glb_path:
                        # Update the gallery data with the 3D model path
                        updated_data[idx]["glb_path"] = glb_path
                        updated_data[idx]["3d_generated"] = True
                        updated_data[idx]["3d_timestamp"] = datetime.datetime.now().isoformat()
                        updated_data[idx]["3d_generating"] = False  # Mark as complete
                        converted_count += 1
                        print(f"  Successfully converted '{object_name}' to 3D: {glb_path}")
                    elif message == "CONTENT_FILTERED":
                        # Handle content filtered case
                        updated_data[idx]["3d_generating"] = False
                        updated_data[idx]["content_filtered"] = True
                        updated_data[idx]["content_filtered_timestamp"] = datetime.datetime.now().isoformat()
                        print(f"   Content filtered for '{object_name}' - inappropriate content detected")
                    else:
                        # Mark generation as failed
                        updated_data[idx]["3d_generating"] = False
                        print(f"  Failed to convert '{object_name}' to 3D: {message}")
            
            # Final pass: clear batch_processing flag for all items
            for idx in range(len(updated_data)):
                updated_data[idx]["batch_processing"] = False
                # Also clear global 3D generation flag
                if "3d_generation_global" in updated_data[idx]:
                    del updated_data[idx]["3d_generation_global"]
            
            print(f"Batch 3D conversion complete: {converted_count}/{total_unconverted} items converted")
            return updated_data
            
        except Exception as e:
            print(f"Error in batch 3D conversion: {str(e)}")
            # Reset any items that were marked as generating but failed
            updated_data = gallery_data.copy()
            for idx, obj in enumerate(updated_data):
                if obj.get("3d_generating", False):
                    updated_data[idx]["3d_generating"] = False
                # Clear batch_processing flag
                updated_data[idx]["batch_processing"] = False
                # Clear global 3D generation flag
                if "3d_generation_global" in updated_data[idx]:
                    del updated_data[idx]["3d_generation_global"]
            return updated_data
    
    return disable_all_buttons, perform_batch_3d_conversion

def create_image_card(image_path, title, output_widget, modal_image_title, modal_image, modal_visible, settings_modal, overlay):
    """Create a single image card with action buttons and modal trigger."""
    with gr.Column(elem_classes=["card-content", "clickable-card"]):
        # Create a container for the image and title that covers most of the card
        with gr.Column(elem_classes=["card-main-area"]) as main_area:
            # Transparent button that covers the entire main area for clickability
            card_click_btn = gr.Button("", elem_classes=["card-click-btn"], visible=True)
            title_component = gr.Markdown(f"### {title if title else ''}", elem_classes=["card-title"])
            image_component = gr.Image(
                image_path if image_path else None,
                show_label=False,
                interactive=False,
                height=180,
                elem_classes=["card-image"]
            )
        
        with gr.Row(elem_classes=["card-actions"]):
            gr.HTML("<div style='flex-grow: 1;'></div>")
            refresh_btn = gr.Button("🔄", size="md", min_width=20, elem_classes=["action-btn"])
            edit_btn = gr.Button("✏️", size="md", min_width=20, elem_classes=["action-btn"])
            delete_btn = gr.Button("🗑️", size="md", min_width=20, elem_classes=["action-btn"])
            to_3d_btn = gr.Button("→ 3D", size="md", min_width=20, elem_classes=["action-btn"], elem_id="to-3d-btn")
            gr.HTML("<div style='flex-grow: 1;'></div>")

    # Return the image component and buttons for event binding in app.py
    return {
        "title_component": title_component,
        "image_component": image_component,
        "main_area": main_area,
        "card_click_btn": card_click_btn,
        "refresh_btn": refresh_btn,
        "edit_btn": edit_btn,
        "delete_btn": delete_btn,
        "to_3d_btn": to_3d_btn
    }

def create_refresh_handler(image_generation_service):
    """Create a refresh handler that generates a new image with a random seed."""
    def refresh_image(card_idx, gallery_data):
        """Refresh the image for a specific card with a new random seed."""
        try:
            if card_idx >= len(gallery_data):
                print(f"Card index {card_idx} out of range")
                return gallery_data
            
            # Get the current object data
            obj = gallery_data[card_idx]
            object_name = obj["title"]
            prompt = obj["description"]
            
            # Validate that we have the required data
            if not object_name or not prompt:
                print(f"Missing required data for card {card_idx}: title='{object_name}', prompt='{prompt}'")
                return gallery_data
            
            # Generate a new random seed
            new_seed = random.randint(1, 999999)
            
            # Set output directory for generated images
            import config
            output_dir = config.GENERATED_IMAGES_DIR
            
            print(f"Refreshing image for '{object_name}' with seed {new_seed}")
            print(f"   Prompt: {prompt}")
            
            # Generate new image using SANA service
            success, message, new_image_path = image_generation_service.generate_image_from_prompt(
                object_name=object_name,
                prompt=prompt,
                output_dir=output_dir,
                seed=new_seed
            )

            # Move SANA to CPU after image generation to free GPU memory
            # This prevents system slowdown when Gradio displays the image
            image_generation_service.move_sana_pipeline_to_cpu()
            
            # Ensure GPU operations complete before Gradio displays image
            import torch
            import gc
            if torch.cuda.is_available():
                # Note: Removed synchronize() - it blocks system-wide
                torch.cuda.empty_cache()
                gc.collect()

            invalidate_reason = None
            
            if success and new_image_path:
                # Update the gallery data with the new image path
                updated_data = gallery_data.copy()
                updated_data[card_idx]["path"] = new_image_path
                updated_data[card_idx]["seed"] = new_seed
                
                # Clear any previous failure flags since image generation succeeded
                updated_data[card_idx] = clear_image_generation_failure_flags(updated_data[card_idx])
                 
                invalidate_reason = "image update"
                print(f"Successfully refreshed image: {new_image_path}")
                
            elif message == "PROMPT_CONTENT_FILTERED":
                # Handle 2D prompt content filtered case
                updated_data = gallery_data.copy()
                updated_data[card_idx]["path"] = "static/images/content_filtered.svg"
                updated_data[card_idx]["prompt_content_filtered"] = True
                updated_data[card_idx]["prompt_content_filtered_timestamp"] = datetime.datetime.now().isoformat()
                
                invalidate_reason = "2D prompt content filtered"
                print(f"2D prompt content filtered for '{object_name}' - using dummy image")
                
            else:
                updated_data = gallery_data.copy()
                updated_data[card_idx]["image_generation_failed"] = True
                updated_data[card_idx]["image_generation_error"] = message
                invalidate_reason = "image generation failed"
                print(f"Failed to refresh image: {message}")
            
            # Clear the image_generating flag since the operation is complete
            if "image_generating" in updated_data[card_idx]:
                del updated_data[card_idx]["image_generating"]
            
            updated_data = invalidate_3d_model(updated_data, card_idx, object_name, invalidate_reason)
            if "batch_processing" in updated_data[card_idx]:
                del updated_data[card_idx]["batch_processing"]
            return updated_data
                
        except Exception as e:
            print(f"Error refreshing image: {str(e)}")
            # Ensure we clear the image_generating flag even on exception
            updated_data = gallery_data.copy()
            if "image_generating" in updated_data[card_idx]:
                del updated_data[card_idx]["image_generating"]
            return updated_data
    
    return refresh_image 

def create_3d_generation_handler(model_3d_service):
    """Create a 3D generation handler that converts images to 3D models."""
    def generate_3d_model(card_idx, gallery_data):
        """Generate a 3D model for a specific card."""
        try:
            if card_idx >= len(gallery_data):
                print(f"Card index {card_idx} out of range")
                return gallery_data
            
            # Get the current object data
            obj = gallery_data[card_idx]
            object_name = obj["title"]
            image_path = obj["path"]
            
            # Validate that we have the required data
            if not object_name or not image_path:
                print(f"Missing required data for card {card_idx}: title='{object_name}', path='{image_path}'")
                return gallery_data
            
            # Check if 3D model already exists
            if obj.get("glb_path"):
                print(f"3D model already exists for '{object_name}': {obj['glb_path']}")
                return gallery_data
            
            # Check if generation is already in progress (should be true from immediate update)
            if obj.get("3d_generating"):
                print(f"3D generation in progress for '{object_name}' - continuing...")
            else:
                print(f"3D generation not marked as in progress, but continuing...")
            
            # Set output directory for generated 3D models
            output_dir = config.MODELS_DIR
            
            # Prepare GPU for TRELLIS (moves LLM and SANA to CPU)
            if config.USE_NATIVE_TRELLIS:
                gpu_manager = get_gpu_memory_manager()
                gpu_manager.prepare_for_trellis()
              
            # Generate 3D model using Model3DService
            success, message, glb_path = model_3d_service.generate_3d_model(
                image_path=image_path,
                output_dir=output_dir
            )
            
            # Update the gallery data with the result
            updated_data = gallery_data.copy()
            
            if success and glb_path:
                # Update the gallery data with the 3D model path
                updated_data[card_idx]["glb_path"] = glb_path
                updated_data[card_idx]["3d_generated"] = True
                updated_data[card_idx]["3d_timestamp"] = datetime.datetime.now().isoformat()
                updated_data[card_idx]["3d_generating"] = False  # Mark as complete                
                print(f"Successfully generated 3D model: {glb_path}")
                return updated_data
            elif message == "CONTENT_FILTERED":
                # Handle 3D content filtered case (from model_3d_service)
                updated_data[card_idx]["3d_generating"] = False
                updated_data[card_idx]["content_filtered"] = True
                updated_data[card_idx]["content_filtered_timestamp"] = datetime.datetime.now().isoformat()
                print(f"3D content filtered for '{object_name}' - inappropriate content detected")
                return updated_data
            else:
                # Mark generation as failed
                updated_data[card_idx]["3d_generating"] = False
                print(f"Failed to generate 3D model: {message}")
                return updated_data
                
        except Exception as e:
            print(f"Error generating 3D model: {str(e)}")
            # Mark generation as failed
            updated_data = gallery_data.copy()
            updated_data[card_idx]["3d_generating"] = False
            return updated_data
    
    return generate_3d_model 