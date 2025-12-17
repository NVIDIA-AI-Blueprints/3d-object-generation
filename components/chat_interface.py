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


"""Chat interface component for LLM agent interaction."""

import gradio as gr
import config

# Only import GPU manager for native models
if config.USE_NATIVE_LLM or config.USE_NATIVE_TRELLIS:
    from services.gpu_memory_manager import get_gpu_memory_manager


def create_chat_interface():
    """Create the chat interface for scene planning."""
    
    with gr.Column(elem_classes=["scene-section"]) as chat_section:
        gr.Markdown("### What scene do you want to create?")
        
        with gr.Row():
            with gr.Column(scale=8):
                scene_input = gr.Textbox(
                    placeholder="Describe your scene and any details you want to include...",
                    label="",
                    lines=1,
                    container=False,
                    elem_classes=["scene-input"]
                )
            with gr.Column(scale=1, min_width=50):
                send_btn = gr.Button("▶", elem_classes=["send-button"], size="sm")
        
        # Tip component for non-scene inputs
        tip_component = gr.HTML(
            value="",
            visible=False,
            elem_classes=["tip-component"]
        )
    
    return {
        "section": chat_section,
        "input": scene_input,
        "send_btn": send_btn,
        "tip": tip_component
    }


def handle_scene_description(scene_description, agent_service, gallery_data, image_generation_service=None):
    """Handle scene description and generate objects with 2D prompts, then populate gallery with generated images."""
    if not scene_description.strip():
        tip_html = """
        <div class="tip-message">
            <span class="tip-icon">💡</span>
            <span class="tip-text">Please enter a scene description.</span>
        </div>
        """
        return "Please enter a scene description.", gallery_data, tip_html, True
    
    try:
        # Prepare GPU for LLM inference (moves other models to CPU)
        if config.USE_NATIVE_LLM:
            gpu_manager = get_gpu_memory_manager()
            gpu_manager.prepare_for_llm()
        
        # First, classify the input
        classification, tip_message = agent_service.classify_input(scene_description)
        
        # If it's not a scene, show tip and don't proceed with object generation
        if classification != "SCENE":
            tip_html = f"""
            <div class="tip-message">
                <span class="tip-icon">💡</span>
                <span class="tip-text">{tip_message}</span>
            </div>
            """
            return "", gallery_data, tip_html, True
        
        # If it is a scene, proceed with object generation
        success, prompts, message = agent_service.generate_objects_and_prompts(scene_description)
        
        if success and prompts:
            # Print to console
            print(f"\n=== Scene Description: '{scene_description}' ===")
            print(f"Generated {len(prompts)} objects with 2D prompts:")
            print("=" * 60)
            
            # Update gallery data with new objects
            new_gallery_data = []
            for i, (obj_name, prompt) in enumerate(prompts.items(), 1):
                print(f"{i:2d}. {obj_name}")
                print(f"    2D Prompt: {prompt}")
                print()
                
                # Start with placeholder image
                gallery_item = {
                    "title": obj_name,
                    "path": None,
                    "description": prompt,
                    "image_generating": True,
                }
                new_gallery_data.append(gallery_item)
            
            print(f"Total objects: {len(prompts)}")
            print("=" * 60)
            
            # Automatically generate images if image generation service is available
            if image_generation_service:
                print("Generating images for all objects...")
                
                # Prepare GPU for SANA (moves LLM to CPU)
                if config.USE_NATIVE_LLM or config.USE_NATIVE_TRELLIS:
                    gpu_manager = get_gpu_memory_manager()
                    gpu_manager.prepare_for_sana()
                
                try:
                    success, message, generated_images = image_generation_service.generate_images_for_objects(new_gallery_data)
                    
                    if success and generated_images:
                        # Update gallery data with generated image paths
                        for obj in new_gallery_data:
                            object_name = obj["title"]
                            if object_name in generated_images:
                                obj["path"] = generated_images[object_name]
                                print(f"Generated image for {object_name}: {generated_images[object_name]}")
                            else:
                                print(f"No image generated for {object_name}")
                    else:
                        print(f"Image generation failed: {message}")
                    
                    # Move SANA to CPU after image generation to free GPU memory
                    # This prevents system slowdown when Gradio displays images
                    print("Moving SANA to CPU after image generation...")
                    image_generation_service.move_sana_pipeline_to_cpu()
                    
                    # Ensure GPU operations complete before Gradio displays images
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        # Note: Removed synchronize() - it blocks system-wide
                        torch.cuda.empty_cache()
                        gc.collect()
                    print("GPU memory cleared, ready for image display")
                        
                except Exception as e:
                    print(f"Error during image generation: {str(e)}")
            
            # Handle 2D prompt content filtered cases by setting dummy image paths
            prompt_content_filtered_count = 0
            for obj in new_gallery_data:
                if obj.get("prompt_content_filtered"):
                    obj["path"] = "static/images/content_filtered.svg"
                    prompt_content_filtered_count += 1
                    print(f"2D prompt content filtered for {obj['title']} - using dummy image")
            
            if prompt_content_filtered_count > 0:
                print(f"{prompt_content_filtered_count} objects had 2D prompts content filtered")
            
            # Create LLM-style response with count, subject reiteration, and suggested actions
            response = f"Review the {len(prompts)} objects and delete any that are not needed. "
            response += "Next step: Generate 3D assets for selected objects."
            
            # Hide tip for successful scene processing
            return response, new_gallery_data, "", False
        else:
            print(f"Error: {message}")
            tip_html = f"""
            <div class="tip-message">
                <span class="tip-icon">⚠️</span>
                <span class="tip-text">Error: {message}</span>
            </div>
            """
            return f"Error: {message}", gallery_data, tip_html, True
        
    except Exception as e:
        print(f"Error generating objects and prompts: {str(e)}")
        tip_html = f"""
        <div class="tip-message">
            <span class="tip-icon">⚠️</span>
            <span class="tip-text">Error generating objects and prompts: {str(e)}</span>
        </div>
        """
        return f"Error generating objects and prompts: {str(e)}", gallery_data, tip_html, True 