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


"""Image gallery component for displaying generated objects."""

import gradio as gr
from components.image_card import create_image_card
import config

MAX_CARDS = config.MAX_CARDS
CARDS_PER_ROW = config.CARDS_PER_ROW

def create_image_gallery():
    """Create the object gallery interface."""
    
    with gr.Column(elem_classes=["gallery-section"]) as gallery_section:
        gr.Markdown("### Object Gallery", elem_classes=["gallery-header"])
        
        # Initial gallery data
        initial_gallery_data = []
        gallery_data = gr.State(initial_gallery_data)
        
        # Create placeholder for empty gallery state
        with gr.Column(visible=True, elem_classes=["gallery-placeholder"]) as placeholder_container:
            with gr.Column(elem_classes=["placeholder-content"]):
                gr.HTML(
                    value="<div style='border: 2px solid #e0e0e0; border-radius: 8px; background-color: #f8f8f8; padding: 40px; text-align: center;'>"
                          "<div style='display: flex; justify-content: center; align-items: center; margin-bottom: 20px;'>"
                          "<svg width='48' height='48' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>"
                          "<circle cx='12' cy='12' r='10' stroke='#ccc' stroke-width='2' fill='none'/>"
                          "<path d='M12 16v-4' stroke='#ccc' stroke-width='2' stroke-linecap='round'/>"
                          "<path d='M12 8h.01' stroke='#ccc' stroke-width='2' stroke-linecap='round'/>"
                          "</svg>"
                          "</div>"
                          "<div style='color: #666; font-size: 16px; line-height: 1.5;'>"
                          "First describe the scene you want above, and then we'll generate some ideas to populate the scene"
                          "</div>"
                          "</div>",
                    elem_classes=["placeholder-text"]
                )
        
        # Create cards in a responsive grid using Gradio's native layout
        card_components = []
        card_containers = []
        
        for row_start in range(0, MAX_CARDS, CARDS_PER_ROW):
            with gr.Row() as row:
                for col_idx in range(CARDS_PER_ROW):
                    card_idx = row_start + col_idx
                    if card_idx < MAX_CARDS:
                        # Use scale=1 to fill available space, but with min_width to maintain consistency
                        with gr.Column(scale=1, min_width=180, visible=False, elem_classes=["gallery-card"]) as card_container:
                            # Create placeholder modal components for card creation
                            # These will be replaced by actual modal components from main app
                            card = create_image_card("", "", None, None, None, None, None, None)
                            
                            card_components.append(card)
                            card_containers.append(card_container)
        
        # Add "Convert all to 3D" button at the bottom
        with gr.Row(elem_classes=["convert-all-section"]):
            convert_all_btn = gr.Button(
                "Convert all to 3D", 
                #size="lg", 
                #variant="primary",
                elem_classes=["convert-all-btn"],
                visible=False,  # Initially hidden, will be shown when gallery has items
                interactive=False  # Initially disabled, will be enabled when there are unconverted items
            )
        
        # Utility: get all card outputs
        def get_all_card_outputs():
            outputs = []
            for card in card_components:
                outputs.extend([card["title_component"], card["image_component"], card["refresh_btn"], card["edit_btn"], card["delete_btn"], card["to_3d_btn"]])
            outputs.extend(card_containers)
            outputs.append(placeholder_container)  # Add placeholder to outputs
            outputs.append(convert_all_btn)  # Add convert all button to outputs
            return outputs
        
        # Logic functions
        def delete_card_by_index(index, gallery_data):
            print(f"Deleting card at index: {index} title: {gallery_data[index]['title']}")
            updated = [item for i, item in enumerate(gallery_data) if i != index]
            return updated

        def create_delete_function(card_idx):
            def delete_specific_card(gallery_data):
                return delete_card_by_index(card_idx, gallery_data)
            return delete_specific_card

        def shift_card_ui(gallery_data):
            """Update the UI to reflect the current gallery data state."""
            import time
            start_time = time.time()
            updates = []
            for idx in range(MAX_CARDS):
                if idx < len(gallery_data):
                    obj = gallery_data[idx]
                    
                    updates.append(gr.update(value=f"### {obj['title']}"))
                    
                    # Determine image to display (actual image, generating placeholder, or empty)
                    is_image_generating = obj.get("image_generating", False)
                    if obj.get("path"):
                        updates.append(gr.update(value=obj["path"]))
                    elif is_image_generating:
                        updates.append(gr.update(value=str(config.GENERATING_PLACEHOLDER_FILE)))
                    else:
                        updates.append(gr.update(value=None))
                    
                    # Check processing states to disable other buttons
                    is_3d_generating = obj.get("3d_generating", False)
                    is_batch_processing = obj.get("batch_processing", False)
                    is_3d_generation_global = obj.get("3d_generation_global", False)
                    is_image_operations_global = obj.get("image_operations_global", False)
                    # is_image_generating already computed above
                    is_processing = is_3d_generating or is_batch_processing or is_image_generating or is_3d_generation_global or is_image_operations_global
                    
                    # Update refresh button state
                    refresh_interactive = not is_processing
                    refresh_classes = ["action-btn"]
                    if is_processing:
                        refresh_classes.append("disabled-btn")
                    updates.append(gr.update(interactive=refresh_interactive, elem_classes=refresh_classes))
                    
                    # Update edit button state
                    edit_interactive = not is_processing
                    edit_classes = ["action-btn"]
                    if is_processing:
                        edit_classes.append("disabled-btn")
                    updates.append(gr.update(interactive=edit_interactive, elem_classes=edit_classes))
                    
                    # Update delete button state
                    delete_interactive = not is_processing
                    delete_classes = ["action-btn"]
                    if is_processing:
                        delete_classes.append("disabled-btn")
                    updates.append(gr.update(interactive=delete_interactive, elem_classes=delete_classes))
                    
                    # Update 3D button text and state based on status
                    if "glb_path" in obj and obj["glb_path"]:
                        button_text = "✓ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "three-d-completed"]
                    elif obj.get("content_filtered"):
                        button_text = "🚫 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "content-filtered"]
                    elif obj.get("image_generation_failed", False) or obj.get("prompt_content_filtered", False):
                        print(f"-> 3D button disabled for {obj['title']} due to image generation failed or prompt content filtered")
                        button_text = "→ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "disabled-btn"]
                    elif obj.get("3d_generating"):
                        button_text = "⏳ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "three-d-generating"]
                    elif is_batch_processing:
                        button_text = "⏳ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "three-d-generating"]
                    elif is_image_generating:
                        button_text = "→ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "disabled-btn"]
                    elif is_image_operations_global:
                        button_text = "→ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "disabled-btn"]
                    else:
                        button_text = "→ 3D"
                        button_interactive = True
                        button_classes = ["action-btn"]
                    
                    updates.append(gr.update(value=button_text, interactive=button_interactive, elem_classes=button_classes))
                else:
                    updates.append(gr.update(value=""))
                    updates.append(gr.update(value=None))
                    updates.append(gr.update(interactive=True, elem_classes=["action-btn"]))  # refresh
                    updates.append(gr.update(interactive=True, elem_classes=["action-btn"]))  # edit
                    updates.append(gr.update(interactive=True, elem_classes=["action-btn"]))  # delete
                    updates.append(gr.update(value="→ 3D", interactive=True, elem_classes=["action-btn"]))  # 3D
            
            # Show/hide card containers based on data
            for idx in range(MAX_CARDS):
                updates.append(gr.update(visible=(idx < len(gallery_data))))
            
            # Show/hide placeholder based on whether gallery has items
            updates.append(gr.update(visible=(len(gallery_data) == 0)))
            
            # Show/enable convert all button based on items, unconverted, and processing states
            has_items = len(gallery_data) > 0
            
            has_unconverted_items = any(
                idx < len(gallery_data) and 
                not gallery_data[idx].get("glb_path") and 
                not gallery_data[idx].get("3d_generating", False) and
                not gallery_data[idx].get("content_filtered", False) and
                not gallery_data[idx].get("image_generation_failed", False) and
                not gallery_data[idx].get("prompt_content_filtered", False)
                for idx in range(len(gallery_data))
            )
            
            is_batch_processing = any(
                idx < len(gallery_data) and 
                gallery_data[idx].get("batch_processing", False)
                for idx in range(len(gallery_data))
            )

            any_image_generating = any(
                idx < len(gallery_data) and 
                gallery_data[idx].get("image_generating", False)
                for idx in range(len(gallery_data))
            )

            any_3d_generating = any(
                idx < len(gallery_data) and 
                gallery_data[idx].get("3d_generating", False)
                for idx in range(len(gallery_data))
            )
            
            show_convert_all = has_items
            enable_convert_all = has_unconverted_items and not is_batch_processing and not any_image_generating and not any_3d_generating
            
            # Set button text based on processing state
            if is_batch_processing:
                convert_all_text = "Generating - will take up to 45 secs per object"
            else:
                convert_all_text = "Convert all to 3D"
            
            updates.append(gr.update(visible=show_convert_all, interactive=enable_convert_all, value=convert_all_text))
            
            elapsed = time.time() - start_time
            print(f"Gallery UI updated with {len(gallery_data)} visible cards (took {elapsed:.3f}s)")
            return updates
        
        # Note: Delete button events are handled in the main app to enable export section updates
    
    return {
        "section": gallery_section,
        "data": gallery_data,
        "card_components": card_components,
        "card_containers": card_containers,
        "placeholder": placeholder_container,
        "convert_all_btn": convert_all_btn,
        "get_all_card_outputs": get_all_card_outputs,
        "shift_card_ui": shift_card_ui,
    } 