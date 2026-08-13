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


"""Blender export component for exporting 3D objects to Blender."""

import gradio as gr
import os
import base64
from PIL import Image
import io
import zipfile
import tempfile
from pathlib import Path
from config import ASSETS_DIR

def create_blender_export_section():
    """Create the Blender export section interface."""
    
    with gr.Column(elem_classes=["blender-export-section"]) as export_section:
        gr.Markdown("### Export your collection for Blender", elem_classes=["export-header"])
        
        # Instructional link
        with gr.Row():
            gr.HTML(
                value="<a href='https://github.com/NVIDIA-AI-Blueprints/3d-object-generation/tree/main?tab=readme-ov-file#usage---blender-add-on' target='_blank' style='color: #0066cc; text-decoration: none; display: flex; align-items: center; gap: 4px;'>"
                      "Learn how to load them into Blender's library"
                      "<svg width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'>"
                      "<path d='M7 7h10v10'></path>"
                      "<path d='M7 17 17 7'></path>"
                      "</svg>"
                      "</a>",
                elem_classes=["export-link"]
            )
        
        # Main content area with dynamic content
        with gr.Column(elem_classes=["export-content"]) as content_container:
            # Placeholder for empty state (shown when no 3D assets)
            placeholder = gr.HTML(
                value="<div style='border: 2px solid #e0e0e0; border-radius: 8px; background-color: #f8f8f8; padding: 40px; text-align: center;'>"
                      "<div style='display: flex; justify-content: center; align-items: center; margin-bottom: 20px;'>"
                      "<svg width='48' height='48' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>"
                      "<circle cx='12' cy='12' r='10' stroke='#ccc' stroke-width='2' fill='none'/>"
                      "<path d='M12 16v-4' stroke='#ccc' stroke-width='2' stroke-linecap='round'/>"
                      "<path d='M12 8h.01' stroke='#ccc' stroke-width='2' stroke-linecap='round'/>"
                      "</svg>"
                      "</div>"
                      "<div style='color: #666; font-size: 16px; line-height: 1.5;'>"
                      "Convert objects from your object gallery to 3D to export them for Blender"
                      "</div>"
                      "</div>",
                elem_classes=["placeholder-text"],
                visible=True
            )
            
            # Export content (shown when 3D assets are available)
            with gr.Column(visible=False, elem_classes=["export-content-active"]) as export_content_active:
                # Count display
                count_display = gr.HTML(
                    value="<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>0 objects ready to export</div>",
                    elem_classes=["export-count"]
                )
                
                # Thumbnails container
                thumbnails_container = gr.HTML(
                    value="<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'></div>",
                    elem_classes=["thumbnails-container"]
                )
                
                # Export button
                export_btn = gr.Button(
                    "Export objects to file",
                    elem_classes=["export-btn"],
                    interactive=False
                )
    
    return {
        "section": export_section,
        "content": content_container,
        "count_display": count_display,
        "thumbnails_container": thumbnails_container,
        "export_btn": export_btn,
        "placeholder": placeholder,
        "export_content_active": export_content_active,
    }

def update_export_section(gallery_data):
    """Update the export section based on gallery data with 3D assets."""
    if not gallery_data:
        return (
            gr.update(value="<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>0 objects ready to export</div>"),
            gr.update(value="<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'></div>"),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )
    
    # Filter for objects that have 3D models
    exportable_objects = []
    for obj in gallery_data:
        if obj.get("glb_path") and os.path.exists(obj["glb_path"]):
            exportable_objects.append(obj)
    
    count = len(exportable_objects)
    
    if count == 0:
        # No 3D assets to export - show placeholder, hide export content
        return (
            gr.update(value="<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>0 objects ready to export</div>"),
            gr.update(value="<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'></div>"),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )
    
    # Generate thumbnails HTML
    thumbnails_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'>"
    
    for obj in exportable_objects:
        try:
            # Create thumbnail from original image
            if obj.get("path") and os.path.exists(obj["path"]):
                # Read and resize image for thumbnail
                with Image.open(obj["path"]) as img:
                    # Resize to thumbnail size (64x64)
                    img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                    
                    # Convert to base64 for inline display
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Create thumbnail HTML
                    thumbnails_html += f"""
                    <div style='display: inline-block; text-align: center;'>
                        <img src='data:image/png;base64,{img_base64}' 
                             alt='{obj["title"]}' 
                             style='width: 64px; height: 64px; object-fit: cover; border-radius: 8px; border: 2px solid #e0e0e0;'
                             title='{obj["title"]}'>
                    </div>
                    """
            else:
                # Fallback if image not found
                thumbnails_html += f"""
                <div style='display: inline-block; text-align: center;'>
                    <div style='width: 64px; height: 64px; background-color: #f0f0f0; border-radius: 8px; border: 2px solid #e0e0e0; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px;'>
                        {obj["title"][:8]}...
                    </div>
                </div>
                """
        except Exception as e:
            print(f"Error creating thumbnail for {obj['title']}: {e}")
            # Fallback thumbnail
            thumbnails_html += f"""
            <div style='display: inline-block; text-align: center;'>
                <div style='width: 64px; height: 64px; background-color: #f0f0f0; border-radius: 8px; border: 2px solid #e0e0e0; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px;'>
                    {obj["title"][:8]}...
                </div>
            </div>
            """
    
    thumbnails_html += "</div>"
    
    # Has 3D assets - hide placeholder, show export content
    return (
        gr.update(value=f"<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>{count} objects ready to export</div>"),
        gr.update(value=thumbnails_html),
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(visible=True)
    )

def create_export_modal():
    """Create the export modal UI for specifying scene folder name."""
    with gr.Group(elem_id="export-modal", visible=False) as export_modal:
        with gr.Column(elem_classes=["export-modal-content"]):
            gr.Markdown("Export 3D Assets", elem_classes=["modal-title"])
            
            display_path = str(ASSETS_DIR).replace("\\", "\\\\")
            # Information about save location
            gr.Markdown(
                f'Save Location: {display_path}',
                elem_classes=["save-location-info"]
            )
            
            # Scene folder name input
            scene_folder_input = gr.Textbox(
                label="Scene Folder Name",
                placeholder="Enter a name for your scene folder (e.g., 'my_scene', 'kitchen_objects')",
                lines=1,
                elem_classes=["scene-folder-input"]
            )
            
            # Error message (hidden by default)
            error_message = gr.HTML(
                value="",
                visible=False,
                elem_classes=["export-error-message"]
            )
            
            # Action buttons
            with gr.Row():
                cancel_btn = gr.Button("Cancel", variant="secondary", elem_classes=["modal-cancel-btn"])
                save_btn = gr.Button("Save & Export", variant="primary", elem_classes=["modal-save-btn"])
            
    return export_modal, scene_folder_input, error_message, cancel_btn, save_btn



def open_export_modal(gallery_data):
    """Open the export modal."""
    if not gallery_data:
        return gr.update(visible=True)
    
    # Filter for objects that have 3D models
    exportable_objects = []
    for obj in gallery_data:
        if obj.get("glb_path") and os.path.exists(obj["glb_path"]):
            exportable_objects.append(obj)
    
    return gr.update(visible=True)

def close_export_modal():
    """Close the export modal and clear error message."""
    return gr.update(visible=False), "", gr.update(visible=False)

def export_3d_assets_to_folder(gallery_data, folder_name):
    """Export all 3D assets to a specified folder within ASSETS_DIR.
    
    Returns:
        tuple: (modal_update, error_message_update) or just modal_update on success
    """
    if not gallery_data:
        return gr.update(visible=False), gr.update(visible=False)
    
    if not folder_name or not folder_name.strip():
        # Show error message instead of closing
        error_html = "<div style='color: #dc2626; font-size: 14px; padding: 8px 0;'>⚠️ Folder name cannot be empty</div>"
        return gr.update(visible=True), gr.update(value=error_html, visible=True)
    
    # Filter for objects that have 3D models
    exportable_objects = []
    for obj in gallery_data:
        if obj.get("glb_path") and os.path.exists(obj["glb_path"]):
            exportable_objects.append(obj)
    
    if not exportable_objects:
        return gr.update(visible=False), gr.update(visible=False)
    
    try:
        # Create clean folder name
        clean_folder_name = folder_name.strip().replace(" ", "_")
        export_dir = ASSETS_DIR / clean_folder_name
        
        # Create the directory if it doesn't exist
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all GLB files to the export directory
        exported_files = []
        for obj in exportable_objects:
            glb_path = Path(obj["glb_path"])
            if glb_path.exists():
                # Use object title as filename
                target_filename = f"{obj['title'].replace(' ', '_')}.glb"
                target_path = export_dir / target_filename
                
                # Copy the file
                import shutil
                shutil.copy2(glb_path, target_path)
                exported_files.append(target_filename)
        
        return gr.update(visible=False), gr.update(visible=False)
    
    except Exception as e:
        return gr.update(visible=False), gr.update(visible=False) 