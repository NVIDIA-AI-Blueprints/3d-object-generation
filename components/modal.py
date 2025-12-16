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
import os

def open_image_settings(image_path, title, gallery_data=None, card_idx=None):
    """Open settings modal for the clicked image."""
    overlay_html = "<div id='modal-overlay' style='display:block; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    
    # Check if 3D model exists for this card
    glb_path = None
    if gallery_data and card_idx is not None and card_idx < len(gallery_data):
        obj = gallery_data[card_idx]
        if "glb_path" in obj and obj["glb_path"] and os.path.exists(obj["glb_path"]):
            glb_path = obj["glb_path"]
    
    return f"### {title}", image_path, True, overlay_html, glb_path

def close_modal():
    """Close the settings modal."""
    overlay_html = "<div id='modal-overlay' style='display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    return None, None, False, overlay_html, None

def save_settings(image_path, title):
    """Save the image settings."""
    return f"Settings saved for {title}", False

def create_modal():
    """Create the settings modal UI block and return its components."""
    with gr.Group(elem_id="settings-modal") as settings_modal:
        modal_image_title = gr.Markdown("### Image Preview")
        close_btn = gr.Button("X", elem_id="close-modal-btn", size="sm")
        with gr.Tabs():
            with gr.TabItem("2D"):
                with gr.Row():
                    with gr.Column():
                        modal_image = gr.Image(
                            label="Selected Image",
                            interactive=False,
                            height=350
                        )
            with gr.TabItem("3D"):
                with gr.Column():
                    # 3D model container - will be updated dynamically
                    modal_3d = gr.Model3D(
                        value=None,
                        label="3D Model Preview",
                        clear_color=[0.0, 0.0, 0.0, 0.0],
                        height=400,
                        visible=False,
                        interactive=False
                    )
                    # Message when no 3D model exists
                    no_3d_message = gr.Markdown(
                        "### No 3D Model Available\n\nPlease click the **→ 3D** button on the image card to generate a 3D model.",
                        visible=True
                    )
    return settings_modal, modal_image_title, close_btn, modal_image, modal_3d, no_3d_message

def create_edit_modal():
    """Create a modal for editing image title and description."""
    with gr.Group(elem_id="edit-modal") as edit_modal:
        edit_title = gr.Textbox(
            label="", 
            placeholder="Enter image title..."
        )
        edit_description = gr.Textbox(
            label="", 
            lines=5, 
            placeholder="Enter a detailed description for the image generation..."
        )
        with gr.Row():
            cancel_btn = gr.Button("Cancel", elem_id="cancel-edit-btn", variant="secondary")
            update_btn = gr.Button("Update & Generate", elem_id="update-edit-btn", variant="secondary")
    return edit_modal, edit_title, edit_description, cancel_btn, update_btn

def create_start_over_confirmation_modal():
    """Create a confirmation modal for the start over button."""
    with gr.Group(elem_id="start-over-confirmation-modal", visible=False) as confirmation_modal:
        gr.Markdown(
            "### Start Over with New Scene",
            elem_classes=["modal-title"]
        )
        gr.Markdown(
            "⚠️ **Warning:** This action will clear your current scene and all generated assets.\n\n"
            "**Please make sure you have saved any 3D assets if required before proceeding.**\n\n"
            "Are you sure you want to start over?",
            elem_classes=["confirmation-message"]
        )
        with gr.Row():
            cancel_btn = gr.Button("Cancel", variant="secondary", elem_classes=["modal-cancel-btn"])
            confirm_btn = gr.Button("Yes, Start Over", variant="stop", elem_classes=["modal-confirm-btn"])
    return confirmation_modal, cancel_btn, confirm_btn

def open_start_over_confirmation():
    """Open the start over confirmation modal."""
    overlay_html = "<div id='modal-overlay' style='display:block; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    return overlay_html

def close_start_over_confirmation():
    """Close the start over confirmation modal."""
    overlay_html = "<div id='modal-overlay' style='display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    return overlay_html 