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


"""Main Chat-to-3D Gradio Application.

This is the entry point for the Chat-to-3D application that integrates:
- LLM Agent for scene planning
- Object gallery UI 
- 3D generation pipeline

Configuration:
- ENABLE_STATUS_PANEL: Set to True to enable the status console panel
"""

import gradio as gr
import os
import base64
import signal
import sys
import time
import socket
import logging
import datetime
import config
from components.chat_interface import create_chat_interface, handle_scene_description
from components.image_gallery import create_image_gallery
from components.blender_export import create_blender_export_section, update_export_section, create_export_modal, open_export_modal, close_export_modal, export_3d_assets_to_folder
from components.status_panel import create_status_panel
from components.modal import create_modal, open_image_settings, close_modal, create_edit_modal, create_start_over_confirmation_modal, open_start_over_confirmation, close_start_over_confirmation
from components.image_card import create_refresh_handler, create_3d_generation_handler, create_convert_all_3d_handler, invalidate_3d_model
from services import AgentService, ImageGenerationService, Model3DService
# Import GPU memory manager for native model coordination
if config.USE_NATIVE_LLM or config.USE_NATIVE_TRELLIS:
    from services.gpu_memory_manager import get_gpu_memory_manager
import threading
import subprocess
import requests
import gc
import torch
from pathlib import Path
from nim_llm.manager import stop_container
import shutil
from utils import (
    clear_image_generation_failure_flags, 
    disable_all_buttons_for_3d_generation, 
    enable_all_buttons_after_3d_generation,
    disable_all_buttons_for_image_operations,
    enable_all_buttons_after_image_operations,
    is_llm_should_be_stopped
)

# Set up logging for termination server
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag to track if we're shutting down
_shutdown_requested = False

# Global termination server thread
_termination_server_thread = None
_termination_server_socket = None

class TerminationServer:
    """Server that listens for termination requests from external clients."""
    
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
    def start(self):
        """Start the termination server in a separate thread."""
        def handle_termination():
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True
            logger.info(f"Termination server started on {self.host}:{self.port}")

            while self.running:
                try:
                    self.server_socket.settimeout(1.0)  # 1 second timeout for accept
                    conn, addr = self.server_socket.accept()
                    with conn:
                        data = conn.recv(1024)
                        # Handle empty data gracefully
                        if not data:
                            logger.debug(f"Empty data received from {addr}, ignoring")
                            continue
                        elif data == b'terminate':
                            logger.info(f"Termination signal received from {addr}")
                            # Send PID to client
                            pid = os.getpid()
                            logger.info(f"Sending PID {pid} to client")
                            conn.send(f"terminating:{pid}".encode())
                            # Let client handle the termination
                            logger.info("PID sent, client will handle termination")
                        else:
                            logger.warning(f"Invalid command received: {data}")
                            conn.send(b'error: invalid command')
                except socket.timeout:
                    # Timeout is expected, continue the loop
                    continue
                except Exception as e:
                    if self.running:  # Only log if we're supposed to be running
                        logger.error(f"Error handling connection: {e}")
                    break

        self.thread = threading.Thread(target=handle_termination, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the termination server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logger.error(f"Error closing termination server socket: {e}")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested, _termination_server_thread
    print(f"\nReceived signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True
    
    # Stop the termination server
    if _termination_server_thread:
        _termination_server_thread.stop()
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Load custom CSS and JS
try:
    with open(config.CUSTOM_CSS_FILE) as f:
        custom_css = f.read()
except FileNotFoundError:
    print("Custom CSS and JS not found")
    custom_css = ""

# Load NVIDIA logo
try:
    with open(config.NVIDIA_LOGO_FILE, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    
    nvidia_html = f"""<div class='nvidia-logo-container'> 
            <img src='data:image/png;base64,{encoded}' alt='NVIDIA Logo' class='nvidia-logo' width='24' height='24'> 
            <span class='nvidia-text'></span> 
            </div>"""
except FileNotFoundError:
    # Fallback if logo not found
    nvidia_html = """<div class='nvidia-logo-container'> 
            <span class='nvidia-text'>Chat-to-3D</span> 
            </div>"""

# Background bootstrap for LLM and Trellis NIMs
_nim_bootstrap_started = False
_trellis_bootstrap_started = False
_nim_process = None  # Store reference to the LLM NIM process
_trellis_process = None  # Store reference to the Trellis NIM process

# Global state to track if we're in workspace mode
_in_workspace_mode = False

# Configuration flag to enable/disable status panel
# Set to True to enable the status console panel functionality
ENABLE_STATUS_PANEL = False

def _ensure_llm_nim_started():
    """Start the LLM NIM container in the background if it's not already healthy."""
    global _nim_bootstrap_started
    if _nim_bootstrap_started:
        return
    _nim_bootstrap_started = True

    health_url = f"{config.AGENT_BASE_URL}/health/ready"
    try:
        resp = requests.get(health_url, timeout=1.5)
        if resp.status_code == 200:
            print("LLM NIM already running")
            return
    except Exception:
        pass

    def _runner():
        global _nim_process
        try:
            script_path = Path(__file__).parent / "nim_llm" / "run_llama.py"
            print(f"Starting LLM NIM via {script_path}")
            popen_kwargs = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            _nim_process = subprocess.Popen([sys.executable, str(script_path)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, **popen_kwargs)
        except Exception as e:
            print(f"Failed to start LLM NIM: {e}")

    threading.Thread(target=_runner, daemon=True).start()

def _ensure_trellis_nim_started():
    """Start the Trellis NIM container in the background if it's not already healthy."""
    global _trellis_bootstrap_started
    if _trellis_bootstrap_started:
        return
    _trellis_bootstrap_started = True

    health_url = f"{config.TRELLIS_BASE_URL}/health/ready"
    try:
        resp = requests.get(health_url, timeout=1.5)
        if resp.status_code == 200:
            print("Trellis NIM already running")
            return
    except Exception:
        pass

    def _runner():
        global _trellis_process
        try:
            script_path = Path(__file__).parent / "nim_trellis" / "run_trellis.py"
            print(f"Starting Trellis NIM via {script_path}")
            popen_kwargs = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            _trellis_process = subprocess.Popen([sys.executable, str(script_path)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, **popen_kwargs)
        except Exception as e:
            print(f"Failed to start Trellis NIM: {e}")

    threading.Thread(target=_runner, daemon=True).start()

def _ensure_all_nims_started():
    """Start NIM containers based on config settings."""
    # Only start LLM NIM if not using native LLM
    if not config.USE_NATIVE_LLM:
        _ensure_llm_nim_started()
    # Only start Trellis NIM if not using native TRELLIS
    if not config.USE_NATIVE_TRELLIS:
        _ensure_trellis_nim_started()


def stop_llm_container(force=False):
    """Stop the LLM container after workspace transition."""
    # Skip if using native LLM (no container to stop)
    if config.USE_NATIVE_LLM:
        return
    # Only proceed if we're in workspace mode (valid scene input)
    global _in_workspace_mode, _nim_bootstrap_started
    if not _in_workspace_mode and not force:
        return
    if not force and not is_llm_should_be_stopped():
        print("LLM NIM container is not stopping because VRAM threshold is met")
        return
    try:
        print("Stopping LLM NIM container...")
        print(f"Timestamp before stop_container: {time.time()}")
        success = stop_container()
        print(f"Timestamp after stop_container: {time.time()}")
        time.sleep(2)
        gc.collect()
        torch.cuda.empty_cache()
        _nim_bootstrap_started = False
        if success:
            print("LLM NIM container stopped and bootstrap reset")
            print(f"Timestamp after container stop completion: {time.time()}")
        else:
            print("LLM NIM container stop command executed (may not have been running)")
            print(f"Timestamp after container stop completion: {time.time()}")
    except Exception as e:
        print(f"Error stopping LLM container: {e}")
        time.sleep(2)
        gc.collect()
        torch.cuda.empty_cache()
        _nim_bootstrap_started = False

def stop_trellis_container(force=True):
    """Stop the Trellis container after workspace transition."""
    # Only proceed if we're in workspace mode (valid scene input)
    global _trellis_bootstrap_started
 
    
    try:
        print("Stopping Trellis NIM container...")
        from nim_trellis.manager import stop_container as stop_trellis_container_func
        success = stop_trellis_container_func()
        time.sleep(2)
        gc.collect()
        torch.cuda.empty_cache()
        _trellis_bootstrap_started = False
        if success:
            print("Trellis NIM container stopped and bootstrap reset")
        else:
            print("Trellis NIM container stop command executed (may not have been running)")
    except Exception as e:
        print(f"Error stopping Trellis container: {e}")
        time.sleep(2)
        gc.collect()
        torch.cuda.empty_cache()
        _trellis_bootstrap_started = False

def delete_assets_dir():
    """Delete the assets directory."""

    if os.path.exists(config.GENERATED_IMAGES_DIR):
        shutil.rmtree(config.GENERATED_IMAGES_DIR)
    if os.path.exists(config.MODELS_DIR):
        shutil.rmtree(config.MODELS_DIR)

def create_app():
    """Create and configure the main Gradio application."""
    
    # Initialize services
    agent_service = AgentService()
    image_generation_service = ImageGenerationService()
    model_3d_service = Model3DService()

    # Register services with GPU memory manager for coordinated memory management
    if config.USE_NATIVE_LLM or config.USE_NATIVE_TRELLIS:
        gpu_manager = get_gpu_memory_manager()
        gpu_manager.register_llm_service(agent_service)
        gpu_manager.register_sana_service(image_generation_service)
        gpu_manager.register_trellis_service(model_3d_service)
        print("GPU Memory Manager initialized - services registered")
        
        # Pre-load all models at startup
        # This loads TRELLIS, SANA, and LLM, then moves TRELLIS and SANA to CPU
        # LLM stays on GPU ready for chat
        print("\n" + "=" * 60)
        print("PRE-LOADING MODELS - Please wait...")
        print("=" * 60)
        preload_status = gpu_manager.preload_all_models()
        print(f"Pre-loading complete in {preload_status['total_time']:.2f} seconds")

    delete_assets_dir()

    # Kick off both NIM containers in background if needed (non-blocking)
    _ensure_all_nims_started()

    # Start the termination server
    termination_server = TerminationServer()
    global _termination_server_thread
    _termination_server_thread = termination_server
    try:
        termination_server.start()
        print("Termination server started on localhost:12345")
    except Exception as e:
        print(f"Failed to start termination server: {e}")
        print("   External termination requests will not be available")
        _termination_server_thread = None

    with gr.Blocks(
        title="3D-Object-Generator", 
        # css_paths=["static/css/custom.css"]
    ) as app:
        
        
        # Inject custom CSS and JS
        gr.HTML(f"<style>{custom_css}</style>")
        
        # Header with NVIDIA branding and status
        with gr.Row(elem_classes=["header-section"]):
            with gr.Column(scale=1):
                with gr.Row():
                    gr.HTML(nvidia_html)
            with gr.Column(scale=1):
                with gr.Row(elem_classes=["status-row"]):
                    llm_status = gr.HTML("""
                    <div>
                        <span class="status-text"></span>
                    </div>
                    """)
                    refresh_status_btn = gr.Button("🔄", elem_classes=["refresh-status-btn"], size="sm", visible=False)
                    toggle_btn = gr.Button(">", elem_classes=["toggle-status-btn"], size="sm")
            
        # Global spinner overlay shown until both LLM and Trellis are ready
        llm_spinner = gr.HTML(
            """
            <div class="llm-spinner-overlay">
                <div class="llm-spinner-content">
                    <div class="llm-spinner-ring"></div>
                    <div>
                        <div class="llm-spinner-title">Loading LLM and Trellis models</div>
                        <div class="llm-spinner-subtitle">This could take a few minutes...</div>
                    </div>
                </div>
            </div>
            """,
            visible=True,
        )
        
        # Main layout
        with gr.Row(elem_classes=["content-row"]):
            # Left: Chat and Gallery
            with gr.Column(scale=4, elem_classes=["main-content", "landing"]) as main_col:
                # Chat interface (landing screen) - hidden until LLM is ready
                chat_components = create_chat_interface()
                chat_components["section"].visible = False
                # Don't set visible=False here, let the Timer control it

                # Workspace (hidden initially): gallery + export
                with gr.Column(visible=False, elem_classes=["workspace-section"]) as workspace_section:
                    with gr.Row():
                        start_over_btn = gr.Button("← Start over with a new scene prompt", elem_classes=["start-over-btn"], size="sm")
                    # Object gallery
                    gallery_components = create_image_gallery()
                    
                    # Blender export section
                    export_components = create_blender_export_section()
                
                # Export status textbox
                export_status = gr.Textbox(label="Export Status", interactive=False, visible=False)
                
                # Modal UI
                modal_visible = gr.State(False)
                overlay = gr.HTML("""
                <div id='modal-overlay' style='display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>
                """)
                
                # Modal components
                settings_modal, modal_image_title, close_btn, modal_image, modal_3d, no_3d_message = create_modal()
                settings_modal.visible = False
                
                # Edit modal components
                edit_modal, edit_title, edit_description, cancel_edit_btn, update_edit_btn = create_edit_modal()
                edit_modal.visible = False
                edit_current_index = gr.State(None)
                
                # Export modal components
                export_modal, scene_folder_input, export_cancel_btn, export_save_btn = create_export_modal()
                export_modal.visible = False
                
                # Start over confirmation modal components
                confirmation_modal, confirmation_cancel_btn, confirmation_confirm_btn = create_start_over_confirmation_modal()
                   
                # Session state to track legitimate transitions for browser refresh detection
                session_transition_counter = gr.State(0)
                
            # Right: Status panel
            # Right: Status panel (conditionally enabled)
            if ENABLE_STATUS_PANEL:
                with gr.Column(scale=1, elem_classes=["right-panel"], visible=False) as right_panel:
                    status_components = create_status_panel()
            else:
                # Placeholder for when status panel is disabled
                right_panel = gr.Column(visible=False)
                status_components = {"close_btn": gr.Button(visible=False)}
        
        # Wire up the event handlers
        def process_scene_description(scene_description, gallery_data):
            """Process scene description and generate objects, then update gallery."""
            if not scene_description.strip():
                tip_html = """
                <div class="tip-message">
                    <span class="tip-icon">💡</span>
                    <span class="tip-text">Please enter a scene description.</span>
                </div>
                """
                return "", gallery_data, tip_html, True
            
            message, new_gallery_data, tip_html, show_tip = handle_scene_description(scene_description, agent_service, gallery_data, None)
            
            # If we should show a tip (non-scene input), don't proceed with gallery updates
            if show_tip:
                return "", gallery_data, tip_html, True
            
            # reset agent memory
            agent_service.clear_memory()
            
            # If it's a valid scene, proceed with normal flow
            return message, new_gallery_data, "", False
        
        def handle_scene_with_conditional_flow(scene_description, gallery_data):
            """Handle scene processing with conditional workspace transition."""
            message, new_gallery_data, tip_html, show_tip = process_scene_description(scene_description, gallery_data)
            
            # If it's a valid scene, proceed with all the normal flow
            if not show_tip:
                # This will trigger the workspace transition
                return message, new_gallery_data, "", False, True
            else:
                # This will just show the tip and stay on first screen
                return "", gallery_data, tip_html, True, False
        
        def handle_scene_input(scene_description, gallery_data):
            """Handle scene input and proceed with workspace transition."""
            message, new_gallery_data, tip_html, show_tip, proceed_to_workspace = handle_scene_with_conditional_flow(scene_description, gallery_data)
            
            if proceed_to_workspace:
                # Valid scene - proceed with normal flow, hide tip
                return message, new_gallery_data, gr.update(value="", visible=False)
            else:
                # Non-scene input - don't update gallery, show tip
                return "", gallery_data, gr.update(value=tip_html, visible=True)
        
        # Helper to reveal workspace and switch layout out of landing mode
        def reveal_workspace(gallery_data, current_counter):
            global _in_workspace_mode
            # Only proceed with workspace transition if there's actual gallery data
            if not gallery_data or len(gallery_data) == 0:
                # No gallery data means it was a non-scene input, don't transition
                return (
                    gr.update(visible=False),                 # keep workspace hidden
                    gr.update(elem_classes=["main-content", "landing"]), # keep landing centering
                    gr.update(visible=True),                  # keep chat section visible
                    current_counter,                          # keep current counter (4 outputs expected)
                )
            
            # Valid scene with gallery data - proceed with transition
            new_counter = current_counter + 1
            _in_workspace_mode = True
            print(f"Transitioning to workspace mode, counter: {new_counter}")
            return (
                gr.update(visible=True),                 # show workspace
                gr.update(elem_classes=["main-content"]), # remove landing centering
                gr.update(visible=False),                # hide chat section
                new_counter,                             # increment counter
            )

        # Helper to reset all UI/state and return to landing
        def go_to_first_screen():
            global _in_workspace_mode
            _in_workspace_mode = False
            
            # Clean up SANA pipeline when going back to first screen
            print("Cleaning up SANA pipeline...")
            try:
                image_generation_service.cleanup_sana_pipeline()
                print("SANA pipeline cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up SANA pipeline: {e}")
            
            # Check if LLM is ready before showing chat (Trellis can load in background)
            if config.USE_NATIVE_LLM:
                llm_ready = True  # Native LLM is always ready
            else:
                llm_health_url = f"{config.AGENT_BASE_URL}/health/ready"
                try:
                    llm_resp = requests.get(llm_health_url, timeout=1.0)
                    llm_ready = (llm_resp.status_code == 200)
                except Exception:
                    llm_ready = False
            
            # Check Trellis health (skip if using native TRELLIS)
            if config.USE_NATIVE_TRELLIS:
                trellis_ready = True  # Native TRELLIS is always ready (loaded on demand)
            else:
                trellis_health_url = f"{config.TRELLIS_BASE_URL}/health/ready"
                try:
                    trellis_resp = requests.get(trellis_health_url, timeout=1.0)
                    trellis_ready = (trellis_resp.status_code == 200)
                except Exception:
                    trellis_ready = False
                
            # Start the NIM services again if needed (only for non-native backends)
            if (not llm_ready and not config.USE_NATIVE_LLM) or (not trellis_ready and not config.USE_NATIVE_TRELLIS):
                _ensure_all_nims_started()
            
            return (
                gr.update(visible=False),               # hide workspace
                gr.update(elem_classes=["main-content", "landing"]),  # restore landing centering
                gr.update(visible=llm_ready),           # show chat section when LLM is ready
                gr.update(value=""),                   # clear chat input
                gr.update(visible=False),               # hide export status
                gr.update(visible=False) if ENABLE_STATUS_PANEL else gr.update(visible=False),  # hide right panel if open
                gr.update(active=True),                 # restart health timer
            )

        # New: Mark items as image-generating to disable Start Over immediately
        def mark_images_generating(gallery_data):
            if not gallery_data:
                return gallery_data
            
            # Only proceed if we're in workspace mode (valid scene input)
            global _in_workspace_mode
            if not _in_workspace_mode:
                return gallery_data
            
            updated_data = []
            for obj in gallery_data:
                new_obj = obj.copy()
                new_obj["image_generating"] = True
                updated_data.append(new_obj)
            return updated_data
        
        # New: Generate images for all objects after moving to workspace
        def generate_images_for_gallery(gallery_data):
            print(f"Timestamp before generate_images_for_gallery: {time.time()}")
            # Only proceed if we're in workspace mode (valid scene input)
            global _in_workspace_mode
            if not _in_workspace_mode:
                return gallery_data
            
            print(f"Timestamp after generate_images_for_gallery: {time.time()}")
            try:
                if not gallery_data:
                    return gallery_data
                print("Generating images for all objects (step 2)...")
                print(f"Timestamp before generate_images_for_objects: {time.time()}")
                success, message, generated_images = image_generation_service.generate_images_for_objects(gallery_data, output_dir=config.GENERATED_IMAGES_DIR)
                print(f"Timestamp after generate_images_for_objects: {time.time()}")
                
                # Move SANA to CPU after image generation
                print("Moving SANA to CPU after image generation...")
                image_generation_service.move_sana_pipeline_to_cpu()
                
                # Clear GPU cache (but don't synchronize - it blocks system-wide)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                print("GPU memory cleared, ready for image display")
                
                if success and generated_images:
                    updated_data = []
                    for obj in gallery_data:
                        object_name = obj.get("title")
                        new_obj = obj.copy()
                        if object_name in generated_images:
                            new_obj["path"] = generated_images[object_name]
                            # Clear any previous failure flags since image generation succeeded
                            new_obj = clear_image_generation_failure_flags(new_obj)
                            print(f"Generated image for {object_name}: {generated_images[object_name]}")
                        else:
                            print(f"No image generated for {object_name}")
                        # Clear image generation flag after completion
                        if "image_generating" in new_obj:
                            new_obj["image_generating"] = False
                        updated_data.append(new_obj)
                    return updated_data
                else:
                    print(f"Image generation failed: {message}")
                    # Clear image generation flag even on failure
                    if not gallery_data:
                        return gallery_data
                    updated_data = []
                    for obj in gallery_data:
                        new_obj = obj.copy()
                        if "image_generating" in new_obj:
                            new_obj["image_generating"] = False
                        updated_data.append(new_obj)
                    return updated_data
            except Exception as e:
                print(f"Error during image generation: {str(e)}")
                # Clear image generation flag even on exception
                if not gallery_data:
                    return gallery_data
                updated_data = []
                for obj in gallery_data:
                    new_obj = obj.copy()
                    if "image_generating" in new_obj:
                        new_obj["image_generating"] = False
                    updated_data.append(new_obj)
                return updated_data
        
        # Toggle start-over availability based on processing state
        def update_start_over_state(gallery_data):
            """Update the Start Over button state based on gallery data."""
            # Only proceed if we're in workspace mode (valid scene input)
            global _in_workspace_mode
            if not _in_workspace_mode:
                return gr.update(visible=False)
            
            if not gallery_data:
                return gr.update(visible=False)
            
            # Check if any items are being processed
            any_processing = any(
                obj.get("image_generating", False) or 
                obj.get("3d_generating", False) or 
                obj.get("batch_processing", False)
                for obj in gallery_data
            )
            
            if any_processing:
                return gr.update(visible=False)  # Hide button during processing
            else:
                return gr.update(visible=True)   # Show button when ready
        
        # Health check function for both LLM and Trellis NIMs; updates status and controls UI visibility
        def check_services_health(current_counter):
            global _in_workspace_mode

            if _in_workspace_mode and current_counter == 0:
                # add kill app logic here
                print("DETECTED BROWSER REFRESH: In workspace mode but counter is 0")
                print("This indicates browser refresh or state corruption - killing app")
                print("Cleaning up resources...")
                
                # Stop the termination server
                if _termination_server_thread:
                    print("Stopping termination server...")
                    try:
                        _termination_server_thread.stop()
                        print("Termination server stopped")
                    except Exception as e:
                        print(f"Error stopping termination server: {e}")
                
                # Stop both NIM containers
                print("Stopping LLM NIM container...")
                stop_llm_container(force=True)
                
                print("Stopping Trellis NIM container...")
                stop_trellis_container(force=True)
                
                print("Cleanup completed")
                print("Application shutdown complete")
                os._exit(0)
            
            
            # Check LLM health (skip if using native LLM)
            if config.USE_NATIVE_LLM:
                # Native LLM is always "ready" (loaded on demand)
                llm_ready = True
            else:
                llm_health_url = f"{config.AGENT_BASE_URL}/health/ready"
                try:
                    llm_resp = requests.get(llm_health_url, timeout=1.0)
                    llm_ready = (llm_resp.status_code == 200)
                except Exception:
                    llm_ready = False
            
            # Check Trellis health (skip if using native TRELLIS)
            if config.USE_NATIVE_TRELLIS:
                trellis_ready = True  # Native TRELLIS is always ready (loaded on demand)
            else:
                trellis_health_url = f"{config.TRELLIS_BASE_URL}/health/ready"
                try:
                    trellis_resp = requests.get(trellis_health_url, timeout=1.0)
                    trellis_ready = (trellis_resp.status_code == 200)
                except Exception:
                    trellis_ready = False
            
            # Build status display
            llm_color = "#16be16" if llm_ready else "#f59e0b"
            trellis_color = "#16be16" if trellis_ready else "#f59e0b"
            llm_label = "LLM: Native" if config.USE_NATIVE_LLM else ("LLM: ready" if llm_ready else "LLM: Unloaded")
            trellis_label = "Trellis: Native" if config.USE_NATIVE_TRELLIS else ("Trellis: ready" if trellis_ready else "Trellis: Loading...")
            
            status_html = f'''
            <div class="status-section">
                <span class="status-text" style="color:{llm_color}">{llm_label}</span> | 
                <span class="status-text" style="color:{trellis_color}">{trellis_label}</span>
            </div>
            '''
            
            # Only LLM needs to be ready to show landing page
            # Trellis will continue loading in background and must be ready before 3D generation
            show_spinner = not llm_ready and not _in_workspace_mode
            # Show chat when LLM is ready (don't wait for Trellis)
            show_chat = llm_ready and not _in_workspace_mode
            # Show refresh button when in workspace mode, hide when in landing mode
            show_refresh = True
            # Stop timer if we're in workspace mode
            timer_active = not _in_workspace_mode
            return gr.update(visible=show_spinner), gr.update(value=status_html), gr.update(visible=show_chat), gr.update(visible=show_refresh), gr.update(active=timer_active)
        
        # Timer for initial health polling (only active until we reach workspace mode)
        health_timer = gr.Timer(5, active=True)
        health_timer.tick(
            fn=check_services_health,
            inputs=[session_transition_counter],
            outputs=[llm_spinner, llm_status, chat_components["section"], refresh_status_btn, health_timer]
        )
        
        # Wire up manual refresh button
        refresh_status_btn.click(
            fn=check_services_health,
            inputs=[session_transition_counter],
            outputs=[llm_spinner, llm_status, chat_components["section"], refresh_status_btn, health_timer]
        )
        
        # Connect send button to process scene description
        chat_components["send_btn"].click(
            fn=handle_scene_input,
            inputs=[chat_components["input"], gallery_components["data"]],
            outputs=[chat_components["input"], gallery_components["data"], chat_components["tip"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_export_section,
            inputs=[gallery_components["data"]],
            outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
        ).then(
            fn=reveal_workspace,
            inputs=[gallery_components["data"], session_transition_counter],
            outputs=[workspace_section, main_col, chat_components["section"], session_transition_counter]
        ).then(
            fn=mark_images_generating,
            inputs=[gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_start_over_state,
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        ).then(
            fn=stop_llm_container,
            inputs=[],
            outputs=[]
        ).then(
            fn=generate_images_for_gallery,
            inputs=[gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_export_section,
            inputs=[gallery_components["data"]],
            outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
        ).then(
            fn=update_start_over_state,
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        )
        
        # Connect Enter key for scene input
        chat_components["input"].submit(
            fn=handle_scene_input,
            inputs=[chat_components["input"], gallery_components["data"]],
            outputs=[chat_components["input"], gallery_components["data"], chat_components["tip"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_export_section,
            inputs=[gallery_components["data"]],
            outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
        ).then(
            fn=reveal_workspace,
            inputs=[gallery_components["data"], session_transition_counter],
            outputs=[workspace_section, main_col, chat_components["section"], session_transition_counter]
        ).then(
            fn=mark_images_generating,
            inputs=[gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_start_over_state,
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        ).then(
            fn=stop_llm_container,
            inputs=[],
            outputs=[]
        ).then(
            fn=generate_images_for_gallery,
            inputs=[gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_export_section,
            inputs=[gallery_components["data"]],
            outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
        ).then(
            fn=update_start_over_state,
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        )

        # Start over button: show confirmation dialog first
        start_over_btn.click(
            fn=open_start_over_confirmation,
            inputs=[],
            outputs=[overlay]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[confirmation_modal]
        )
        
        # Confirmation modal cancel button: close modal
        confirmation_cancel_btn.click(
            fn=close_start_over_confirmation,
            inputs=[],
            outputs=[overlay]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[confirmation_modal]
        )
        
        # Confirmation modal confirm button: proceed with start over
        def clear_gallery_state(_):
            return []

        confirmation_confirm_btn.click(
            fn=close_start_over_confirmation,
            inputs=[],
            outputs=[overlay]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[confirmation_modal]
        ).then(
            fn=clear_gallery_state,
            inputs=[gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=go_to_first_screen,
            inputs=[],
            outputs=[workspace_section, main_col, chat_components["section"], chat_components["input"], export_status, right_panel, health_timer]
        )
        
        # Connect toggle button to show/hide right panel (only if status panel is enabled)
        if ENABLE_STATUS_PANEL:
            toggle_btn.click(
                fn=lambda: gr.update(visible=True),
                outputs=[right_panel]
            )
            
            # Connect close button to hide right panel
            status_components["close_btn"].click(
                fn=lambda: gr.update(visible=False),
                outputs=[right_panel]
            )
        else:
            # Hide toggle button when status panel is disabled
            toggle_btn.visible = False
        
        # Modal functionality
        def debug_card_click(path, title, gallery_data, card_idx):
            print(f"DEBUG: Card clicked! Path: {path}, Title: {title}, Card Index: {card_idx}")
            return open_image_settings(path, title, gallery_data, card_idx)
        
        # Wire up card click events for modal
        for idx, card in enumerate(gallery_components["card_components"]):
            def create_dynamic_click_handler(card_idx):
                def click_handler(gallery_data):
                    if card_idx < len(gallery_data):
                        item = gallery_data[card_idx]
                        print(f"DEBUG: Dynamic click for card {card_idx}: {item['title']}")
                        return debug_card_click(item["path"], item["title"], gallery_data, card_idx)
                    else:
                        print(f"DEBUG: Card {card_idx} not found in gallery_data")
                        return debug_card_click("", "", gallery_data, card_idx)
                return click_handler
            
            card["card_click_btn"].click(
                fn=create_dynamic_click_handler(idx),
                inputs=[gallery_components["data"]],
                outputs=[modal_image_title, modal_image, modal_visible, overlay, modal_3d]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[settings_modal]
            ).then(
                fn=lambda glb_path: (gr.update(visible=(glb_path is not None)), gr.update(visible=(glb_path is None))),
                inputs=[modal_3d],
                outputs=[modal_3d, no_3d_message]
            )
        
        # Close modal button
        close_btn.click(
            fn=close_modal,
            inputs=[],
            outputs=[modal_image_title, modal_image, modal_visible, overlay, modal_3d]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[settings_modal]
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[modal_3d, no_3d_message]
        )
        
        # Edit modal functionality
        def open_edit_modal(idx, gallery_data):
            """Open edit modal for the specified card index."""
            if idx < len(gallery_data):
                item = gallery_data[idx]
                return (
                    gr.update(visible=True),  # Show modal
                    idx,                     # Set current index
                    item["title"],           # Populate title
                    item["description"]      # Populate description
                )
            else:
                return (gr.update(visible=True), idx, "", "")
        
        # Create refresh handler
        refresh_handler = create_refresh_handler(image_generation_service)
        
        # Create 3D generation handler
        three_d_handler = create_3d_generation_handler(model_3d_service)
        
        # Create convert all to 3D handler (two-stage process)
        disable_buttons_handler, convert_all_handler = create_convert_all_3d_handler(model_3d_service)
        
        def update_modal_3d_components(gallery_data, card_idx):
            """Update modal 3D components based on 3D model availability."""
            if card_idx < len(gallery_data):
                obj = gallery_data[card_idx]
                if obj.get("glb_path") and os.path.exists(obj["glb_path"]):
                    return gr.update(value=obj["glb_path"], visible=True), gr.update(visible=False)
                else:
                    return gr.update(value=None, visible=False), gr.update(visible=True)
            else:
                return gr.update(value=None, visible=False), gr.update(visible=True)
        
        # Wire up refresh button events for each card
        for idx, card in enumerate(gallery_components["card_components"]):
            def create_refresh_function(card_idx):
                def immediate_disable_buttons(gallery_data):
                    """First stage: immediately disable all buttons."""
                    if card_idx < len(gallery_data):
                        # Create a copy and mark as generating
                        updated_data = gallery_data.copy()
                        updated_data[card_idx]["image_generating"] = True
                        print(f"DEBUG: Set image_generating=True for card {card_idx}")
                        
                        # Disable all buttons globally
                        updated_data = disable_all_buttons_for_image_operations(updated_data)
                        
                        # Return the updated data immediately to show generating state
                        return updated_data
                    else:
                        print(f"DEBUG: Card index {card_idx} out of range")
                        return gallery_data
                
                def perform_image_refresh(gallery_data):
                    """Second stage: perform the actual image refresh."""
                    print(f"DEBUG: Performing actual image refresh for card {card_idx}")
                    result = refresh_handler(card_idx, gallery_data)
                    
                    # Re-enable all buttons after image refresh completes (success or failure)
                    result = enable_all_buttons_after_image_operations(result)
                    
                    return result
                
                return immediate_disable_buttons, perform_image_refresh
            
            # Create the functions for this specific card
            immediate_disable_fn, refresh_fn = create_refresh_function(idx)
            
            # First click: immediate UI update to disable buttons
            card["refresh_btn"].click(
                fn=immediate_disable_fn,
                inputs=[gallery_components["data"]],
                outputs=[gallery_components["data"]]
            ).then(
                fn=gallery_components["shift_card_ui"],
                inputs=[gallery_components["data"]],
                outputs=gallery_components["get_all_card_outputs"]()
            ).then(
                fn=update_start_over_state,              # immediately disable Start Over
                inputs=[gallery_components["data"]],
                outputs=[start_over_btn]
            ).then(
                fn=refresh_fn,
                inputs=[gallery_components["data"]],
                outputs=[gallery_components["data"]]
            ).then(
                fn=gallery_components["shift_card_ui"],
                inputs=[gallery_components["data"]],
                outputs=gallery_components["get_all_card_outputs"]()
            ).then(
                fn=update_export_section,
                inputs=[gallery_components["data"]],
                outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
            ).then(
                fn=update_start_over_state,
                inputs=[gallery_components["data"]],
                outputs=[start_over_btn]
            )
        
        # Wire up 3D generation button events for each card
        for idx, card in enumerate(gallery_components["card_components"]):
            def create_3d_function(card_idx):
                def generate_3d_for_card(gallery_data):
                    # Immediately update the button to show "generating" state
                    if card_idx < len(gallery_data):
                        # Create a copy and mark as generating
                        updated_data = gallery_data.copy()
                        updated_data[card_idx]["3d_generating"] = True
                        print(f"DEBUG: Set 3d_generating=True for card {card_idx}")
                        
                        # Disable all buttons globally if VRAM threshold is met
                        updated_data = disable_all_buttons_for_3d_generation(updated_data)
                        
                        # Return the updated data immediately to show "⏳ 3D..." state
                        return updated_data
                    else:
                        print(f"DEBUG: Card index {card_idx} out of range")
                        return gallery_data
                
                def perform_3d_generation(gallery_data):
                    print(f"DEBUG: Performing actual 3D generation for card {card_idx}")
                    result = three_d_handler(card_idx, gallery_data)
                    
                    # Re-enable all buttons after 3D generation completes (success or failure)
                    result = enable_all_buttons_after_3d_generation(result)
                    
                    return result
                
                return generate_3d_for_card, perform_3d_generation
            
            # Create the functions for this specific card
            immediate_update_fn, generation_fn = create_3d_function(idx)
            
            # First click: immediate UI update
            card["to_3d_btn"].click(
                fn=immediate_update_fn,
                inputs=[gallery_components["data"]],
                outputs=[gallery_components["data"]]
            ).then(
                fn=gallery_components["shift_card_ui"],
                inputs=[gallery_components["data"]],
                outputs=gallery_components["get_all_card_outputs"]()
            ).then(
                fn=update_start_over_state,              # immediately disable Start Over
                inputs=[gallery_components["data"]],
                outputs=[start_over_btn]
            ).then(
                fn=generation_fn,
                inputs=[gallery_components["data"]],
                outputs=[gallery_components["data"]]
            ).then(
                fn=gallery_components["shift_card_ui"],
                inputs=[gallery_components["data"]],
                outputs=gallery_components["get_all_card_outputs"]()
            ).then(
                fn=update_export_section,
                inputs=[gallery_components["data"]],
                outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
            ).then(
                fn=update_start_over_state,
                inputs=[gallery_components["data"]],
                outputs=[start_over_btn]
            ).then(
                fn=lambda data, idx=idx: update_modal_3d_components(data, idx),
                inputs=[gallery_components["data"]],
                outputs=[modal_3d, no_3d_message]
            )
        
        # Wire up edit button events for each card
        for idx, card in enumerate(gallery_components["card_components"]):
            card["edit_btn"].click(
                fn=open_edit_modal,
                inputs=[gr.State(idx), gallery_components["data"]],
                outputs=[edit_modal, edit_current_index, edit_title, edit_description]
            )
        
        # Cancel edit button
        cancel_edit_btn.click(
            fn=lambda: (gr.update(visible=False), None, "", ""),
            outputs=[edit_modal, edit_current_index, edit_title, edit_description]
        )
        
        # Update edit button - two-stage process
        def immediate_disable_buttons_for_edit(edit_idx, new_title, new_description, gallery_data):
            """First stage: immediately disable all buttons when edit update is triggered."""
            if edit_idx is not None and edit_idx < len(gallery_data):
                # Validate the inputs
                if not new_title or not new_title.strip():
                    print(f"Empty title provided for card {edit_idx}")
                    return gallery_data
                
                if not new_description or not new_description.strip():
                    print(f"Empty description provided for card {edit_idx}")
                    return gallery_data
                
                updated_data = gallery_data.copy()
                
                # Mark the specific card as generating
                updated_data[edit_idx]["image_generating"] = True
                print(f"DEBUG: Set image_generating=True for edit card {edit_idx}")
                
                # Disable all buttons globally
                updated_data = disable_all_buttons_for_image_operations(updated_data)
                
                # Return the updated data immediately to show generating state
                return updated_data
            return gallery_data
        
        def perform_edit_update(edit_idx, new_title, new_description, gallery_data):
            """Second stage: perform the actual edit update and image generation."""
            try:
                if edit_idx is not None and edit_idx < len(gallery_data):
                    updated_data = gallery_data.copy()
                    obj = updated_data[edit_idx]
                    old_object_name = obj["title"]
                    
                    # Update the title and description
                    updated_data[edit_idx]["title"] = new_title.strip()
                    updated_data[edit_idx]["description"] = new_description.strip()
                    
                    # Generate a new random seed for the updated prompt
                    import random
                    new_seed = random.randint(1, 999999)
                    
                    print(f"Updating image for '{new_title}' with new prompt and seed {new_seed}")
                    print(f"   New prompt: {new_description}")
                    
                    # Generate new image using SANA service with the updated prompt
                    success, message, new_image_path = image_generation_service.generate_image_from_prompt(
                        object_name=new_title.strip(),
                        prompt=new_description.strip(),
                        output_dir=config.GENERATED_IMAGES_DIR,
                        seed=new_seed
                    )
                    # SANA stays on GPU - will be moved to CPU by GPUMemoryManager
                    # when LLM or TRELLIS needs GPU
                    print(f"Timestamp after generate_image_from_prompt: {time.time()}")

                    invalidate_reason = None
                    
                    if success and new_image_path:
                        # Update the image path and seed
                        updated_data[edit_idx]["path"] = new_image_path
                        updated_data[edit_idx]["seed"] = new_seed
                        
                        # Clear any previous failure flags since image generation succeeded
                        updated_data[edit_idx] = clear_image_generation_failure_flags(updated_data[edit_idx])
                        
                        invalidate_reason = "image update"
                        print(f"Successfully generated new image: {new_image_path}")
                    elif message == "PROMPT_CONTENT_FILTERED":
                        # Handle 2D prompt content filtered case
                        updated_data[edit_idx]["path"] = "static/images/content_filtered.svg"
                        updated_data[edit_idx]["prompt_content_filtered"] = True
                        updated_data[edit_idx]["prompt_content_filtered_timestamp"] = datetime.datetime.now().isoformat()
                        
                        invalidate_reason = "2D prompt content filtered"
                        print(f"2D prompt content filtered for '{new_title}' - using dummy image")
                    else:
                        updated_data[edit_idx]["image_generation_failed"] = True
                        updated_data[edit_idx]["image_generation_error"] = message

                        invalidate_reason = "image generation failed"
                        print(f"Failed to generate new image: {message}")
                    
                    # Clear the image_generating flag since the operation is complete
                    if "image_generating" in updated_data[edit_idx]:
                        del updated_data[edit_idx]["image_generating"]
                    
                    updated_data = invalidate_3d_model(updated_data, edit_idx, new_title.strip(), invalidate_reason)
                    if "batch_processing" in updated_data[edit_idx]:
                        del updated_data[edit_idx]["batch_processing"]
                    
                    # Re-enable all buttons after edit update completes (success or failure)
                    updated_data = enable_all_buttons_after_image_operations(updated_data)
                    
                    return updated_data
                return gallery_data
            except Exception as e:
                print(f"Error in edit update: {str(e)}")
                # Ensure we clear the image_generating flag and re-enable buttons even on exception
                updated_data = gallery_data.copy()
                if edit_idx is not None and edit_idx < len(updated_data):
                    if "image_generating" in updated_data[edit_idx]:
                        del updated_data[edit_idx]["image_generating"]
                updated_data = enable_all_buttons_after_image_operations(updated_data)
                return updated_data
        
        update_edit_btn.click(
            fn=immediate_disable_buttons_for_edit,
            inputs=[edit_current_index, edit_title, edit_description, gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_start_over_state,              # immediately disable Start Over
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        ).then(
            fn=perform_edit_update,
            inputs=[edit_current_index, edit_title, edit_description, gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_export_section,
            inputs=[gallery_components["data"]],
            outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
        ).then(
            fn=update_start_over_state,
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        ).then(
            fn=lambda: (gr.update(visible=False), None, "", ""),
            outputs=[edit_modal, edit_current_index, edit_title, edit_description]
        )
        
        # Wire up delete button events for each card
        for idx, card in enumerate(gallery_components["card_components"]):
            def create_delete_function(card_idx):
                def delete_specific_card(gallery_data):
                    if card_idx < len(gallery_data):
                        print(f"Deleting card at index: {card_idx} title: {gallery_data[card_idx]['title']}")
                        # Check if this card has a 3D asset that will be removed
                        has_3d_asset = gallery_data[card_idx].get("glb_path") and gallery_data[card_idx]["glb_path"]
                        if has_3d_asset:
                            print(f"Removing 3D asset: {gallery_data[card_idx]['glb_path']}")
                        
                        # Remove the card from gallery data
                        updated_data = [item for i, item in enumerate(gallery_data) if i != card_idx]
                        return updated_data
                    else:
                        print(f"Card index {card_idx} out of range")
                        return gallery_data
                return delete_specific_card
            
            card["delete_btn"].click(
                fn=create_delete_function(idx),
                inputs=[gallery_components["data"]],
                outputs=[gallery_components["data"]]
            ).then(
                fn=gallery_components["shift_card_ui"],
                inputs=[gallery_components["data"]],
                outputs=gallery_components["get_all_card_outputs"]()
            ).then(
                fn=update_export_section,
                inputs=[gallery_components["data"]],
                outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
            )
        
        # Wire up convert all to 3D button (two-stage process)
        gallery_components["convert_all_btn"].click(
            fn=disable_buttons_handler,
            inputs=[gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_start_over_state,                  # immediately disable Start Over during batch
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        ).then(
            fn=convert_all_handler,
            inputs=[gallery_components["data"]],
            outputs=[gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        ).then(
            fn=update_export_section,
            inputs=[gallery_components["data"]],
            outputs=[export_components["count_display"], export_components["thumbnails_container"], export_components["export_btn"], export_components["placeholder"], export_components["export_content_active"]]
        ).then(
            fn=update_start_over_state,
            inputs=[gallery_components["data"]],
            outputs=[start_over_btn]
        )
        
        # Wire up export button to open modal
        export_components["export_btn"].click(
            fn=open_export_modal,
            inputs=[gallery_components["data"]],
            outputs=[export_modal]
        )
        
        # Wire up export modal event handlers
        export_cancel_btn.click(
            fn=close_export_modal,
            outputs=[export_modal, scene_folder_input]
        )
        
        # Wire up save button to export assets
        export_save_btn.click(
            fn=export_3d_assets_to_folder,
            inputs=[gallery_components["data"], scene_folder_input],
            outputs=[export_modal]
        ).then(
            fn=close_export_modal,
            outputs=[export_modal, scene_folder_input]
        )
        
    return app


if __name__ == "__main__":
    app = None
    try:
        print("Starting Chat-to-3D application...")
        print("Starting app creation")
        app = create_app()
        print("app created, launching app...")
        app.launch(debug=True, server_name="127.0.0.1", server_port=7860, share=False, quiet=False)
        print("app Launched")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("Cleaning up resources...")
        try:
            # Stop the termination server
            if _termination_server_thread:
                print("Stopping termination server...")
                try:
                    _termination_server_thread.stop()
                    print("Termination server stopped")
                except Exception as e:
                    print(f"Error stopping termination server: {e}")
            
            # Stop the LLM NIM process if it's running
            if _nim_process and _nim_process.poll() is None:
                print("Stopping LLM NIM process...")
                try:
                    _nim_process.terminate()
                    _nim_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("LLM NIM process didn't stop gracefully, forcing...")
                    _nim_process.kill()
                except Exception as e:
                    print(f"Error stopping LLM NIM process: {e}")
            
            # Stop the Trellis NIM process if it's running
            if _trellis_process and _trellis_process.poll() is None:
                print("Stopping Trellis NIM process...")
                try:
                    _trellis_process.terminate()
                    _trellis_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Trellis NIM process didn't stop gracefully, forcing...")
                    _trellis_process.kill()
                except Exception as e:
                    print(f"Error stopping Trellis NIM process: {e}")
            
            # Stop both NIM containers
            print("Stopping LLM NIM container...")
            stop_llm_container(force=True)
            
            print("Stopping Trellis NIM container...")
            stop_trellis_container(force=True)
           
            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("Application shutdown complete") 