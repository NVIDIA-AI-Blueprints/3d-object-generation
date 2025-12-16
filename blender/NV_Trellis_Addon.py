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

bl_info = {
    "name": "3D Object Generation",
    "author": "NVIDIA",
    "version": (1, 0),
    "blender": (4, 2, 0),
    "category": "3D View",
    "description": "Manages the services for 3D asset generation within Blender",
}

import bpy
import os
import subprocess
import threading
import time
import logging
import shutil
import platform
import urllib.request
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty

# Set up logging to file and console
log_file = os.path.join(os.path.expanduser("~"), "trellis_addon.log")
logger = logging.getLogger(__name__)

# Global variables for status
llm_status = "NOT READY"
trellis_status = "NOT READY"
gradio_status = "NOT READY"
starting_services = False
llm_status_lock = threading.Lock()
trellis_status_lock = threading.Lock()
gradio_status_lock = threading.Lock()

# Default conda environment name (can be overridden by config.py)
DEFAULT_CONDA_ENV_NAME = "trellis"

def get_conda_env_name():
    """Read CONDA_ENV_NAME from config.py, with fallback to default."""
    try:
        addon_prefs = bpy.context.preferences.addons[__name__].preferences
        base_path = addon_prefs.base_path.strip()
        if base_path:
            config_path = os.path.join(base_path, "config.py")
            if os.path.isfile(config_path):
                with open(config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("CONDA_ENV_NAME") and "=" in line:
                            value = line.split("=")[1].strip()
                            # Remove quotes and inline comments
                            value = value.strip('"').strip("'")
                            if "#" in value:
                                value = value.split("#")[0].strip()
                            if value:
                                return value
    except Exception as e:
        logger.debug(f"Could not read CONDA_ENV_NAME from config.py: {e}")
    return DEFAULT_CONDA_ENV_NAME
starting_services_lock = threading.Lock()
stop_thread = False
log_output_stop = threading.Event()

# Console handler for dynamic level adjustment
console_handler = logging.StreamHandler()

def update_logging_level():
    """Update the console handler's logging level based on user preference."""
    addon_prefs = bpy.context.preferences.addons[__name__].preferences
    log_level_str = addon_prefs.console_log_level
    log_level = {
        "ERROR": logging.ERROR,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }.get(log_level_str, logging.DEBUG)
    console_handler.setLevel(log_level)
    logger.debug(f"Console logging level updated to: {log_level_str}")

def setup_logging():
    """Set up the logger with file and console handlers."""
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    
    console_handler.setFormatter(log_format)
    
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    update_logging_level()

def get_conda_python_path():
    """Attempt to find the Conda environment's Python executable."""
    addon_prefs = bpy.context.preferences.addons[__name__].preferences
    user_python_path = addon_prefs.python_path.strip()
    env_name = get_conda_env_name()

    # Step 1: Check CONDA_PREFIX
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and os.path.isdir(conda_prefix):
        if os.path.basename(conda_prefix) == env_name and os.path.basename(os.path.dirname(conda_prefix)) == "envs":
            conda_base = os.path.dirname(os.path.dirname(conda_prefix))
        else:
            conda_base = conda_prefix
        if platform.system() == "Windows":
            python_path = os.path.normpath(os.path.join(conda_base, "envs", env_name, "python.exe"))
        else:
            python_path = os.path.join(conda_base, "envs", env_name, "bin", "python")
        if os.path.isfile(python_path):
            try:
                result = subprocess.run(
                    [python_path, "-c", "import sys; print(sys.version)"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info("Using Python path from CONDA_PREFIX: %s", python_path)
                    return python_path
            except Exception as e:
                logger.debug("Failed to verify Python path from CONDA_PREFIX: %s", str(e))

    # Step 2: Check ~/.conda/environments.txt
    env_file = os.path.join(os.path.expanduser("~"), ".conda", "environments.txt")
    if os.path.isfile(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    env_path = line.strip()
                    if env_path and os.path.basename(env_path) == env_name and os.path.basename(os.path.dirname(env_path)) == "envs":
                        logger.debug("Found %s environment in environments.txt: %s", env_name, env_path)
                        if platform.system() == "Windows":
                            python_path = os.path.normpath(os.path.join(env_path, "python.exe"))
                        else:
                            python_path = os.path.join(env_path, "bin", "python")
                        if os.path.isfile(python_path):
                            try:
                                result = subprocess.run(
                                    [python_path, "-c", "import sys; print(sys.version)"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if result.returncode == 0:
                                    logger.info("Found Python executable from environments.txt: %s", python_path)
                                    return python_path
                            except Exception as e:
                                logger.debug("Failed to verify Python from environments.txt: %s", str(e))
        except Exception as e:
            logger.warning("Failed to read environments.txt: %s", str(e))

    # Step 3: Check user-supplied python_path
    if user_python_path and os.path.isfile(user_python_path):
        try:
            result = subprocess.run(
                [user_python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("Using user-provided Python path: %s", user_python_path)
                return user_python_path
            else:
                logger.warning("User-provided Python path is invalid: %s", user_python_path)
        except Exception as e:
            logger.warning("Failed to verify user-provided Python path: %s", str(e))

    # Step 4: Check 'conda info --base'
    conda_exe = shutil.which("conda")
    if not conda_exe:
        default_conda_base = os.path.expanduser("~/Miniconda3")
        conda_exe = os.path.join(default_conda_base, "Scripts", "conda.exe") if platform.system() == "Windows" else os.path.join(default_conda_base, "bin", "conda")
        if not os.path.isfile(conda_exe):
            conda_exe = None
    if conda_exe:
        try:
            result = subprocess.run(
                [conda_exe, "info", "--base"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                conda_base = result.stdout.strip()
                if platform.system() == "Windows":
                    python_path = os.path.normpath(os.path.join(conda_base, "envs", env_name, "python.exe"))
                else:
                    python_path = os.path.join(conda_base, "envs", env_name, "bin", "python")
                if os.path.isfile(python_path):
                    try:
                        result = subprocess.run(
                            [python_path, "-c", "import sys; print(sys.version)"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            logger.info("Using Python path from 'conda info --base': %s", python_path)
                            return python_path
                    except Exception as e:
                        logger.debug("Failed to verify Python path from conda info: %s", str(e))
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug("Failed to locate Conda base using 'conda info --base': %s", str(e))

    # Step 5: Fallback to default
    default_conda_base = os.path.expanduser("~/Miniconda3")
    if platform.system() == "Windows":
        python_path = os.path.normpath(os.path.join(default_conda_base, "envs", env_name, "python.exe"))
    else:
        python_path = os.path.join(default_conda_base, "envs", env_name, "bin", "python")
    if os.path.isfile(python_path):
        try:
            result = subprocess.run(
                [python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("Using default fallback Python path: %s", python_path)
                return python_path
        except Exception as e:
            logger.debug("Failed to verify default Python path: %s", str(e))

    logger.error("Conda Python not found. Ensure the '%s' environment is set up correctly.", env_name)
    return None

def get_services_status(python_path):
    """Run check_services.py to get LLM and Trellis status."""
    try:
        script_path = os.path.join(bpy.context.preferences.addons[__name__].preferences.base_path, "check_services.py")
        logger.debug(f"Running check_services.py at: {script_path} with python: {python_path}")
        result = subprocess.run(
            [python_path, script_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        logger.debug(f"check_services.py return code: {result.returncode}")
        logger.debug(f"check_services.py stdout: {repr(result.stdout)}")
        logger.debug(f"check_services.py stderr: {repr(result.stderr)}")
        output = result.stdout.strip()
        if output.endswith("ALL_READY"):
            return "READY", "READY"
        elif output.endswith("LLM_READY"):
            return "READY", "NOT READY"
        elif output.endswith("TRELLIS_READY"):
            return "NOT READY", "READY"
        else:
            return "NOT READY", "NOT READY"
    except Exception as e:
        logger.debug(f"Failed to check services: {str(e)}")
        return "NOT READY", "NOT READY"

def check_gradio_service():
    """Check Gradio service status using urllib."""
    try:
        req = urllib.request.Request('http://127.0.0.1:7860/', method='HEAD')
        with urllib.request.urlopen(req, timeout=5) as response:
            logger.debug(f"Gradio HTTP response code: {response.code}")
            if response.code == 200 or response.code == 302:
                return "READY"
        return "NOT READY"
    except Exception as e:
        logger.debug(f"Failed to check Gradio: {str(e)}")
        return "NOT READY"

def check_services_status():
    """Thread function to periodically check all services."""
    global llm_status, trellis_status, gradio_status
    python_path = get_conda_python_path()
    while not stop_thread:
        llm, trellis = get_services_status(python_path)
        gradio = check_gradio_service()
        with llm_status_lock:
            llm_status = llm
        with trellis_status_lock:
            trellis_status = trellis
        with gradio_status_lock:
            gradio_status = gradio
        time.sleep(10)

def start_status_threads():
    """Start the status checking thread."""
    global check_thread
    check_thread = threading.Thread(target=check_services_status, daemon=True)
    check_thread.start()

def stop_status_threads():
    """Stop the status checking thread."""
    global stop_thread
    stop_thread = True
    if 'check_thread' in globals() and check_thread.is_alive():
        check_thread.join(timeout=5)

class TrellisAddonPreferences(AddonPreferences):
    bl_idname = __name__

    base_path: StringProperty(
        name="Blueprint Base Path",
        description="Base path for 3D object generation project (e.g., C:\\path\\to\\chat-to-3d)",
        default=os.environ.get("CHAT_TO_3D_PATH", ""),
        subtype='DIR_PATH'
    )

    python_path: StringProperty(
        name="Conda Python Path",
        subtype='FILE_PATH',
        description="Path to the Python executable in the trellis Conda environment"
    )

    console_log_level: EnumProperty(
        name="Console Log Level",
        items=[
            ("ERROR", "Error", "Show only errors"),
            ("INFO", "Info", "Show info and errors"),
            ("DEBUG", "Debug", "Show all messages")
        ],
        default="DEBUG",
        update=lambda self, context: update_logging_level()
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "base_path")
        layout.prop(self, "python_path")
        layout.prop(self, "console_log_level")

class TrellisManager:
    def __init__(self):
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None

    def start_services(self, python_path, base_path):
        """Start all services using app.py."""
        try:
            script_path = os.path.join(base_path, "app.py")
            self.process = subprocess.Popen(
                [python_path, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=base_path
            )
            logger.info("Services started with PID %d via app.py", self.process.pid)

            def log_output(pipe, log_func):
                while not log_output_stop.is_set():
                    try:
                        line = pipe.readline()
                        if not line:
                            break
                        if "INFO" in line:
                            logger.info(line.strip())
                        elif "DEBUG" in line:
                            logger.debug(line.strip())
                        elif "WARNING" in line:
                            logger.warning(line.strip())
                        else:
                            log_func(line.strip())
                    except ValueError as e:
                        logger.debug(f"Pipe closed while logging: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error while logging output: {e}")
                        break

            self.stdout_thread = threading.Thread(
                target=log_output,
                args=(self.process.stdout, lambda x: logger.info("app.py stdout: %s", x)),
                daemon=True
            )
            self.stderr_thread = threading.Thread(
                target=log_output,
                args=(self.process.stderr, lambda x: logger.error("app.py stderr: %s", x)),
                daemon=True
            )
            self.stdout_thread.start()
            self.stderr_thread.start()

            return True
        except Exception as e:
            logger.error(f"Failed to start services via app.py: {str(e)}")
            return False

    def stop_services(self):
        """Stop all services and clean up resources."""
        if self.process:
            log_output_stop.set()
            if self.stdout_thread:
                self.stdout_thread.join(timeout=5)
            if self.stderr_thread:
                self.stderr_thread.join(timeout=5)
            try:
                self.process.stdout.close()
                self.process.stderr.close()
            except Exception as e:
                logger.debug(f"Error closing pipes: {e}")
            self.process = None
            self.stdout_thread = None
            self.stderr_thread = None
            log_output_stop.clear()

# Global variable to store service start result
service_start_result = None

def check_service_start_result():
    """Timer function to check the result of service startup and report it."""
    global service_start_result
    if service_start_result is not None:
        # Trigger UI redraw
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        # Report result using a new operator
        bpy.ops.trellis.report_service_status('INVOKE_DEFAULT', success=service_start_result)
        service_start_result = None
        return None  # Stop the timer
    return 0.1  # Continue checking

class TRELLIS_OT_ReportServiceStatus(Operator):
    bl_idname = "trellis.report_service_status"
    bl_label = "Report Service Status"
    
    success: bpy.props.BoolProperty()

    def execute(self, context):
        if self.success:
            self.report({'INFO'}, "Services started, waiting for readiness...")
        else:
            self.report({'ERROR'}, "Failed to start services via app.py")
        return {'FINISHED'}

class TRELLIS_OT_ManageTrellis(Operator):
    bl_idname = "trellis.manage_trellis"
    bl_label = "Manage TRELLIS"

    def check_services_ready(self, timeout=300):
        """Check if all services are ready, with timeout in seconds."""
        global starting_services
        start_time = time.time()
        last_log_time = start_time
        while time.time() - start_time < timeout:
            with llm_status_lock, trellis_status_lock, gradio_status_lock:
                if llm_status == "READY" and trellis_status == "READY" and gradio_status == "READY":
                    logger.info("All services are ready")
                    return True
                current_time = time.time()
                if current_time - last_log_time >= 5:
                    logger.info("Services not ready after %d seconds: LLM=%s, Trellis=%s, Gradio=%s",
                                int(current_time - start_time), llm_status, trellis_status, gradio_status)
                    last_log_time = current_time
            time.sleep(1)
        logger.error("Services failed to start within %d seconds", timeout)
        return False

    def start_services_thread(self, python_path, base_path):
        """Start services in a separate thread and store the result."""
        global starting_services, service_start_result
        manager = TrellisManager()
        success = manager.start_services(python_path, base_path)
        with starting_services_lock:
            starting_services = False
        # Store the result in a global variable
        service_start_result = success

    def execute(self, context):
        global trellis_status, llm_status, gradio_status, starting_services
        with trellis_status_lock, llm_status_lock, gradio_status_lock, starting_services_lock:
            overall_status = "READY" if trellis_status == "READY" and llm_status == "READY" and gradio_status == "READY" else "NOT READY"
            logger.info(f"Current Status is: {overall_status}")

        base_path = bpy.context.preferences.addons[__name__].preferences.base_path
        if not base_path or not os.path.isdir(base_path):
            self.report({'ERROR'}, "Invalid or missing Trellis Base Path")
            return {'CANCELLED'}

        app_script = os.path.join(base_path, "app.py")
        if not os.path.isfile(app_script):
            self.report({'ERROR'}, f"app.py script not found: {app_script}")
            return {'CANCELLED'}

        python_path = get_conda_python_path()
        if not python_path:
            self.report({'ERROR'}, "Conda Python executable not found. Set Conda Base Path in addon preferences or ensure conda is in PATH.")
            return {'CANCELLED'}

        if overall_status == "READY":
            # Stop services
            try:
                subprocess.run(["stop_services.bat"], shell=True, check=True, cwd=base_path)
                logger.info("Services stopped via stop_services.bat")
                # Wait briefly for services to become NOT READY
                start_time = time.time()
                timeout = 30
                while time.time() - start_time < timeout:
                    with llm_status_lock, trellis_status_lock, gradio_status_lock:
                        if llm_status == "NOT READY" and trellis_status == "NOT READY" and gradio_status == "NOT READY":
                            break
                    time.sleep(1)
                TrellisManager().stop_services()
                self.report({'INFO'}, "All services terminated successfully")
                with trellis_status_lock:
                    trellis_status = "NOT READY"
                with llm_status_lock:
                    llm_status = "NOT READY"
                with gradio_status_lock:
                    gradio_status = "NOT READY"
            except Exception as e:
                logger.error(f"Failed to stop services via stop_services.bat: {str(e)}")
                self.report({'ERROR'}, "Failed to stop services")
                return {'CANCELLED'}
        else:
            # Check if services are already running
            llm, trellis = get_services_status(python_path)
            gradio = check_gradio_service()
            if llm == "READY" and trellis == "READY" and gradio == "READY":
                with llm_status_lock:
                    llm_status = "READY"
                with trellis_status_lock:
                    trellis_status = "READY"
                with gradio_status_lock:
                    gradio_status = "READY"
                logger.info("All services are already ready, skipping startup")
                self.report({'INFO'}, "All services are already running")
            else:
                # Stop any existing partial services
                try:
                    subprocess.run(["stop_services.bat"], shell=True, check=True, cwd=base_path)
                    logger.info("Existing partial services stopped via stop_services.bat")
                except Exception as e:
                    logger.warning(f"Failed to stop existing services: {str(e)}")
                    self.report({'WARNING'}, "Failed to stop existing services, continuing with startup")

                # Set starting state
                with starting_services_lock:
                    starting_services = True

                # Start services in a separate thread
                threading.Thread(
                    target=self.start_services_thread,
                    args=(python_path, base_path),
                    daemon=True
                ).start()

                # Register a timer to check the result
                bpy.app.timers.register(
                    check_service_start_result,
                    first_interval=0.1,
                    persistent=True
                )

                self.report({'INFO'}, "Services are starting... Check status below.")

        return {'FINISHED'}

class VIEW3D_PT_CHAT_TO_3D(Panel):
    bl_label = "3D Object Generation"
    bl_idname = "VIEW3D_PT_chat_to_3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '3D Object Generation'
    
    def draw(self, context):
        layout = self.layout
        addon_prefs = context.preferences.addons[__name__].preferences
        
        layout.label(text="Base Path is set in Add-on Preferences")
        
        with trellis_status_lock, llm_status_lock, gradio_status_lock, starting_services_lock:
            overall_status = "READY" if trellis_status == "READY" and llm_status == "READY" and gradio_status == "READY" else "NOT READY"
            button_text = "Starting Services" if starting_services else ("Services Started .. Click to Terminate" if overall_status == "READY" else "Start Services")
            button_enabled = not starting_services
        
        row = layout.row()
        row.operator(
            TRELLIS_OT_ManageTrellis.bl_idname,
            text=button_text
        )
        row.enabled = button_enabled
        
        with llm_status_lock:
            llm_stat = llm_status
        layout.label(
            text=f"LLM Service Status: {llm_stat}",
            icon='CHECKMARK' if llm_stat == "READY" else 'ERROR'
        )
        
        with trellis_status_lock:
            trellis_stat = trellis_status
        layout.label(
            text=f"Trellis Server Status: {trellis_stat}",
            icon='CHECKMARK' if trellis_stat == "READY" else 'ERROR'
        )
        
        with gradio_status_lock:
            gradio_stat = gradio_status
        layout.label(
            text=f"Gradio Web UI Status: {gradio_stat}",
            icon='CHECKMARK' if gradio_stat == "READY" else 'ERROR'
        )
        
        row = layout.row(align=True)
        button_row = row.row(align=True)
        button_row.operator(
            "wm.url_open",
            text="Open 3D Object Generation UI",
            icon='URL'
        ).url = "http://127.0.0.1:7860/?__theme=light"
        button_row.enabled = (gradio_stat == "READY")

def update_status_ui():
    """Timer function to refresh the UI with the latest statuses."""
    global starting_services
    with llm_status_lock, trellis_status_lock, gradio_status_lock, starting_services_lock:
        if llm_status == "READY" and trellis_status == "READY" and gradio_status == "READY" and starting_services:
            logger.info("All services are ready, resetting starting_services")
            starting_services = False
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
    return 1.0

classes = (
    TrellisAddonPreferences,
    TRELLIS_OT_ManageTrellis,
    TRELLIS_OT_ReportServiceStatus,
    VIEW3D_PT_CHAT_TO_3D,
)

def register():
    print("3D Object Generation Manager Add-on Loaded - Version 1.0 - June 16, 2025")
    if bpy.app.version < (4, 2, 0):
        print("Warning: This add-on requires Blender 4.2.0 or higher")
        return
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Set up logging
    setup_logging()
    
    # Attempt to find and store Python path if not set
    addon_prefs = bpy.context.preferences.addons[__name__].preferences
    if not addon_prefs.python_path.strip():
        python_path = get_conda_python_path()
        if python_path:
            addon_prefs.python_path = python_path
            logger.info(f"Automatically set Python path to: {python_path}")
        else:
            logger.warning(f"Could not find Conda Python executable for {get_conda_env_name()} environment")
    
    # Start status threads and UI timer
    start_status_threads()
    try:
        bpy.app.timers.register(update_status_ui, persistent=True)
    except TypeError as e:
        logger.error(f"Failed to register timer: {str(e)}")
        raise

def unregister():
    global log_output_stop
    log_output_stop.set()
    
    base_path = bpy.context.preferences.addons[__name__].preferences.base_path
    try:
        subprocess.run(["stop_services.bat"], shell=True, check=True, cwd=base_path)
        logger.info("Services stopped successfully during unregister")
    except Exception as e:
        logger.warning(f"Failed to stop services during unregister: {str(e)}")

    TrellisManager().stop_services()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    stop_status_threads()
    if bpy.app.timers.is_registered(update_status_ui):
        bpy.app.timers.unregister(update_status_ui)

if __name__ == "__main__":
    register()
