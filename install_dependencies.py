#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Automatic dependency installer based on config.py settings.

This script reads the configuration and installs the appropriate dependencies
using the corresponding requirements files:

- requirements.txt           : Core dependencies (always installed via -r)
- requirements-nim.txt       : NIM backend (when USE_NATIVE_LLM = False)
- requirements-native.txt    : Native LLM (when USE_NATIVE_LLM = True)

Usage:
    python install_dependencies.py
"""

import subprocess
import sys
from pathlib import Path


def run_pip_install(requirements_file: str, description: str) -> bool:
    """Run pip install for a requirements file."""
    print(f"\n{'='*60}")
    print(f"Installing {description}...")
    print(f"  Using: {requirements_file}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements_file],
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"✓ {description} installed successfully")
        return True
    else:
        print(f"✗ Failed to install {description}")
        return False


def parse_config(config_path: Path) -> dict:
    """Parse config.py to extract relevant settings."""
    settings = {
        "USE_NATIVE_LLM": True,
    }
    
    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Extract USE_NATIVE_LLM
                if line.startswith("USE_NATIVE_LLM") and "=" in line:
                    value = line.split("=")[1].strip()
                    settings["USE_NATIVE_LLM"] = value.lower() == "true"
                    
    except Exception as e:
        print(f"Warning: Could not parse config.py: {e}")
        print("Using default settings...")
    
    return settings


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    print("=" * 60)
    print("3D Object Generation - Dependency Installer")
    print("=" * 60)
    
    # Parse config.py
    config_path = script_dir / "config.py"
    settings = parse_config(config_path)
    
    use_native_llm = settings["USE_NATIVE_LLM"]
    
    print(f"\nDetected configuration:")
    print(f"  USE_NATIVE_LLM: {use_native_llm}")
    
    # Determine which requirements file to use
    if use_native_llm:
        requirements_file = script_dir / "requirements-native.txt"
        description = "Native LLM dependencies"
    else:
        requirements_file = script_dir / "requirements-nim.txt"
        description = "NIM backend with Griptape"
    
    # Check if the requirements file exists
    if not requirements_file.exists():
        print(f"\n✗ Error: Requirements file not found: {requirements_file}")
        print(f"  Falling back to core requirements.txt")
        requirements_file = script_dir / "requirements.txt"
        description = "Core dependencies"
    
    # Install dependencies
    success = run_pip_install(str(requirements_file), description)
    
    if not success:
        print("\n" + "=" * 60)
        print("✗ Installation failed!")
        print("=" * 60)
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All dependencies installed successfully!")
    print("=" * 60)
    print("\nYou can now run the application with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
