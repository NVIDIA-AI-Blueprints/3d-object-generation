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
- requirements-trellis.txt   : Native TRELLIS (when USE_NATIVE_TRELLIS = True)

Usage:
    python install_dependencies.py
"""

import subprocess
import sys
import os
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


def verify_pytorch_cuda() -> bool:
    """Verify that PyTorch has CUDA support.
    
    Returns True if PyTorch has CUDA, False otherwise.
    """
    print(f"\n{'='*60}")
    print("Verifying PyTorch CUDA support...")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "-c", 
         "import torch; "
         "cuda_ok = '+cu' in torch.__version__; "
         "print(f'  PyTorch version: {torch.__version__}'); "
         "print(f'  CUDA available: {torch.cuda.is_available()}'); "
         "print(f'  CUDA version: {torch.version.cuda}'); "
         "exit(0 if cuda_ok else 1)"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("✓ PyTorch with CUDA verified")
        return True
    else:
        print("✗ PyTorch does NOT have CUDA support!")
        print("  To fix, run:")
        print("    pip uninstall torch torchvision torchaudio -y")
        print("    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128")
        return False


def parse_config(config_path: Path) -> dict:
    """Parse config.py to extract relevant settings."""
    settings = {
        "USE_NATIVE_LLM": True,
        "USE_NATIVE_TRELLIS": True,
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
                    # Remove inline comments (e.g., "True  # comment" -> "True")
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    settings["USE_NATIVE_LLM"] = value.lower() == "true"
                
                # Extract USE_NATIVE_TRELLIS
                if line.startswith("USE_NATIVE_TRELLIS") and "=" in line:
                    value = line.split("=")[1].strip()
                    # Remove inline comments (e.g., "True  # comment" -> "True")
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    settings["USE_NATIVE_TRELLIS"] = value.lower() == "true"
                    
    except Exception as e:
        print(f"Warning: Could not parse config.py: {e}")
        print("Using default settings...")
    
    return settings


def setup_cuda_env() -> bool:
    """Set up CUDA environment variables if not already set.
    
    This blueprint requires CUDA 12.8 on Windows.
    """
    if os.environ.get("CUDA_HOME"):
        print(f"  CUDA_HOME already set: {os.environ['CUDA_HOME']}")
        return True
    
    # This blueprint requires CUDA 12.8 (Windows only)
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    
    if Path(cuda_path).exists():
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["CUDA_PATH"] = cuda_path  # Some tools use CUDA_PATH
        print(f"  Auto-detected CUDA 12.8: {cuda_path}")
        return True
    
    print("  WARNING: CUDA 12.8 not found!")
    print("  This blueprint requires CUDA 12.8. Please either:")
    print("    1. Install CUDA 12.8 from https://developer.nvidia.com/cuda-downloads")
    print("    2. Set CUDA_HOME manually if installed elsewhere:")
    print('       set CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8')
    return False


def install_cuda_extension(name: str, pip_url: str) -> bool:
    """Install a single CUDA extension from source.
    
    Args:
        name: Display name of the extension
        pip_url: The pip install URL (git+https://...)
    
    Returns:
        True if installation succeeded, False otherwise
    """
    print(f"\n  Installing {name}...")
    print(f"    pip install --no-build-isolation {pip_url}")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", pip_url],
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"  ✓ {name} installed successfully")
        return True
    else:
        print(f"  ✗ {name} installation failed!")
        return False


def install_trellis_extensions() -> bool:
    """Install TRELLIS CUDA extensions from source.
    
    These extensions must be built from source to support the user's
    specific CUDA version and GPU architecture.
    
    Extensions:
        - vox2seq: Voxel to sequence conversion
        - nvdiffrast: NVIDIA differentiable rasterizer
        - diffoctreerast: Differentiable octree rasterization
        - diff-gaussian-rasterization: Gaussian splatting renderer
    """
    print(f"\n{'='*60}")
    print("Installing TRELLIS CUDA Extensions")
    print("(Building from source - this may take several minutes)")
    print(f"{'='*60}")
    
    # Ensure CUDA_HOME is set
    if not setup_cuda_env():
        print("\n  ERROR: Cannot install CUDA extensions without CUDA_HOME!")
        return False
    
    # Define extensions to install (with pinned git commits for reproducibility)
    extensions = [
        (
            "vox2seq",
            "git+https://huggingface.co/spaces/dkatz2391/TRELLIS_TextTo3D_Try2@f29eac513da77cf6e2c185f52180bb245df11c8a#subdirectory=extensions/vox2seq"
        ),
        (
            "nvdiffrast",
            "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
        ),
        (
            "diffoctreerast",
            "git+https://github.com/JeffreyXiang/diffoctreerast.git@b09c20b84ec3aace4729e6e18a613112320eca3a"
        ),
        (
            "diff-gaussian-rasterization",
            "git+https://github.com/autonomousvision/mip-splatting.git@dda02ab5ecf45d6edb8c540d9bb65c7e451345a9#subdirectory=submodules/diff-gaussian-rasterization/"
        ),
    ]
    
    all_success = True
    for i, (name, url) in enumerate(extensions, 1):
        print(f"\n[{i}/{len(extensions)}] ", end="")
        if not install_cuda_extension(name, url):
            all_success = False
    
    print(f"\n{'='*60}")
    if all_success:
        print("✓ All CUDA extensions installed successfully!")
    else:
        print("⚠ Some CUDA extensions failed to install.")
        print("  You may need to install them manually.")
    print(f"{'='*60}")
    
    return all_success


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
    use_native_trellis = settings["USE_NATIVE_TRELLIS"]
    
    print(f"\nDetected configuration:")
    print(f"  USE_NATIVE_LLM: {use_native_llm}")
    print(f"  USE_NATIVE_TRELLIS: {use_native_trellis}")
    
    all_success = True
    
    # =========================================================================
    # Step 1: Install LLM dependencies
    # =========================================================================
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
    
    # Install LLM dependencies
    if not run_pip_install(str(requirements_file), description):
        all_success = False
    
    # =========================================================================
    # Step 2: Install TRELLIS dependencies (if enabled)
    # =========================================================================
    if use_native_trellis:
        trellis_requirements = script_dir / "requirements-trellis.txt"
        
        if trellis_requirements.exists():
            if not run_pip_install(str(trellis_requirements), "Native TRELLIS dependencies"):
                all_success = False
        else:
            print(f"\nWarning: requirements-trellis.txt not found")
            print("  Skipping TRELLIS dependencies")
    
    # =========================================================================
    # Step 3: Verify PyTorch has CUDA support
    # =========================================================================
    # requirements-trellis.txt includes --extra-index-url for PyTorch CUDA
    # This check ensures the correct version is installed
    if not verify_pytorch_cuda():
        print("\n⚠ Warning: PyTorch CUDA verification failed!")
        all_success = False
    
    # =========================================================================
    # Step 4: Install TRELLIS CUDA extensions (requires PyTorch with CUDA)
    # =========================================================================
    if use_native_trellis:
        if not install_trellis_extensions():
            all_success = False
    
    # =========================================================================
    # Summary
    # =========================================================================
    if not all_success:
        print("\n" + "=" * 60)
        print("⚠ Installation completed with warnings!")
        print("  Some packages may need manual installation.")
        print("=" * 60)
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All dependencies installed successfully!")
    print("=" * 60)
    print("\nYou can now run the application with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
