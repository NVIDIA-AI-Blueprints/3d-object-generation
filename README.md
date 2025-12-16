# 3D Object Generation Blueprint

## Description

The 3D Object Generation Blueprint is an end-to-end generative AI workflow that allows users to prototype 3D scenes quickly by simply describing the scene. The Blueprint takes a user's 3D scene idea, generates object recommendations, associated prompts and previews using a Llama 3.1 8B LLM and NVIDIA SANA, and ready-to-use 3D objects with Microsoft TRELLIS.  

> This blueprint supports the following NVIDIA GPUs: RTX 5090, RTX 5080, RTX 4090, RTX 4080, RTX 6000 Ada. We're planning to add wider GPU support in the near future. We recommend at least 48 GB of system RAM. 

## Features

- Chat interface for scene planning
- AI-assisted object and prompt generation
- Automatic 3D asset generation from text prompts
- Blender import functionality for generated assets
- **Two backend modes**: Native PyTorch or NIM containers
- **GPU memory management** - Intelligent model loading/unloading
- VRAM management with model termination

## Backend Options

This blueprint supports two backend configurations:

| Feature | Native Mode | NIM Mode |
|---------|-------------|----------|
| **LLM** | Qwen3-4B or Llama 3.1 8B (local) | Llama 3.1 8B (NIM container) |
| **TRELLIS** | Native PyTorch | NIM container |
| **Requirements** | CUDA 12.8 | NIM containers, Docker |
| **Setup** | Simpler | More complex |

Configure in `config.py`:
```python
USE_NATIVE_LLM = True      # True = Native model, False = NIM
USE_NATIVE_TRELLIS = True  # True = Native TRELLIS, False = NIM
```

## Installation 

### Prerequisites

The NIM Prerequisite Installer requires Microsoft User Account Control (UAC) to be enabled. UAC is enabled by default for Windows, but if it has been disabled, it must be enabled to ensure successful installation of the NIM Prerequisite Installer. More information on Microsoft UAC can found [HERE](https://support.microsoft.com/en-us/windows/user-account-control-settings-d5b2046b-dcb8-54eb-f732-059f321afe18)

Use winget to install Miniconda:
```
winget install miniconda3
```

Download the [NIM Prerequisite Installer](https://assets.ngc.nvidia.com/products/api-catalog/rtx/NIMSetup.exe), and run the NIMSetup.exe file, and follow the instructions in the setup dialogs. This will install the necessary system components to work with NVIDIA NIMs on your system.

You will need to reboot your computer to complete the installation.

Git is required and should be installed using winget from a command prompt:
```
winget install --id Git.Git 
```

Install Git LFS (large file system) support:
```
winget install --id=GitHub.GitLFS -e
```

This blueprint requires the installation of Blender. The blueprint has been tested with the Blender 4.27 LTS (Long Term Support) build.   
[https://www.blender.org/download/release/Blender4.2/blender-4.2.7-windows-x64.msi](https://www.blender.org/download/release/Blender4.2/blender-4.2.7-windows-x64.msi)

Blender 4.27 can also be installed using winget from a command prompt:
```
winget install --id 9NW1B444LDLW
```

**For Native Mode (Additional):**
- CUDA Toolkit 12.8 installed from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- ~50GB disk space for models

### Initialize Conda

1. From the Windows Start button, open a Anaconda (Miniconda) Prompt
2. <img width="827" height="78" alt="image" src="https://github.com/user-attachments/assets/0cd7a72f-54ba-44b1-91b5-f58628665568" />
3. You should see the command prompt with prompt prefixed with the (base) conda environment name
4. Initialize Conda for your system:
```
conda init
```
5. <img width="1159" height="863" alt="image" src="https://github.com/user-attachments/assets/12cd9d81-1803-488f-83b4-d556fb72bd3c" />
6. Close the command prompt

### Installation Steps

1. Open a new command prompt
2. Clone this repository with submodules:
```bash
git clone --recurse-submodules https://github.com/NVIDIA-AI-Blueprints/3d-object-generation.git
```

3. Run the installer script:
```bash
cd 3d-object-generation
.\install.bat
```

The installation process will:
- Create a Conda virtual environment
- Install all required dependencies
- Build CUDA extensions for TRELLIS (if native mode)
- Download required models
- Set up necessary configurations

After successful installation, you'll see:
```
Installation completed successfully
Press any key to continue . . .
```

---

## Native Mode Configuration

When using native models (`USE_NATIVE_LLM = True` and/or `USE_NATIVE_TRELLIS = True`), you can customize the following settings in `config.py`:

### LLM Model Options

```python
# Model name from HuggingFace
# Qwen3-4B: "Qwen/Qwen3-4B" (4B params)
# Llama-3.1-8B: "meta-llama/Llama-3.1-8B-Instruct" (8B params)
NATIVE_LLM_MODEL = "Qwen/Qwen3-4B"

# Precision options: "float16", "bfloat16", "float32", "int4" (for GPTQ)
NATIVE_LLM_PRECISION = "bfloat16"
```

### GPU Memory Management

The application automatically manages GPU memory across three models:
- **LLM** (Qwen3-4B or Llama 3.1 8B)
- **SANA** (Image generation)
- **TRELLIS** (3D generation)

```python
# NATIVE_VRAM_RESTRICTED_MODE: Restrict VRAM usage
#   - True:  Aggressive memory management for GPUs with limited VRAM
#   - False: Use all available VRAM
NATIVE_VRAM_RESTRICTED_MODE = False

# NATIVE_RAM_RESTRICTED_MODE: Unload models instead of moving to CPU
#   - True:  Completely unload models when not needed (saves system RAM, slower)
#   - False: Move models to CPU when not needed (uses system RAM, faster switching)
NATIVE_RAM_RESTRICTED_MODE = False
```

| Setting | Behavior |
|---------|----------|
| Both `False` (default) | Pre-load all models at startup, move to CPU when idle |
| `VRAM_RESTRICTED = True` | Aggressive memory management for limited VRAM |
| `RAM_RESTRICTED = True` | Load on-demand, unload when idle (saves system RAM) |
| Both `True` | Most memory efficient, slowest switching |

---

## Usage - Manual Steps

### Starting the Application

1. In a new terminal window, start the main application:
```
conda activate trellis
cd 3d-object-generation

python app.py
```

4. Open your browser to the URL shown in the terminal (typically http://127.0.0.1:7860/)

**💡 Recommended**: For the best experience, use the light theme by accessing the application with: `http://127.0.0.1:7860/?__theme=light`

**⚠️ Important Note**: Browser refresh is not supported and may cause the application to crash. In that case, please restart the application instead.

### Managing the Application

To terminate the application and free VRAM:
Open a command prompt at the 3d-object-generation install directory
```
conda activate trellis
cd 3d-object-generation
python terminator.py
```

This will:
- Gracefully terminate the Gradio application
- Free up GPU memory
- Allow you to proceed with other operations which may require VRAM (e.g., Blender)

## Usage - Blender Add On

The **3D Object Generation Blender** add-on can automatically manage required services without the need to manually start or stop these services outside of Blender.

### Initial Setup Step

1. Open Blender
2. Open Edit >> Preferences >> Add-Ons
3. <img width="996" height="384" alt="image" src="https://github.com/user-attachments/assets/a858877d-d182-44f2-bcc9-72f47358070c" />
4. Enable 3D Object Generation and Asset Importer by checking the boxes next to the add on names.
5. Open the 3D Object Generation add on preferences and set the Blueprint Base folder to the 3d-object-generation local repository directory (This setting should be automatically set, but may be manually set if desired).
6. <img width="983" height="537" alt="image" src="https://github.com/user-attachments/assets/55cb9cc8-3493-4b19-a139-14572e254c9a" />

### Normal Usage

7. In the 3D layout view look for the Add On tabs on the right edge of the viewport, press N if they are not visible
8. <img width="590" height="476" alt="image" src="https://github.com/user-attachments/assets/51a7b2dd-be42-44f4-8572-d35a3c3967ad" />
9. Note: It is recommended to open a system console viewer to monitor the services and any information or errors that may be output.
   a. Blender Menu >> Window >> Toggle System Console
10. Click the Start Services button to start the LLM agent, and the Trellis 3D services. (It may take up to 3 minutes for all services to fully load and start)
11. Once all services have successfully started, the service will indicate: READY and the **Open 3D Object Generation UI** button will become available
12. <img width="547" height="625" alt="image" src="https://github.com/user-attachments/assets/e4a62fd1-8948-4750-bba6-59438febc7a0" />
13. Click the **Open 3D Object Generation UI** button to launch the 3D Object Generation interface.
14. Clicking the **Services Started .. Click to Terminate** button will shut down the LLM and Trellis services and release system resources being utilized. 
    
### Using the Interface

Once the application is running, you can:

1. **Scene Planning**:
   - Describe your desired scene in natural language
   <kbd>
   <img width="1508" height="999" alt="image" src="https://github.com/user-attachments/assets/a65600ee-3507-4bb1-86d9-66d734134f63" />
   </kbd>

2. **Asset Generation**:
   - The LLM will automatically create prompts for suggested items which will be sent to the 2D image generator
     <kbd>
     <img width="1606" height="1352" alt="image" src="https://github.com/user-attachments/assets/55b11095-1b0a-42c5-95e6-7d32f56ed72f" />
     </kbd>
     
   - Each image contains additional controls
     
     <kbd>
     <img width="640" height="370" alt="image" src="https://github.com/user-attachments/assets/6a03b8ab-ee65-40ee-a458-ef64475d7a50" />
     </kbd>
     
     - <img width="86" height="74" alt="image" src="https://github.com/user-attachments/assets/4dd79a46-bb97-46f9-bcba-905606c168bf" /> **Refresh** - Generate a new image based on the existing prompt.
     - <img width="87" height="69" alt="image" src="https://github.com/user-attachments/assets/577d88ee-fc54-47f2-be77-922d0df2fba0" /> **Edit** - Edit the prompt an generate a new image.
     - <img width="77" height="65" alt="image" src="https://github.com/user-attachments/assets/7e740931-29e0-4ff4-af0b-cc080e130c2a" /> **Delete** - Remove the image from the gallery display.
     - <img width="114" height="63" alt="image" src="https://github.com/user-attachments/assets/e125e259-bf10-4e28-97f1-b416c08a168e" /> **Generate 3D** - Generate a 3D object from the image.
        - The color of the Generate 3D button indicates the status of 3D generation for that object
        -  <img width="122" height="72" alt="Screenshot 2025-08-22 141820" src="https://github.com/user-attachments/assets/f35824da-8311-4af6-ada0-a34c2795d8be" /> Object has not been queued for 3D generation.
        -  <img width="130" height="68" alt="Screenshot 2025-08-22 141836" src="https://github.com/user-attachments/assets/b58d7341-68dd-4bdb-ae75-281f4491327d" /> Object has been queued for 3D generation, but object generation has not completed.
        -  <img width="117" height="64" alt="Screenshot 2025-08-22 141810" src="https://github.com/user-attachments/assets/fd1119a5-b51b-4013-9c16-6cca27ed7834" /> 3D model has been generated for this object.
        -  <img width="120" height="66" alt="Screenshot 2025-08-22 141708" src="https://github.com/user-attachments/assets/66cd9956-d622-4059-b132-2470eb2ba42f" /> Object has been flagged by guardrails as potentially inappropriate, 3D object will not be generated.

<img width="2313" height="125" alt="image" src="https://github.com/user-attachments/assets/8ca38e36-c245-4e06-93e9-5bec518025c9" />

   - Convert all images to 3D Objects (Delete unwanted images before converting to 3D)
     
   - **NOTE**: Image to 3D Object processing takes up to 45 seconds *per object* on a RTX 5090, when using the Convert All image option this time will be a multiple of the number of objects being converted, using the Convert All option may take a significant amount of time. The UI will not be updated until all objects have been converted. 

3. **Save Objects**:
   - The Export Objects to File allows saving the generated objects to a folder.
   <kbd>
   <img width="2384" height="613" alt="image" src="https://github.com/user-attachments/assets/21a5b43f-7dda-42d9-bbf3-40518a3d3754" />
   </kbd>
   <img width="50%" height="50%" alt="image" src="https://github.com/user-attachments/assets/b302f89a-e282-4a22-ba8a-607cc2a40c82" />

4. **Blender Integration**:
   - Import generated assets directly into Blender
   - <img width="455" height="222" alt="image" src="https://github.com/user-attachments/assets/11b4f471-3fb1-4980-bba6-338886219202" />
   - Use the Asset Importer add-on and select the desired scene folder, and click Import assets
   - <img width="498" height="133" alt="image" src="https://github.com/user-attachments/assets/da88971b-ce42-454c-b3db-3ec8f32d0f68" />
   - Assets are imported and the asset tag is applied, saving the scene to the %userprofile%\Documents\Blender\assets folder will add the imported objects to the Blender asset browser.
   - <img width="1933" height="1234" alt="image" src="https://github.com/user-attachments/assets/f148936c-27da-428c-9d92-5603446deb37" />
   - Continue working with the assets in your 3D workflow
   - Can be used with [3D Guided Gen AI BP](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx)

---

## Configuration Reference

### `config.py` Key Settings

```python
# =============================================================================
# Backend Selection
# =============================================================================
USE_NATIVE_LLM = True           # True = Native PyTorch, False = NIM
USE_NATIVE_TRELLIS = True       # True = Native PyTorch, False = NIM

# =============================================================================
# Native LLM Settings (when USE_NATIVE_LLM = True)
# =============================================================================
NATIVE_LLM_MODEL = "Qwen/Qwen3-4B"    # HuggingFace model ID
NATIVE_LLM_PRECISION = "bfloat16"      # float16, bfloat16, or int4 (for GPTQ)

# =============================================================================
# NIM Settings (when USE_NATIVE_LLM = False)
# =============================================================================
AGENT_MODEL = "meta/llama-3.1-8b-instruct"
AGENT_BASE_URL = "http://localhost:19002/v1"
TRELLIS_BASE_URL = "http://localhost:8000/v1"

# =============================================================================
# Memory Management (Native Mode)
# =============================================================================
NATIVE_VRAM_RESTRICTED_MODE = False    # Restrict to 16GB VRAM
NATIVE_RAM_RESTRICTED_MODE = False     # Unload models vs move to CPU

# =============================================================================
# Logging
# =============================================================================
VERBOSE = False                        # Detailed timing/memory logs
```

### Requirements Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Core dependencies |
| `requirements-torch.txt` | PyTorch with CUDA 12.8 |
| `requirements-native.txt` | Native LLM support (GPTQ) |
| `requirements-trellis.txt` | TRELLIS dependencies |
| `requirements-nim.txt` | NIM/Griptape dependencies |
| `requirements-freeze.txt` | Exact versions for reproducibility |

---

<details><summary><h2>Manual Installation Guide for Trellis Project</h2></summary>

This guide provides a step-by-step manual for setting up the Trellis environment on a Windows system with Conda, as a manual alternative to running the operations via the `install.bat` script. It covers checking prerequisites, creating a Conda environment, installing dependencies, setting environment variables, downloading models, installing Blender addons, and verifying services by starting, checking, and stopping them.

**Prerequisites:**
- Conda must be installed and added to your system's PATH.
- Required files in the current directory: `requirements-torch.txt`, `requirements.txt`, `set_environment_variable.py`, `download_models.py`, `check_services.py`.
- Subdirectories: `blender` (containing `NV_Trellis_Addon.py` and `asset_importer.py`), `nim_llm` (containing `run_llama.py` and `manager.py`), `nim_trellis` (containing `run_trellis.py` and `manager.py`).
- Blender installed (versions 4.2 to 5.0 supported; defaults to 4.2 if no version detected).

Run commands in a Command Prompt with administrative privileges where possible. Replace placeholders (e.g., paths) as needed for your setup.

## Step 1: Check for Conda Installation
1. Open Command Prompt.
2. Run: `where conda`.
3. If Conda is not found (errorlevel ≠ 0), install Conda from the official website (e.g., Miniconda or Anaconda) and add it to PATH. Exit if not installed.

## Step 2: Verify Requirements Files
1. Ensure `requirements-torch.txt` exists in the current directory. If not, obtain it and place it there.
2. Ensure `requirements.txt` exists in the current directory. If not, obtain it and place it there.
3. If either file is missing, stop the process and resolve the issue.

## Step 3: Locate Conda Installation Path
1. Check common Conda installation paths:
   - `%USERPROFILE%\miniconda3`
   - `%USERPROFILE%\AppData\Local\miniconda3`
   - `%USERPROFILE%\anaconda3`
2. Verify the presence of `Scripts\conda.exe` or `condabin\conda.bat` in one of these paths.
3. If not found, verify your Conda installation and adjust paths manually. Set `CONDA_PATH` to the valid location.

## Step 4: Accept Anaconda Terms of Service (If Applicable)
1. Run the following commands to accept terms for Anaconda channels (note: this may be specific to certain setups; skip if `conda tos accept` is not recognized):
   - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
   - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r`
   - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2`
2. If the command fails, check Conda documentation for equivalent configuration steps (e.g., using `conda config`).

## Step 5: Create or Verify Conda Environment
1. List existing environments: `conda env list`.
2. If "trellis" does not exist:
   - Create it: `conda create -n trellis python=3.11.9 -y`.
3. If creation fails, troubleshoot Conda issues (e.g., network, permissions).

## Step 6: Activate Conda Environment
1. Activate: `conda activate trellis`.
2. If activation fails, ensure the environment exists and retry.

## Step 7: Update Pip and Install Build Tools
1. Upgrade pip and wheel: `python -m pip install --upgrade pip wheel`.
2. Install setuptools: `python -m pip install setuptools==75.8.2`. (Note: Version is hardcoded; use latest if needed.)
3. If any installation fails, check pip logs or network connectivity.

## Step 8: Install Torch Requirements
1. Install packages from file: `pip install -r requirements-torch.txt`.
2. If fails, verify the file contents and resolve dependency conflicts.

## Step 9: Install Main Requirements
1. Install packages from file: `pip install -r requirements.txt`.
2. If fails, verify the file contents and resolve dependency conflicts.

## Step 10: Set Environment Variable
1. Run: `python set_environment_variable.py` (this sets `CHAT_TO_3D_PATH`).
2. Verify the variable is set: `echo %CHAT_TO_3D_PATH%`.
3. If the script fails, inspect `set_environment_variable.py` for issues.

## Step 11: Download Required Models
1. Note your current directory: `echo %cd%`.
2. Run: `python download_models.py`.
3. If fails, check the script for download URLs and ensure internet access.
4. Return to the original directory if changed.

## Step 12: Install Blender Addons
1. Verify `blender` folder exists with `NV_Trellis_Addon.py` and `asset_importer.py`.
2. Set Blender root: `%APPDATA%\Blender Foundation\Blender`.
3. If the root doesn't exist, create it: `mkdir "%APPDATA%\Blender Foundation\Blender"`.
4. Check for version folders (4.2 to 5.0).
5. For each matching version (e.g., 4.2):
   - Create addons dir if needed: `mkdir "<BlenderRoot>\<version>\scripts\addons"`.
   - Copy files: `copy blender\NV_Trellis_Addon.py "<addons_dir>\NV_Trellis_Addon.py"` and `copy blender\asset_importer.py "<addons_dir>\asset_importer.py"`.
6. If no versions found, create default 4.2 folder and copy addons as above.
7. After copying, enable the addon in Blender preferences.

## Step 13: Deactivate Conda Environment (Temporarily)
1. Run: `conda deactivate`.

## Step 14: Start LLM and Trellis Services
1. Start LLM in background: Open a new Command Prompt and run `conda activate trellis && python nim_llm\run_llama.py`.
2. Wait 10 seconds.
3. Start Trellis in background: Open another Command Prompt and run `conda activate trellis && python nim_trellis\run_trellis.py`.
4. Wait 10 seconds.
5. Check running processes: `tasklist /FI "IMAGENAME eq python.exe" /FO TABLE`.

## Step 15: Monitor Services Readiness
1. Reactivate environment in main prompt: `conda activate trellis`.
2. Loop to check readiness (up to 150 attempts, ~75 minutes):
   - Run: `python check_services.py`.
   - Interpret exit code:
     - 0: Both ready.
     - 1: LLM ready, Trellis not.
     - 2: Trellis ready, LLM not.
     - Other: Neither ready.
   - Wait 30 seconds between checks.
3. Services endpoints: LLM at `http://localhost:19002`, Trellis at `http://localhost:8000`.
4. If timeout, check logs: `nim_llm\llama_container.log` and `nim_trellis\trellis_container.log`.

## Step 16: Stop Services
1. Stop LLM: `python -c "from nim_llm.manager import stop_container; stop_container()"`.
2. Stop Trellis: `python -c "from nim_trellis.manager import stop_container; stop_container()"`.
3. Kill remaining Python processes: `taskkill /f /im python.exe`.
4. Deactivate environment: `conda deactivate`.

## Completion
If all steps succeed without errors, the installation is complete. You can now use the Trellis environment. Troubleshoot any errors by checking console output, logs, or dependencies. For first-time setups, model downloads and service starts may take significant time (60-120 minutes).

</details>

---

## Troubleshooting

### Common Issues

1. **CUDA not found during installation**
   - Ensure CUDA 12.8 is installed
   - Set `CUDA_HOME` environment variable:
     ```bash
     set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
     ```

2. **Out of VRAM**
   - Enable restricted mode in `config.py`:
     ```python
     NATIVE_VRAM_RESTRICTED_MODE = True
     NATIVE_RAM_RESTRICTED_MODE = True
     ```
   - Use a smaller LLM model (Qwen3-4B vs Llama-3.1-8B)

3. **Slow LLM inference**
   - Ensure using Gradio 5.x (not 6.x)
   - Check `requirements.txt` has `gradio==5.50.0`

4. **Model download fails**
   - Set HuggingFace token: `set HF_TOKEN=your_token`
   - Check internet connection

5. **TRELLIS import errors**
   - Ensure submodules are initialized:
     ```bash
     git submodule update --init --recursive
     ```

6. **Installation Issues**:
   - Ensure all prerequisites are installed correctly
   - Check if Python is in your system PATH
   - Verify Visual Studio Build Tools installation

7. **Runtime Issues (NIM Mode)**:
   - Make sure both NIM containers are running
   - Check GPU memory usage
   - Verify all environment variables are set correctly

### Logs

- Application logs: Console output
- Verbose logging: Set `VERBOSE = True` in `config.py`
- NIM logs: `nim_llm\llama_container.log` and `nim_trellis\trellis_container.log`

## Acknowledgments

- [TRELLIS](https://github.com/microsoft/TRELLIS) - Microsoft's 3D generation model
- [Qwen3](https://github.com/QwenLM/Qwen3) - Alibaba's LLM (Native mode)
- [SANA](https://github.com/NVlabs/Sana) - NVIDIA's image generation model
- [Griptape](https://github.com/griptape-ai/griptape) - Agent framework (NIM mode)
- [Gradio](https://github.com/gradio-app/gradio) - Web interface framework

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
