@REM #
@REM # SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
@REM # SPDX-License-Identifier: Apache-2.0
@REM #
@REM # Licensed under the Apache License, Version 2.0 (the "License");
@REM # you may not use this file except in compliance with the License.
@REM # You may obtain a copy of the License at
@REM #
@REM # http://www.apache.org/licenses/LICENSE-2.0
@REM #
@REM # Unless required by applicable law or agreed to in writing, software
@REM # distributed under the License is distributed on an "AS IS" BASIS,
@REM # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM # See the License for the specific language governing permissions and
@REM # limitations under the License.
@REM #
@echo off
setlocal enabledelayedexpansion

echo Starting installation process...

REM Read CONDA_ENV_NAME from config.py
set "ENV_NAME=trellis"
for /f "tokens=2 delims==" %%a in ('findstr /C:"CONDA_ENV_NAME" config.py 2^>nul') do (
    set "RAW_VALUE=%%a"
    REM Remove quotes and spaces
    set "RAW_VALUE=!RAW_VALUE: =!"
    set "RAW_VALUE=!RAW_VALUE:"=!"
    if not "!RAW_VALUE!"=="" set "ENV_NAME=!RAW_VALUE!"
)
echo Using conda environment name: %ENV_NAME%

REM Read USE_NATIVE_LLM from config.py
set "USE_NATIVE_LLM=True"
for /f "tokens=2 delims==" %%a in ('findstr /C:"USE_NATIVE_LLM" config.py 2^>nul') do (
    set "RAW_VALUE=%%a"
    set "RAW_VALUE=!RAW_VALUE: =!"
    set "RAW_VALUE=!RAW_VALUE:"=!"
    REM Remove inline comments
    for /f "tokens=1 delims=#" %%b in ("!RAW_VALUE!") do set "RAW_VALUE=%%b"
    if /i "!RAW_VALUE!"=="True" set "USE_NATIVE_LLM=True"
    if /i "!RAW_VALUE!"=="False" set "USE_NATIVE_LLM=False"
)

REM Read USE_NATIVE_TRELLIS from config.py
set "USE_NATIVE_TRELLIS=True"
for /f "tokens=2 delims==" %%a in ('findstr /C:"USE_NATIVE_TRELLIS" config.py 2^>nul') do (
    set "RAW_VALUE=%%a"
    set "RAW_VALUE=!RAW_VALUE: =!"
    set "RAW_VALUE=!RAW_VALUE:"=!"
    REM Remove inline comments
    for /f "tokens=1 delims=#" %%b in ("!RAW_VALUE!") do set "RAW_VALUE=%%b"
    if /i "!RAW_VALUE!"=="True" set "USE_NATIVE_TRELLIS=True"
    if /i "!RAW_VALUE!"=="False" set "USE_NATIVE_TRELLIS=False"
)

echo Configuration:
echo   Conda environment: %ENV_NAME%
echo   USE_NATIVE_LLM: %USE_NATIVE_LLM%
echo   USE_NATIVE_TRELLIS: %USE_NATIVE_TRELLIS%

REM Check if conda is installed
echo Checking for Conda...
where conda >nul 2>&1
set CONDA_ERRORLEVEL=%errorlevel%
echo Conda command returned errorlevel: %CONDA_ERRORLEVEL%
if %CONDA_ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda is not installed or not in PATH. Please install Conda first.
    pause
    exit /b 1
)
echo Conda check passed, proceeding to file checks...

REM Check if requirements files exist
if not exist requirements-torch.txt (
    echo [ERROR] requirements-torch.txt not found!
    pause
    exit /b 1
)
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)
if not exist install_dependencies.py (
    echo [ERROR] install_dependencies.py not found!
    pause
    exit /b 1
)
if not exist config.py (
    echo [ERROR] config.py not found!
    pause
    exit /b 1
)

:: Ensure Conda is initialized in the current shell
echo Initializing Conda...
:: Use %USERPROFILE% for Conda path, checking common installation locations
set "CONDA_FOUND=0"

:: Check first location
set "CONDA_PATH=%USERPROFILE%\miniconda3"
if exist "%CONDA_PATH%\Scripts\conda.exe" (
    set "CONDA_FOUND=1"
) else if exist "%CONDA_PATH%\condabin\conda.bat" (
    set "CONDA_FOUND=1"
)

:: Check second location if first not found
if !CONDA_FOUND! equ 0 (
    set "CONDA_PATH=%USERPROFILE%\AppData\Local\miniconda3"
    if exist "%CONDA_PATH%\Scripts\conda.exe" (
        set "CONDA_FOUND=1"
    ) else if exist "%CONDA_PATH%\condabin\conda.bat" (
        set "CONDA_FOUND=1"
    )
)

:: Check third location if still not found
if !CONDA_FOUND! equ 0 (
    set "CONDA_PATH=%USERPROFILE%\anaconda3"
    if exist "%CONDA_PATH%\Scripts\conda.exe" (
        set "CONDA_FOUND=1"
    ) else if exist "%CONDA_PATH%\condabin\conda.bat" (
        set "CONDA_FOUND=1"
    )
)

:: If still not found, show error
if !CONDA_FOUND! equ 0 (
    echo [ERROR] Conda not found in common locations. Please verify your Conda installation path.
    echo Checked locations:
    echo   %USERPROFILE%\miniconda3
    echo   %USERPROFILE%\AppData\Local\miniconda3
    echo   %USERPROFILE%\anaconda3
    pause
    exit /b 1
)
:: Initialize Conda for this session
echo Conda found at: %CONDA_PATH%
echo Using conda command from PATH...

:: Accept Anaconda Terms of Service
echo Accepting Anaconda Terms of Service...
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

:: Create conda environment if it doesn't exist
echo Checking for %ENV_NAME% environment...
call conda env list | findstr "%ENV_NAME%" >nul
if errorlevel 1 (
    echo Creating conda environment '%ENV_NAME%'...
    call conda create -n %ENV_NAME% python=3.11.9 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment!
        pause
        exit /b 1
    )
    echo Conda environment '%ENV_NAME%' created successfully.
) else (
    echo Conda environment '%ENV_NAME%' already exists.
)

:: Activate conda environment
echo Activating conda environment...
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment!
    pause
    exit /b 1
)


REM Update pip and install build tools
echo Updating pip and installing build tools...
python -m pip install --upgrade pip wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip and wheel!
    pause
    exit /b 1
)

python -m pip install setuptools==75.8.2
if errorlevel 1 (
    echo [ERROR] Failed to install setuptools!
    pause
    exit /b 1
)

REM Install torch requirements
echo Installing torch requirements...
pip install -r requirements-torch.txt
if errorlevel 1 (
    echo [ERROR] Failed to install torch requirements!
    pause
    exit /b 1
)


REM Install dependencies based on config.py settings
echo Installing dependencies based on config.py settings...
python install_dependencies.py
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

REM Set CHAT_TO_3D_PATH environment variable using conda Python
echo Setting CHAT_TO_3D_PATH environment variable...
call conda run -n %ENV_NAME% --no-capture-output python set_environment_variable.py
if errorlevel 1 (
    echo [ERROR] Failed to set CHAT_TO_3D_PATH environment variable!
    pause
    exit /b 1
)
echo CHAT_TO_3D_PATH environment variable set successfully.

REM Download required models
echo Downloading required models...
set CURRENT_DIR=%cd%
echo current directory: %CURRENT_DIR%
call conda run -n %ENV_NAME% --no-capture-output python download_models.py
if errorlevel 1 (
    echo [ERROR] Failed to download models!
    cd "%CURRENT_DIR%"
    pause
    exit /b 1
)
echo current directory: %CURRENT_DIR%
cd "%CURRENT_DIR%"

echo Installation completed successfully!


:: Start Blender addon installation
echo Starting Blender addon installation...

:: Check if blender folder exists
if not exist blender (
    echo [ERROR] blender folder not found in current directory!
    pause
    exit /b 1
)

:: Check if source files exist
if not exist blender\NV_Trellis_Addon.py (
    echo [ERROR] NV_Trellis_Addon.py not found in blender folder!
    pause
    exit /b 1
)
if not exist blender\asset_importer.py (
    echo [ERROR] asset_importer.py not found in blender folder!
    pause
    exit /b 1
)

:: Set Blender root directory
set "BLENDER_ROOT=%appdata%\Blender Foundation\Blender"
echo Blender root directory: %BLENDER_ROOT%

:: Check if Blender root exists, create if not
if not exist "%BLENDER_ROOT%" (
    echo Creating Blender root directory: %BLENDER_ROOT%
    mkdir "%BLENDER_ROOT%"
    if errorlevel 1 (
        echo [ERROR] Failed to create Blender root directory!
        pause
        exit /b 1
    )
)

:: Check for existing Blender version folders
echo Checking for Blender version folders...
set "VERSION_FOUND=0"
for /d %%D in ("%BLENDER_ROOT%\*") do (
    set "VERSION=%%~nxD"
    set "IS_VERSION=0"
    if "!VERSION!"=="4.2" set "IS_VERSION=1"
    if "!VERSION!"=="4.3" set "IS_VERSION=1"
    if "!VERSION!"=="4.4" set "IS_VERSION=1"
    if "!VERSION!"=="4.5" set "IS_VERSION=1"
    if "!VERSION!"=="4.6" set "IS_VERSION=1"
    if "!VERSION!"=="4.7" set "IS_VERSION=1"
    if "!VERSION!"=="4.8" set "IS_VERSION=1"
    if "!VERSION!"=="4.9" set "IS_VERSION=1"
    if "!VERSION!"=="5.0" set "IS_VERSION=1"
    if !IS_VERSION! equ 1 (
        set "VERSION_FOUND=1"
        echo Processing Blender version !VERSION!
        set "ADDONS_DIR=%%D\scripts\addons"
        if not exist "!ADDONS_DIR!" (
            echo Creating addons directory: !ADDONS_DIR!
            mkdir "!ADDONS_DIR!"
            if errorlevel 1 (
                echo [ERROR] Failed to create addons directory: !ADDONS_DIR!
                pause
                exit /b 1
            )
        )
        echo Copying NV_Trellis_Addon.py to !ADDONS_DIR!...
        copy blender\NV_Trellis_Addon.py "!ADDONS_DIR!\NV_Trellis_Addon.py"
        if errorlevel 1 (
            echo [ERROR] Failed to copy NV_Trellis_Addon.py to !ADDONS_DIR!
            pause
            exit /b 1
        )
        echo Copying asset_importer.py to !ADDONS_DIR!...
        copy blender\asset_importer.py "!ADDONS_DIR!\asset_importer.py"
        if errorlevel 1 (
            echo [ERROR] Failed to copy asset_importer.py to !ADDONS_DIR!
            pause
            exit /b 1
        )
        echo Successfully copied addons to !ADDONS_DIR!
    )
)

:: If no version folders found, create 4.2 and install addons
if !VERSION_FOUND! equ 0 (
    echo No Blender version folders found. Creating default Blender 4.2 folder...
    set "DEFAULT_VERSION_DIR=%BLENDER_ROOT%\4.2"
    set "DEFAULT_ADDONS_DIR=%BLENDER_ROOT%\4.2\scripts\addons"
    echo Creating default version directory: !DEFAULT_VERSION_DIR!
    mkdir "!DEFAULT_VERSION_DIR!"
    if errorlevel 1 (
        echo [ERROR] Failed to create default version directory: !DEFAULT_VERSION_DIR!
        pause
        exit /b 1
    )
    echo Creating default addons directory: !DEFAULT_ADDONS_DIR!
    mkdir "!DEFAULT_ADDONS_DIR!"
    if errorlevel 1 (
        echo [ERROR] Failed to create default addons directory: !DEFAULT_ADDONS_DIR!
        pause
        exit /b 1
    )
    echo Copying NV_Trellis_Addon.py to !DEFAULT_ADDONS_DIR!...
    copy blender\NV_Trellis_Addon.py "!DEFAULT_ADDONS_DIR!\NV_Trellis_Addon.py"
    if errorlevel 1 (
        echo [ERROR] Failed to copy NV_Trellis_Addon.py to !DEFAULT_ADDONS_DIR!
        pause
        exit /b 1
    )
    echo Copying asset_importer.py to !DEFAULT_ADDONS_DIR!...
    copy blender\asset_importer.py "!DEFAULT_ADDONS_DIR!\asset_importer.py"
    if errorlevel 1 (
        echo [ERROR] Failed to copy asset_importer.py to !DEFAULT_ADDONS_DIR!
        pause
        exit /b 1
    )
    echo Successfully copied addons to !DEFAULT_ADDONS_DIR!
)

echo Blender addon installation completed successfully!

:: Deactivate conda environment
echo Deactivating conda environment...
call conda deactivate

:: Check if we need to start NIM services
:: Skip NIM services if using both native LLM and native TRELLIS
if /i "%USE_NATIVE_LLM%"=="True" if /i "%USE_NATIVE_TRELLIS%"=="True" (
    echo.
    echo ========================================
    echo Skipping NIM services (using native models)
    echo ========================================
    echo   USE_NATIVE_LLM: %USE_NATIVE_LLM%
    echo   USE_NATIVE_TRELLIS: %USE_NATIVE_TRELLIS%
    echo.
    echo Native models will be loaded when you run: python app.py
    echo.
    goto :installation_complete
)

:: Start LLM and Trellis NIM services (only if not using native models)
echo.
echo ========================================
echo Starting NIM services...
echo ========================================
echo This may take several minutes as containers need to download and start...

:: Start LLM service in background (only if not using native LLM)
if /i "%USE_NATIVE_LLM%"=="False" (
    echo Starting LLM NIM service...
    start /B "LLM Service" cmd /c "call conda activate %ENV_NAME% && python nim_llm\run_llama.py"
    if errorlevel 1 (
        echo [ERROR] Failed to start LLM service!
        pause
        exit /b 1
    )
    echo Waiting 10 seconds for LLM service to initialize...
    timeout /t 10 /nobreak >nul
) else (
    echo Skipping LLM NIM service (using native LLM)
)

:: Start Trellis service in background (only if not using native TRELLIS)
if /i "%USE_NATIVE_TRELLIS%"=="False" (
    echo Starting Trellis NIM service...
    start /B "Trellis Service" cmd /c "call conda activate %ENV_NAME% && python nim_trellis\run_trellis.py"
    if errorlevel 1 (
        echo [ERROR] Failed to start Trellis service!
        pause
        exit /b 1
    )
    echo Waiting 10 seconds for Trellis service to initialize...
    timeout /t 10 /nobreak >nul
) else (
    echo Skipping Trellis NIM service (using native TRELLIS)
)

:: Check if Python processes are running
echo Checking if service processes are running...
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
echo.

:: Wait for services to be ready
echo.
echo Waiting for NIM services to be ready...
echo This may take 60-120 minutes for first-time setup...

:: Use conda environment for service checking
echo Setting up service monitoring environment...
call conda activate %ENV_NAME%

set /a attempts=0
set /a max_attempts=150
set /a llm_ready=0
set /a trellis_ready=0

:: If using native, mark as ready
if /i "%USE_NATIVE_LLM%"=="True" set /a llm_ready=1
if /i "%USE_NATIVE_TRELLIS%"=="True" set /a trellis_ready=1

:wait_loop
set /a attempts+=1
echo Attempt %attempts%/%max_attempts% - Checking services...

:: Use Python health checker
python check_services.py
set check_result=%errorlevel%

:: Handle health checker failures gracefully
if %check_result% geq 3 (
    echo [WARNING] Health checker failed, retrying in 30 seconds...
    timeout /t 30 /nobreak >nul
    goto :wait_loop
)

if %check_result% equ 0 (
    set /a llm_ready=1
    set /a trellis_ready=1
    echo ✅ Both services are ready!
) else if %check_result% equ 1 (
    if %llm_ready% equ 0 (
        set /a llm_ready=1
        echo ✅ LLM service is ready!
    )
    if /i "%USE_NATIVE_TRELLIS%"=="False" echo Trellis NIM service not ready yet...
) else if %check_result% equ 2 (
    if %trellis_ready% equ 0 (
        set /a trellis_ready=1
        echo ✅ Trellis service is ready!
    )
    if /i "%USE_NATIVE_LLM%"=="False" echo LLM NIM service not ready yet...
) else (
    echo Services not ready yet...
)

:: Check if both services are ready
if %llm_ready% equ 1 if %trellis_ready% equ 1 (
    echo.
    echo 🎉 All services are ready!
    echo.
    echo Services running:
    if /i "%USE_NATIVE_LLM%"=="False" echo - LLM NIM Service: http://localhost:19002
    if /i "%USE_NATIVE_LLM%"=="True" echo - LLM: Native model (loaded on demand)
    if /i "%USE_NATIVE_TRELLIS%"=="False" echo - Trellis NIM Service: http://localhost:8000
    if /i "%USE_NATIVE_TRELLIS%"=="True" echo - Trellis: Native model (loaded on demand)
    echo.
    echo.
    goto :stop_services
)

:: Check if we've exceeded max attempts
if %attempts% geq %max_attempts% (
    echo.
    echo ⚠️ Timeout waiting for services to be ready
    echo.
    echo Current status:
    if %llm_ready% equ 1 (
        echo - LLM Service: ✅ Ready
    ) else (
        echo - LLM Service: ❌ Not ready
    )
    if %trellis_ready% equ 1 (
        echo - Trellis Service: ✅ Ready
    ) else (
        echo - Trellis Service: ❌ Not ready
    )
    echo.
    echo You can check the service logs manually:
    echo - LLM logs: nim_llm\llama_container.log
    echo - Trellis logs: nim_trellis\trellis_container.log
    echo.
    goto :stop_services
)

:: Wait 10 seconds before next attempt
timeout /t 30 /nobreak >nul
goto :wait_loop

:stop_services
:: Stop the services automatically (only NIM services that were started)
echo.
echo Stopping NIM services...
echo ========================================

:: Stop LLM container (only if not using native LLM)
if /i "%USE_NATIVE_LLM%"=="False" (
    echo Stopping LLM container...
    call conda run -n %ENV_NAME% --no-capture-output python -c "from nim_llm.manager import stop_container; stop_container()"
    if errorlevel 1 (
        echo [WARNING] Failed to stop LLM container gracefully
    )
)

:: Stop Trellis container (only if not using native TRELLIS)
if /i "%USE_NATIVE_TRELLIS%"=="False" (
    echo Stopping Trellis container...
    call conda run -n %ENV_NAME% --no-capture-output python -c "from nim_trellis.manager import stop_container; stop_container()"
    if errorlevel 1 (
        echo [WARNING] Failed to stop Trellis container gracefully
    )
)

:: Kill any remaining Python processes that might be running the services
echo Stopping Python processes...
taskkill /f /im python.exe 2>nul
if errorlevel 1 (
    echo No Python processes found to stop
) else (
    echo Python processes stopped
)

echo.
echo ✅ Services stopped successfully!
echo.

:installation_complete
echo Installation completed successfully

pause