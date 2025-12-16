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

REM Read CONDA_ENV_NAME from config.py
set "ENV_NAME=trellis"
for /f "tokens=2 delims==" %%a in ('findstr /C:"CONDA_ENV_NAME" config.py 2^>nul') do (
    set "RAW_VALUE=%%a"
    set "RAW_VALUE=!RAW_VALUE: =!"
    set "RAW_VALUE=!RAW_VALUE:"=!"
    if not "!RAW_VALUE!"=="" set "ENV_NAME=!RAW_VALUE!"
)

REM Read USE_NATIVE_LLM from config.py
set "USE_NATIVE_LLM=True"
for /f "tokens=2 delims==" %%a in ('findstr /C:"USE_NATIVE_LLM" config.py 2^>nul') do (
    set "RAW_VALUE=%%a"
    set "RAW_VALUE=!RAW_VALUE: =!"
    set "RAW_VALUE=!RAW_VALUE:"=!"
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
    for /f "tokens=1 delims=#" %%b in ("!RAW_VALUE!") do set "RAW_VALUE=%%b"
    if /i "!RAW_VALUE!"=="True" set "USE_NATIVE_TRELLIS=True"
    if /i "!RAW_VALUE!"=="False" set "USE_NATIVE_TRELLIS=False"
)

echo Configuration:
echo   Conda environment: %ENV_NAME%
echo   USE_NATIVE_LLM: %USE_NATIVE_LLM%
echo   USE_NATIVE_TRELLIS: %USE_NATIVE_TRELLIS%
echo.

:: Check if we need to start any NIM services
if /i "%USE_NATIVE_LLM%"=="True" if /i "%USE_NATIVE_TRELLIS%"=="True" (
    echo ========================================
    echo No NIM services needed (using native models)
    echo ========================================
    echo.
    echo Both LLM and TRELLIS are configured to use native PyTorch models.
    echo Simply run: python app.py
    echo.
    pause
    exit /b 0
)

echo Starting NIM services...
echo Using conda environment: %ENV_NAME%

:: Check if conda environment exists
call conda env list | findstr "%ENV_NAME%" >nul
if errorlevel 1 (
    echo [ERROR] Conda environment '%ENV_NAME%' not found!
    echo Please run install.bat first to set up the environment.
    pause
    exit /b 1
)

:: Start LLM service in background (only if not using native LLM)
if /i "%USE_NATIVE_LLM%"=="False" (
    echo Starting LLM NIM service...
    start /B cmd /c "call conda activate %ENV_NAME% && python nim_llm\run_llama.py"
    if errorlevel 1 (
        echo [ERROR] Failed to start LLM service!
        pause
        exit /b 1
    )
    timeout /t 10 /nobreak
) else (
    echo Skipping LLM NIM service (using native LLM)
)

:: Start Trellis service in background (only if not using native TRELLIS)
if /i "%USE_NATIVE_TRELLIS%"=="False" (
    echo Starting Trellis NIM service...
    start /B cmd /c "call conda activate %ENV_NAME% && python nim_trellis\run_trellis.py"
    if errorlevel 1 (
        echo [ERROR] Failed to start Trellis service!
        pause
        exit /b 1
    )
) else (
    echo Skipping Trellis NIM service (using native TRELLIS)
)

echo.
echo Services started in background!
echo.
echo Service logs:
if /i "%USE_NATIVE_LLM%"=="False" echo - LLM logs: nim_llm\llama_container.log
if /i "%USE_NATIVE_TRELLIS%"=="False" echo - Trellis logs: nim_trellis\trellis_container.log
echo.
echo To check if services are ready, run: python check_services.py
echo.
pause 