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

echo Stopping NIM services...

:: Stop LLM container (only if not using native LLM)
if /i "%USE_NATIVE_LLM%"=="False" (
    echo Stopping LLM container...
    call conda run -n %ENV_NAME% --no-capture-output python -c "from nim_llm.manager import stop_container; stop_container()"
    if errorlevel 1 (
        echo [WARNING] Failed to stop LLM container gracefully
    )
) else (
    echo Skipping LLM container (using native LLM)
)

:: Stop Trellis container (only if not using native TRELLIS)
if /i "%USE_NATIVE_TRELLIS%"=="False" (
    echo Stopping Trellis container...
    call conda run -n %ENV_NAME% --no-capture-output python -c "from nim_trellis.manager import stop_container; stop_container()"
    if errorlevel 1 (
        echo [WARNING] Failed to stop Trellis container gracefully
    )
) else (
    echo Skipping Trellis container (using native TRELLIS)
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
echo Services stopped!
echo.
