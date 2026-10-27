@echo off
setlocal EnableDelayedExpansion

REM Double-clickable Windows launcher for OCTolyzer.
REM Automatically installs uv, provisions a virtual environment with the latest
REM Python, updates OCTolyzer and its dependencies, and starts `python -m gui.app`.

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%..\..\octolyzer" (
    cd /d "%SCRIPT_DIR%..\.."
) else (
    cd /d "%SCRIPT_DIR%"
)

if not exist "OCTolyzer.bat" (
    copy /Y "%~f0" "OCTolyzer.bat" >nul 2>nul
)

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv;%PATH%"

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing uv ^(first-time setup^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv;%PATH%"
)

if not exist ".venv\Scripts\python.exe" (
    echo Selecting the latest Python version compatible with all dependencies...
    set "SELECTED_PYTHON="
    set "LOCKFILE=%TEMP%\octolyzer-reqs-%RANDOM%.txt"
    set "BINARY_PKGS=torch,torchvision,numpy,scipy,matplotlib,opencv-python,scikit-image,scikit-learn,SimpleITK,numba,llvmlite,bottleneck,PySide6,pandas,Pillow"
    for %%V in (3.13 3.12 3.11) do (
        if not defined SELECTED_PYTHON (
            uv pip compile requirements.txt --python-version %%V --only-binary "!BINARY_PKGS!" --quiet --output-file "!LOCKFILE!" >nul 2>nul
            if !ERRORLEVEL! equ 0 (
                uv venv .venv --python %%V --python-preference managed
                if !ERRORLEVEL! equ 0 (
                    set "SELECTED_PYTHON=%%V"
                )
            )
        )
    )
    if not defined SELECTED_PYTHON (
        if exist "!LOCKFILE!" del /f /q "!LOCKFILE!" >nul 2>nul
        echo Failed to find a Python version compatible with all dependencies.
        pause
        exit /b 1
    )
    echo Installing OCTolyzer dependencies for Python !SELECTED_PYTHON! ^(this may take a few minutes on first run^)...
    uv pip install --python ".venv\Scripts\python.exe" --only-binary "!BINARY_PKGS!" -r "!LOCKFILE!"
    if !ERRORLEVEL! neq 0 (
        if exist "!LOCKFILE!" del /f /q "!LOCKFILE!" >nul 2>nul
        rmdir /s /q ".venv" >nul 2>nul
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
    if exist "!LOCKFILE!" del /f /q "!LOCKFILE!" >nul 2>nul
)

echo Checking for OCTolyzer updates...
".venv\Scripts\python.exe" -m gui.updater

echo Launching OCTolyzer GUI...
set "OCTOLYZER_UPDATED_ONCE=1"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m gui.app %*
) else (
    ".venv\Scripts\python.exe" -m gui.app %*
)
