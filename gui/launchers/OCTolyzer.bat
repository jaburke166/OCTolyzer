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
    echo Creating Python environment with the latest Python...
    uv venv .venv --python-preference managed
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Installing OCTolyzer dependencies ^(this may take a few minutes on first run^)...
    uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
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
