@echo off
setlocal EnableDelayedExpansion

REM Double-clickable Windows launcher for OCTolyzer.
REM Step 1: Resolve repository root and place OS-specific launcher in root
REM Step 2: Ensure uv is installed
REM Step 3: Check and pull OCTolyzer code updates from GitHub
REM Step 4: Provision .venv with the newest compatible Python (if missing)
REM Step 5: Sync requirements.txt into .venv
REM Step 6: Launch `python -m gui.app`

REM --- Step 1: Locate repository root & place OS launcher ---
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%..\..\octolyzer" (
    cd /d "%SCRIPT_DIR%..\.."
) else (
    cd /d "%SCRIPT_DIR%"
)

if not exist "OCTolyzer.bat" (
    if exist "gui\launchers\OCTolyzer.bat" (
        copy /Y "gui\launchers\OCTolyzer.bat" "OCTolyzer.bat" >nul 2>nul
    )
)
if exist "OCTolyzer.command" del /f /q "OCTolyzer.command" >nul 2>nul
if exist "OCTolyzer.sh" del /f /q "OCTolyzer.sh" >nul 2>nul

REM --- Step 2: Ensure uv is installed ---
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv;%PATH%"

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing uv ^(first-time setup^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv;%PATH%"
)

REM --- Step 3: Check & pull OCTolyzer code updates first ---
if not "%OCTOLYZER_LAUNCHER_REEXEC%"=="1" (
    echo Checking for OCTolyzer updates...
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" -m gui.updater --code-only
    ) else (
        if exist ".git" (
            where git >nul 2>nul
            if !ERRORLEVEL! equ 0 (
                git fetch --quiet origin main >nul 2>nul
                git merge --ff-only origin/main >nul 2>nul
            )
        )
    )

    REM --- Step 3b: If the launcher template was updated, update root launcher & re-exec ---
    if exist "gui\launchers\OCTolyzer.bat" (
        fc /b "gui\launchers\OCTolyzer.bat" "OCTolyzer.bat" >nul 2>nul
        if !ERRORLEVEL! neq 0 (
            echo Updating launcher script to latest version...
            set "OCTOLYZER_LAUNCHER_REEXEC=1" & copy /Y "gui\launchers\OCTolyzer.bat" "OCTolyzer.bat" >nul 2>nul & call "OCTolyzer.bat" %* & exit /b !ERRORLEVEL!
        )
    )
)

REM --- Step 4 & 5: Provision or self-heal .venv and sync requirements.txt ---
set "BINARY_PKGS=torch,torchvision,numpy,scipy,matplotlib,opencv-python,scikit-image,scikit-learn,SimpleITK,numba,llvmlite,bottleneck,PySide6,pandas,Pillow"
set "NEED_PROVISION=0"
set "HAD_VENV=0"

if not exist ".venv\Scripts\python.exe" (
    set "NEED_PROVISION=1"
) else (
    set "HAD_VENV=1"
    echo Verifying OCTolyzer dependencies...
    uv pip install --python ".venv\Scripts\python.exe" --only-binary "!BINARY_PKGS!" --quiet -r requirements.txt >nul 2>nul
    if !ERRORLEVEL! neq 0 (
        set "CURRENT_PY_VER="
        for /f "delims=" %%P in ('".venv\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "CURRENT_PY_VER=%%P"
        if not defined CURRENT_PY_VER (
            set "NEED_PROVISION=1"
        ) else (
            uv pip compile requirements.txt --python-version "!CURRENT_PY_VER!" --only-binary "!BINARY_PKGS!" --quiet >nul 2>nul
            if !ERRORLEVEL! neq 0 (
                echo Current Python ^(!CURRENT_PY_VER!^) is no longer compatible with updated requirements. Rebuilding environment...
                set "NEED_PROVISION=1"
            )
        )
    )
)

if "!NEED_PROVISION!"=="1" (
    echo Selecting the latest Python version compatible with all dependencies...
    set "SELECTED_PYTHON="
    set "LOCKFILE=%TEMP%\octolyzer-reqs-%RANDOM%.txt"
    for %%V in (3.13 3.12 3.11) do (
        if not defined SELECTED_PYTHON (
            uv pip compile requirements.txt --python-version %%V --only-binary "!BINARY_PKGS!" --quiet --output-file "!LOCKFILE!" >nul 2>nul
            if !ERRORLEVEL! equ 0 (
                if exist ".venv" rmdir /s /q ".venv" >nul 2>nul
                uv venv .venv --python %%V --python-preference managed
                if !ERRORLEVEL! equ 0 (
                    set "SELECTED_PYTHON=%%V"
                )
            )
        )
    )
    if not defined SELECTED_PYTHON (
        if exist "!LOCKFILE!" del /f /q "!LOCKFILE!" >nul 2>nul
        if exist ".venv\Scripts\python.exe" (
            echo Warning: Could not resolve updated requirements ^(offline?^); continuing with existing environment.
        ) else (
            echo Failed to find a Python version compatible with all dependencies.
            pause
            exit /b 1
        )
    ) else (
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
        if "!HAD_VENV!"=="0" (
            REM Check for code updates now that Python is available (for non-git ZIP installs)
            ".venv\Scripts\python.exe" -m gui.updater
        )
    )
)

REM --- Step 6: Launch GUI ---
echo Launching OCTolyzer GUI...
set "OCTOLYZER_UPDATED_ONCE=1"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m gui.app %*
) else (
    ".venv\Scripts\python.exe" -m gui.app %*
)
