#!/usr/bin/env bash
# Double-clickable Linux launcher for OCTolyzer.
# Step 1: Resolve repository root and place OS-specific launcher in root
# Step 2: Ensure uv is installed
# Step 3: Check and pull OCTolyzer code updates from GitHub (and self-update root launcher if changed)
# Step 4: Provision .venv with the newest compatible Python (if missing)
# Step 5: Sync requirements.txt into .venv
# Step 6: Launch `python -m gui.app`

set -euo pipefail

main() {
    # --- Step 1: Locate repository root & place OS launcher ---
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ "$(basename "$SCRIPT_DIR")" = "launchers" ] && [ "$(basename "$(dirname "$SCRIPT_DIR")")" = "gui" ]; then
        REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    else
        REPO_ROOT="$SCRIPT_DIR"
    fi
    cd "$REPO_ROOT"

    TARGET_LAUNCHER="$REPO_ROOT/OCTolyzer.sh"
    TEMPLATE_LAUNCHER="$REPO_ROOT/gui/launchers/OCTolyzer.sh"
    if [ ! -f "$TARGET_LAUNCHER" ] && [ -f "$TEMPLATE_LAUNCHER" ]; then
        cp "$TEMPLATE_LAUNCHER" "$TARGET_LAUNCHER"
        chmod +x "$TARGET_LAUNCHER"
    fi
    rm -f "$REPO_ROOT/OCTolyzer.command" "$REPO_ROOT/OCTolyzer.bat" 2>/dev/null || true

    # --- Step 2: Ensure uv is installed ---
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv (first-time setup)..."
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://astral.sh/uv/install.sh | sh
        else
            echo "Error: curl or wget is required to download uv." >&2
            read -r -p "Press Enter to close..." _
            exit 1
        fi
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi

    # --- Step 3: Check & pull OCTolyzer code updates first ---
    if [ "${OCTOLYZER_LAUNCHER_REEXEC:-0}" != "1" ]; then
        echo "Checking for OCTolyzer updates..."
        if [ -x ".venv/bin/python" ]; then
            .venv/bin/python -m gui.updater --code-only || true
        elif command -v python3 >/dev/null 2>&1; then
            python3 -m gui.updater --code-only || true
        elif [ -d ".git" ] && command -v git >/dev/null 2>&1; then
            if [ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
                git fetch --quiet origin main 2>/dev/null && git merge --ff-only origin/main 2>/dev/null || true
            fi
        fi

        # --- Step 3b: If the launcher template was updated, update root launcher & re-exec ---
        if [ -f "$TEMPLATE_LAUNCHER" ] && ! cmp -s "$TEMPLATE_LAUNCHER" "$TARGET_LAUNCHER"; then
            echo "Updating launcher script to latest version..."
            cp "$TEMPLATE_LAUNCHER" "$TARGET_LAUNCHER"
            chmod +x "$TARGET_LAUNCHER"
            OCTOLYZER_LAUNCHER_REEXEC=1 exec "$TARGET_LAUNCHER" "$@"
        fi
    fi

    # --- Step 4: Provision .venv with newest compatible Python (if missing) ---
    BINARY_PKGS="torch,torchvision,numpy,scipy,matplotlib,opencv-python,scikit-image,scikit-learn,SimpleITK,numba,llvmlite,bottleneck,PySide6,pandas,Pillow"
    LOCKFILE=""

    if [ ! -x ".venv/bin/python" ]; then
        echo "Selecting the latest Python version compatible with all dependencies..."
        SELECTED_PYTHON=""
        LOCKFILE="$(mktemp "${TMPDIR:-/tmp}/octolyzer-reqs.XXXXXX")"
        for PY_VER in 3.13 3.12 3.11; do
            if uv pip compile requirements.txt \
                --python-version "$PY_VER" \
                --only-binary "$BINARY_PKGS" \
                --quiet \
                --output-file "$LOCKFILE" >/dev/null 2>&1; then
                if uv venv .venv --python "$PY_VER" --python-preference managed; then
                    SELECTED_PYTHON="$PY_VER"
                    break
                fi
            fi
        done
        if [ -z "$SELECTED_PYTHON" ]; then
            rm -f "$LOCKFILE"
            echo "Error: Could not find a Python version compatible with all dependencies." >&2
            read -r -p "Press Enter to close..." _
            exit 1
        fi
        echo "Installing OCTolyzer dependencies for Python $SELECTED_PYTHON (this may take a few minutes on first run)..."
        if ! uv pip install --python .venv/bin/python --only-binary "$BINARY_PKGS" -r "$LOCKFILE"; then
            rm -f "$LOCKFILE"
            rm -rf .venv
            echo "Error: Failed to install dependencies." >&2
            read -r -p "Press Enter to close..." _
            exit 1
        fi
        rm -f "$LOCKFILE"
    else
        # --- Step 5: Sync dependencies into existing .venv ---
        echo "Verifying OCTolyzer dependencies..."
        uv pip install --python .venv/bin/python --only-binary "$BINARY_PKGS" --quiet -r requirements.txt || true
    fi

    # --- Step 6: Launch GUI ---
    echo "Launching OCTolyzer GUI..."
    OCTOLYZER_UPDATED_ONCE=1 exec .venv/bin/python -m gui.app "$@"
}

main "$@"
