#!/usr/bin/env bash
# Double-clickable macOS / Linux launcher for OCTolyzer.
# Automatically installs uv, provisions a virtual environment with the latest
# Python, updates OCTolyzer and its dependencies, and starts `python -m gui.app`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$SCRIPT_DIR")" = "launchers" ] && [ "$(basename "$(dirname "$SCRIPT_DIR")")" = "gui" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    REPO_ROOT="$SCRIPT_DIR"
fi
cd "$REPO_ROOT"

if [ "$(uname -s)" = "Darwin" ]; then
    TARGET_LAUNCHER="$REPO_ROOT/OCTolyzer.command"
else
    TARGET_LAUNCHER="$REPO_ROOT/OCTolyzer.sh"
fi
if [ ! -f "$TARGET_LAUNCHER" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    cp "${BASH_SOURCE[0]}" "$TARGET_LAUNCHER"
    chmod +x "$TARGET_LAUNCHER"
fi

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

if [ ! -x ".venv/bin/python" ]; then
    echo "Selecting the latest Python version compatible with all dependencies..."
    SELECTED_PYTHON=""
    LOCKFILE="$(mktemp "${TMPDIR:-/tmp}/octolyzer-reqs.XXXXXX")"
    BINARY_PKGS="torch,torchvision,numpy,scipy,matplotlib,opencv-python,scikit-image,scikit-learn,SimpleITK,numba,llvmlite,bottleneck,PySide6,pandas,Pillow"
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
fi

echo "Checking for OCTolyzer updates..."
.venv/bin/python -m gui.updater || true

echo "Launching OCTolyzer GUI..."
OCTOLYZER_UPDATED_ONCE=1 exec .venv/bin/python -m gui.app "$@"
