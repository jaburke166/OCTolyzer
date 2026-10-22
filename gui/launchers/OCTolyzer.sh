#!/usr/bin/env bash
# Double-clickable Linux launcher for OCTolyzer.
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

TARGET_LAUNCHER="$REPO_ROOT/OCTolyzer.sh"
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
    echo "Creating Python environment with the latest Python..."
    uv venv .venv --python-preference managed
    echo "Installing OCTolyzer dependencies (this may take a few minutes on first run)..."
    uv pip install --python .venv/bin/python -r requirements.txt
fi

echo "Checking for OCTolyzer updates..."
.venv/bin/python -m gui.updater || true

echo "Launching OCTolyzer GUI..."
OCTOLYZER_UPDATED_ONCE=1 exec .venv/bin/python -m gui.app "$@"
