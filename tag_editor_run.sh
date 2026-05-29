#!/usr/bin/env bash
# AI Tag Editor launcher (Linux / macOS)
# Mirrors tag_editor_run.bat: keeps the window open on failure so logs stay
# readable even when launched by double-click from a file manager.

# Run from the directory this script lives in, regardless of caller's cwd.
cd "$(dirname "$0")" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1

LOG_FILE="tag_editor.log"

# Hold the terminal open and tell the user where the full log is, so a crash
# is never invisible (the .bat uses `pause` for the same reason).
hold_open() {
    echo
    echo "Full log saved to: $(pwd)/${LOG_FILE}"
    # Skip the prompt in non-interactive shells (CI, pipes).
    if [ -t 0 ]; then
        read -r -p "Press Enter to close..." _
    fi
}

# Mirror everything below to the log file as well as the console.
exec > >(tee "${LOG_FILE}") 2>&1

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] '${PYTHON}' not found. Install Python 3.10+ first."
    hold_open
    exit 1
fi

echo "Setting up AI Tag Editor..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if ! "${PYTHON}" -m venv venv; then
        echo "[ERROR] Failed to create virtual environment."
        hold_open
        exit 1
    fi
fi

# shellcheck disable=SC1091
source venv/bin/activate || { echo "[ERROR] Activation failed."; hold_open; exit 1; }

# Quick check: are the core packages already importable? If so, skip the
# (slow) dependency install on subsequent launches.
if python -c "import onnxruntime, PyQt6, PIL, huggingface_hub, numpy, requests" 2>/dev/null; then
    echo "Dependencies already present."
else
    echo "Installing dependencies... (This may take a while)"
    python -m pip install --upgrade pip -q
    if ! python -m pip install -r requirements.txt; then
        echo "[ERROR] Dependency installation failed. See the log above."
        hold_open
        exit 1
    fi
fi

echo "Starting application..."
if ! python main.py; then
    echo
    echo "[ERROR] Application crashed. See the traceback above."
    hold_open
    exit 1
fi
