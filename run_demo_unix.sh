#!/usr/bin/env bash
set -euo pipefail

# Make uv use the OS trust store (fixes TLS behind enterprise proxies)
export UV_NATIVE_TLS=true

# === Ensure ~/.local/bin is in PATH ===
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "[INFO] Adding ~/.local/bin to PATH for this session..."
    export PATH="$HOME/.local/bin:$PATH"
fi

# === Prerequisite checks ===
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is required but not found in PATH."
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "[ERROR] curl is required to install uv but was not found."
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found in $(pwd)."
    echo "        Run this script from the project root directory."
    exit 1
fi

# === Ensure uv is installed ===
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # After installation, add ~/.local/bin to PATH again (for safety)
    export PATH="$HOME/.local/bin:$PATH"

    # Try sourcing the env file if it exists
    if [ -f "$HOME/.local/bin/env" ]; then
        echo "[INFO] Sourcing uv environment..."
        source "$HOME/.local/bin/env"
    fi

    # Verify uv installation
    if ! command -v uv &> /dev/null; then
        echo "[ERROR] uv was installed but not detected in PATH."
        echo "[ERROR] Please restart your terminal and rerun this script."
        exit 1
    fi
fi

# === Create virtual environment ===
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    uv venv --python 3.12
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[ERROR] Virtual environment Python not found at .venv/bin/python."
    echo "        Remove .venv and rerun this script."
    exit 1
fi

# === Install dependencies ===
echo "[INFO] Installing dependencies..."
uv pip install --python .venv/bin/python --upgrade pip
uv pip install --python .venv/bin/python -r requirements.txt

# === Run the app ===
echo "[INFO] Starting Streamlit app..."
.venv/bin/python -m streamlit run app.py
