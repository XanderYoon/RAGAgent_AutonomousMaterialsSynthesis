#!/usr/bin/env bash
set -e

# === Make uv use the OS trust store (enterprise proxy friendly) ===
export UV_NATIVE_TLS=true

# === Ensure ~/.local/bin is in PATH ===
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "[INFO] Adding ~/.local/bin to PATH for this session..."
    export PATH="$HOME/.local/bin:$PATH"
fi

# === Ensure uv is installed ===
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.local/bin:$PATH"

    if [ -f "$HOME/.local/bin/env" ]; then
        echo "[INFO] Sourcing uv environment..."
        source "$HOME/.local/bin/env"
    fi

    if ! command -v uv &> /dev/null; then
        echo "[ERROR] uv installed but not available in PATH."
        echo "        Please restart your terminal and rerun."
        exit 1
    fi
fi

# === Ensure Node + npx exist ===
if ! command -v npx &> /dev/null; then
    echo "[ERROR] npx not found."
    echo "        Please install Node.js (https://nodejs.org)"
    exit 1
fi

# === Create virtual environment ===
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    uv venv --python 3.12
fi

# === Activate virtual environment ===
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# === Install Python dependencies ===
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing Python dependencies..."
    uv pip install -r requirements.txt
else
    echo "[WARN] requirements.txt not found, skipping install."
fi

# === Clean shutdown on Ctrl+C ===
trap "echo '[INFO] Shutting down...'; kill 0" SIGINT SIGTERM

# === Start MCP server ===
echo "[INFO] Starting MCP server..."
python3 -c "from RAG.mcp.app import create_app; create_app().run(transport='http', host='127.0.0.1', port=8000)" &

# === Wait for MCP server ===
echo "[INFO] Waiting for MCP server on port 8000..."
until nc -z 127.0.0.1 8000; do
    sleep 0.3
done

echo "[INFO] MCP server is up."

INSPECTOR_UI_URL="http://localhost:6274/"
MCP_SERVER_URL="http://127.0.0.1:8000/mcp/"

echo
echo "=================================================="
echo "🧪 MCP Inspector connection info"
echo "--------------------------------------------------"
echo "Steps:"
echo "  1) Open the Inspector UI in your browser:"
echo "     $INSPECTOR_UI_URL"
echo
echo "  2) In the UI, set:"
echo "     • Transport Type: Streamable HTTP"
echo "     • URL: $MCP_SERVER_URL"
echo "     • Connection type: Via Proxy"
echo "     • Press 'Connect'"
echo "=================================================="
echo
wait
