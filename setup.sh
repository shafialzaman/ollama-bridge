#!/usr/bin/env bash
# Ollama Bridge — one-shot setup script
# Works on macOS (M5) and Linux (CUDA workstation)
# Usage: bash setup.sh

set -euo pipefail

OS="$(uname -s)"
ARCH="$(uname -m)"

echo ""
echo "  Ollama Bridge Setup"
echo "  ───────────────────────────────────────"
echo "  OS   : $OS"
echo "  Arch : $ARCH"
echo ""

# ── 1. Install Ollama if missing ──────────────────────────────────────────────
if ! command -v ollama &> /dev/null; then
  echo "  [1/4] Installing Ollama..."
  if [[ "$OS" == "Darwin" ]]; then
    if command -v brew &> /dev/null; then
      brew install ollama
    else
      echo "  Homebrew not found. Download Ollama from https://ollama.com/download"
      exit 1
    fi
  elif [[ "$OS" == "Linux" ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi
else
  echo "  [1/4] Ollama already installed: $(ollama --version 2>/dev/null || echo 'ok')"
fi

# ── 2. Start Ollama (background) ──────────────────────────────────────────────
echo "  [2/4] Starting Ollama daemon..."
if [[ "$OS" == "Darwin" ]]; then
  # macOS: run as launchd service or background process
  if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &>/tmp/ollama.log &
    sleep 2
  fi
elif [[ "$OS" == "Linux" ]]; then
  if command -v systemctl &> /dev/null; then
    sudo systemctl enable --now ollama || true
  else
    ollama serve &>/tmp/ollama.log &
    sleep 2
  fi
fi

# ── 3. Python dependencies ────────────────────────────────────────────────────
echo "  [3/4] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Dependencies installed."

# ── 4. Pull a starter model ───────────────────────────────────────────────────
echo ""
echo "  [4/4] Pulling starter model..."
if [[ "$ARCH" == "arm64" && "$OS" == "Darwin" ]]; then
  echo "  Detected Apple Silicon — pulling llama3.2:3b (fast on M-series)"
  ollama pull llama3.2:3b
else
  echo "  Detected Linux/CUDA — pulling llama3.1:8b"
  ollama pull llama3.1:8b
fi

echo ""
echo "  ✓ Setup complete."
echo ""
echo "  Start the API server :"
echo "    python server.py"
echo ""
echo "  Start the chat client :"
echo "    python client.py"
echo ""
echo "  Run sample prompts    :"
echo "    python sample_prompt.py"
echo ""
