#!/usr/bin/env bash
set -e

# Change directory to script directory
cd "$(dirname "$0")"

echo "======================================================================="
echo " RNN Explainer Video - Rendering Full HD Quality (1080p, 60fps)"
echo "======================================================================="

if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

"$PYTHON_BIN" render.py --quality high "$@"

echo ""
echo "[SUCCESS] Video rendered successfully!"
