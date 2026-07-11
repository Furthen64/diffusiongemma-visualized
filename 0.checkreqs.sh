#!/usr/bin/env bash
set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: No Python virtual environment is active."
    echo "Activate one first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    exit 1
fi

echo "OK: venv active at $VIRTUAL_ENV"

if command -v uv &>/dev/null; then
    echo "OK: uv found at $(which uv)"
else
    echo "OK: uv not found, will use pip"
fi

python -c "import sys; print(f'Python {sys.version}')"
