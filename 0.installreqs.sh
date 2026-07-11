#!/usr/bin/env bash
set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: No Python virtual environment is active."
    exit 1
fi

if command -v uv &>/dev/null; then
    echo "Installing with uv..."
    uv pip install -e .
else
    echo "Installing with pip..."
    pip install -e .
fi
