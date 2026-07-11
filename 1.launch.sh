#!/usr/bin/env bash
set -euo pipefail

./0.checkreqs.sh

echo "Launching Streamlit..."
exec streamlit run app.py
