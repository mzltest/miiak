#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Activate virtualenv if it exists
if [ -d .venv ]; then
  source .venv/bin/activate
fi

# Run the Gradio app
python app.py
