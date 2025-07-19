#!/usr/bin/env bash
set -euo pipefail
# Simple helper to install Python dependencies.
# Usage: ./setup.sh

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found" >&2
  exit 1
fi

python3 -m pip install -r requirements.txt
