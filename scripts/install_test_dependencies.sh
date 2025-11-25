#!/usr/bin/env bash
set -euo pipefail

echo "Installing monGARS runtime and test dependencies..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -r "$SCRIPT_DIR/../requirements.txt"
