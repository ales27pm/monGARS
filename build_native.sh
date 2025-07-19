#!/usr/bin/env bash
set -euo pipefail
# Build optimized Docker image for a high-performance Intel-based workstation.
# Usage: ./build_native.sh [IMAGE_NAME]

IMAGE_NAME=${1:-mongars-native}
JOBS=$(nproc)

docker buildx build \
  --platform linux/amd64 \
  --build-arg JOBS="$JOBS" \
  -t "$IMAGE_NAME" \
  --load .
