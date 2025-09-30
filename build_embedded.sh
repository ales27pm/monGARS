#!/usr/bin/env bash
set -euo pipefail
# Build and push a multi-architecture image for Raspberry Pi and Jetson boards.
# Usage: ./build_embedded.sh [IMAGE_NAME]

IMAGE_NAME=${1:-mongars-embedded}
PLATFORMS="linux/arm/v7,linux/arm64/v8"

# The --push flag is required for multi-platform images. Ensure you are
# authenticated to your container registry before running this script.
docker buildx build \
  --platform "$PLATFORMS" \
  --build-arg JOBS="$(nproc)" \
  -f Dockerfile.embedded \
  -t "$IMAGE_NAME" \
  --push .
