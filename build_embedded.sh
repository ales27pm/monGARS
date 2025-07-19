#!/usr/bin/env bash
set -e
IMAGE_NAME=${1:-mongars-embedded}
PLATFORMS="linux/arm/v7,linux/arm64/v8"

docker buildx build \
  --platform "$PLATFORMS" \
  -f Dockerfile.embedded \
  -t "$IMAGE_NAME" \
  --load .
