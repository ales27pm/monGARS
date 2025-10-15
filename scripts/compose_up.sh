#!/usr/bin/env bash
set -euo pipefail

if docker compose version >/dev/null 2>&1; then
  COMPOSE_BIN=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_BIN=(docker-compose)
else
  echo "[monGARS] docker compose plugin or docker-compose binary is required" >&2
  exit 1
fi

if [ "${DOCKER_API_VERSION:-}" = "" ]; then
  export DOCKER_API_VERSION="1.43"
fi

if [ "${COMPOSE_DOCKER_CLI_BUILD:-}" = "" ]; then
  export COMPOSE_DOCKER_CLI_BUILD=1
fi

if [ "${DOCKER_BUILDKIT:-}" = "" ]; then
  export DOCKER_BUILDKIT=0
fi

exec "${COMPOSE_BIN[@]}" "$@"
