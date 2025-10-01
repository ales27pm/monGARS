#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSINFO-}" ]; then
  echo "[monGARS] This script must be run with Bash (e.g., bash scripts/deploy_docker.sh)" >&2
  exit 1
fi
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "[monGARS] Bash 4.4 or newer is required" >&2
  exit 1
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
ENV_FILE="${PROJECT_ROOT}/.env"
PROJECT_NAME=${COMPOSE_PROJECT_NAME:-mongars}

log() {
  printf '\033[1;34m[monGARS]\033[0m %s\n' "$1"
}

warn() {
  printf '\033[1;33m[monGARS]\033[0m %s\n' "$1"
}

err() {
  printf '\033[1;31m[monGARS]\033[0m %s\n' "$1" >&2
}

ensure_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required tool '$1' is not installed or not on PATH."
    exit 1
  fi
}

resolve_compose_command() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_BIN=(docker compose)
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_BIN=(docker-compose)
    return
  fi
  err "Docker Compose plugin or docker-compose binary is required."
  exit 1
}

compose() {
  "${COMPOSE_BIN[@]}" -f "$COMPOSE_FILE" --project-name "$PROJECT_NAME" "$@"
}

ensure_env_file() {
  local template="${PROJECT_ROOT}/.env.example"
  if [[ ! -f "$template" ]]; then
    err "Missing template file: $template"
    exit 1
  fi
  if [[ ! -f "$ENV_FILE" ]]; then
    cp "$template" "$ENV_FILE"
    log "Created .env from .env.example"
  fi
  python3 <<'PY' "$ENV_FILE"
import secrets
import sys
from pathlib import Path
from typing import Dict

WEAK_DEFAULTS = {
    "SECRET_KEY": {"dev-secret-change-me", ""},
    "DJANGO_SECRET_KEY": {"django-insecure-change-me", ""},
    "DB_PASSWORD": {"changeme", ""},
    "VAULT_TOKEN": {"dev-root-token"},
}

GENERATORS = {
    "SECRET_KEY": lambda: secrets.token_urlsafe(64),
    "DJANGO_SECRET_KEY": lambda: secrets.token_urlsafe(64),
    "DB_PASSWORD": lambda: secrets.token_urlsafe(24),
    "VAULT_TOKEN": lambda: secrets.token_hex(16),
}

env_path = Path(sys.argv[1])
lines = env_path.read_text().splitlines()

# Parse only simple KEY=VALUE lines; preserve others verbatim.
def parse_kv(line: str):
    if "=" not in line or line.lstrip().startswith("#"):
        return None
    key, _, value = line.partition("=")
    return key, value

values: Dict[str, str] = {}
for line in lines:
    kv = parse_kv(line)
    if kv:
        k, v = kv
        values[k] = v

updates: Dict[str, str] = {}
for key, weak_set in WEAK_DEFAULTS.items():
    current = values.get(key)
    if current is not None and current in weak_set:
        updates[key] = GENERATORS[key]()

if not updates:
    sys.exit(0)

new_lines = []
for line in lines:
    kv = parse_kv(line)
    if not kv:
        new_lines.append(line)
        continue
    key, _ = kv
    if key in updates:
        new_lines.append(f"{key}={updates[key]}")
        updates.pop(key, None)
    else:
        new_lines.append(line)

for key, val in updates.items():
    new_lines.append(f"{key}={val}")

env_path.write_text("\n".join(new_lines) + "\n")
print("Refreshed sensitive defaults in", env_path)
PY
}

usage() {
  cat <<'USAGE'
Usage: scripts/deploy_docker.sh <command> [options] [-- additional docker compose args]

Commands:
  up            Build (unless --no-build) and start the stack in the background.
  down          Stop services while preserving volumes.
  destroy       Stop services and remove volumes and orphaned containers.
  restart       Restart running services.
  logs          Tail service logs (default: all).
  ps|status     Show service status.

Options (for `up` unless noted):
  --with-ollama    Include the Ollama profile (downloads ~12GB image).
  --with-ray       Include the Ray Serve profile.
  --with-all       Enable all optional profiles.
  --pull           Run `docker compose pull` before starting.
  --no-build       Skip image rebuild during `up`.
  -h, --help       Show this help message.

Examples:
  scripts/deploy_docker.sh up --with-ollama
  scripts/deploy_docker.sh logs api
  scripts/deploy_docker.sh destroy
USAGE
}

main() {
  ensure_tool docker
  resolve_compose_command

  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  local command="$1"
  shift || true

  local profiles=()
  local pull_before=0
  local build_images=1
  local passthrough=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-ollama)
        profiles+=(inference)
        ;;
      --with-ray)
        profiles+=(ray)
        ;;
      --with-all)
        profiles+=(inference ray)
        ;;
      --profile)
        shift || { err "--profile requires an argument"; exit 1; }
        profiles+=("$1")
        ;;
      --pull)
        pull_before=1
        ;;
      --no-build)
        build_images=0
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        passthrough+=("$@")
        break
        ;;
      *)
        passthrough+=("$1")
        ;;
    esac
    shift || true
  done

  local profile_args=()
  if [[ ${#profiles[@]} -gt 0 ]]; then
    local unique_profiles=()
    while IFS= read -r -d '' profile; do
      if [[ -n "$profile" ]]; then
        unique_profiles+=("$profile")
      fi
    done < <(printf "%s\0" "${profiles[@]}" | sort -uz)

    for profile in "${unique_profiles[@]}"; do
      profile_args+=(--profile "$profile")
    done
  fi

  case "$command" in
    up)
      ensure_env_file
      if (( pull_before )); then
        log "Pulling container images"
        compose "${profile_args[@]}" pull
      fi
      local up_args=(-d)
      if (( build_images )); then
        up_args+=(--build)
      fi
      log "Starting docker-compose stack"
      compose "${profile_args[@]}" up "${up_args[@]}" "${passthrough[@]}"
      log "Services status:"
      compose "${profile_args[@]}" ps
      ;;
    down)
      log "Stopping services"
      compose down "${passthrough[@]}"
      ;;
    destroy)
      log "Removing services, volumes, and orphans"
      compose down -v --remove-orphans "${passthrough[@]}"
      ;;
    restart)
      log "Restarting services"
      compose restart "${passthrough[@]}"
      ;;
    logs)
      log "Streaming logs"
      compose logs -f "${passthrough[@]}"
      ;;
    ps|status)
      compose ps "${passthrough[@]}"
      ;;
    *)
      err "Unknown command: $command"
      usage
      exit 1
      ;;
  esac
}

main "$@"
