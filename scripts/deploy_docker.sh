#!/usr/bin/env bash
set -euo pipefail

# Require Bash 4.4+
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
# Keep the default project name aligned with the docker-compose network alias
# so that the generated network name matches the expectation inside
# docker-compose.yml. Users can still override COMPOSE_PROJECT_NAME in their
# environment.
PROJECT_NAME=${COMPOSE_PROJECT_NAME:-mongars-1}

# Colors (only if stdout is a TTY)
if [[ -t 1 ]]; then
  BLUE=$'\033[1;34m'; YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; NC=$'\033[0m'
else
  BLUE=""; YELLOW=""; RED=""; NC=""
fi

log()  { printf '%s[monGARS]%s %s\n' "${BLUE}"   "${NC}" "$1"; }
warn() { printf '%s[monGARS]%s %s\n' "${YELLOW}" "${NC}" "$1"; }
err()  { printf '%s[monGARS]%s %s\n' "${RED}"    "${NC}" "$1" >&2; }

update_env_var() {
  local key="$1"
  local value="$2"
  python3 - "$ENV_FILE" "$key" "$value" <<'PY'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]

if env_path.exists():
    lines = env_path.read_text().splitlines()
else:
    lines = []

updated = False
for idx, raw in enumerate(lines):
    stripped = raw.strip()
    if not stripped or stripped.startswith('#') or '=' not in raw:
        continue
    name, _, _ = raw.partition('=')
    if name.strip() == key:
        lines[idx] = f"{key}={value}"
        updated = True
        break

if not updated:
    lines.append(f"{key}={value}")

env_path.write_text("\n".join(lines) + "\n")
PY
}

read_env_var_with_source() {
  local key="$1"
  local default_value="$2"
  python3 - "$ENV_FILE" "$key" "$default_value" <<'PY'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
key = sys.argv[2]
default_value = sys.argv[3]

found = False
value = default_value

if env_path.exists():
    for raw in env_path.read_text().splitlines():
        if '=' not in raw:
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        name, _, rest = raw.partition('=')
        if name.strip() == key:
            value = rest.strip()
            found = True
            break

print(f"{value}::{int(found)}")
PY
}

port_is_available() {
  local port="$1"
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sys.exit(1)

sys.exit(0)
PY
}

RESERVED_PORTS=()

port_is_reserved() {
  local port="$1"
  for reserved in "${RESERVED_PORTS[@]}"; do
    if [[ "$reserved" == "$port" ]]; then
      return 0
    fi
  done
  return 1
}

mark_port_reserved() {
  local port="$1"
  if ! port_is_reserved "$port"; then
    RESERVED_PORTS+=("$port")
  fi
}

port_owned_by_project() {
  local port="$1"
  python3 - "$PROJECT_NAME" "$port" <<'PY'
import json
import subprocess
import sys

project = sys.argv[1]
port = sys.argv[2]

try:
    output = subprocess.check_output(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{json .}}",
        ],
        text=True,
    )
except subprocess.CalledProcessError:
    sys.exit(1)

target = str(int(port))

for line in output.splitlines():
    if not line.strip():
        continue
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        continue
    ports = data.get("Ports") or ""
    for chunk in ports.split(","):
        chunk = chunk.strip()
        if "->" not in chunk:
            continue
        host_part = chunk.split("->", 1)[0].strip()
        if not host_part:
            continue
        # Entries can appear as "0.0.0.0:8000" or ":::8000"
        host_port = host_part.rsplit(":", 1)[-1]
        host_port = host_port.split("-", 1)[0]
        if host_port == target:
            sys.exit(0)

sys.exit(1)
PY
}


ensure_port_available() {
  local key="$1"
  local default_value="$2"
  local description="$3"

  local source=""
  local candidate=""

  if [[ -n "${!key-}" ]]; then
    candidate="${!key}"
    source="env"
  else
    local result
    result=$(read_env_var_with_source "$key" "$default_value")
    local value_part=${result%%::*}
    local found_flag=${result##*::}
    candidate="$value_part"
    if [[ "$found_flag" == "1" ]]; then
      source="file"
    else
      source="default"
    fi
  fi

  if [[ -z "$candidate" ]]; then
    candidate="$default_value"
    source="default"
  fi

  if [[ ! "$candidate" =~ ^[0-9]+$ ]]; then
    err "${description}: invalid port '$candidate' for $key"
    exit 1
  fi

  if (( candidate < 1 || candidate > 65535 )); then
    err "${description}: port '$candidate' for $key is out of valid range (1-65535)"
    exit 1
  fi

  local original="$candidate"
  local max_port=65535

  if (( candidate < 1 )); then
    candidate=1
  fi

  while (( candidate <= max_port )); do
    if port_is_reserved "$candidate"; then
      ((candidate++))
      continue
    fi
    if port_is_available "$candidate"; then
      break
    fi
    if port_owned_by_project "$candidate"; then
      break
    fi
    ((candidate++))
  done

  if (( candidate > max_port )); then
    err "${description}: unable to find an available port starting from $original"
    exit 1
  fi

  if [[ "$candidate" != "$original" ]]; then
    if [[ "$source" == "env" ]]; then
      warn "${description}: environment override ${key}=$original is busy; using $candidate for this run"
    else
      warn "${description}: port $original is busy; updated to $candidate"
      update_env_var "$key" "$candidate"
    fi
  else
    if [[ "$source" == "default" ]]; then
      update_env_var "$key" "$candidate"
    fi
  fi

  mark_port_reserved "$candidate"
  printf -v "$key" '%s' "$candidate"
  export "$key"
}

synchronise_ws_allowed_origins() {
  local api_port="$1"
  local webapp_port="$2"

  local override="${WS_ALLOWED_ORIGINS-__UNSET__}"

  local updated
  updated=$(python3 - "$ENV_FILE" "$api_port" "$webapp_port" "$override" <<'PY'
import json
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
api_port = int(sys.argv[2])
webapp_port = int(sys.argv[3])
override = sys.argv[4]

def load_current() -> str:
    if override != "__UNSET__":
        return override
    if env_path.exists():
        for raw in env_path.read_text().splitlines():
            if '=' not in raw:
                continue
            stripped = raw.strip()
            if not stripped or stripped.startswith('#'):
                continue
            name, _, value = raw.partition('=')
            if name.strip() == 'WS_ALLOWED_ORIGINS':
                return value.strip()
    return '["http://localhost:8000","http://localhost:8001"]'

def normalise(value: str):
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        pass
    value = value.strip().strip('[]')
    if not value:
        return []
    parts = []
    for chunk in value.split(','):
        chunk = chunk.strip().strip("\"'")
        if chunk:
            parts.append(chunk)
    return parts

current_raw = load_current()
entries = normalise(current_raw)

def ensure(entry: str):
    if entry not in entries:
        entries.append(entry)

ensure(f"http://localhost:{api_port}")
ensure(f"http://127.0.0.1:{api_port}")
ensure(f"http://localhost:{webapp_port}")
ensure(f"http://127.0.0.1:{webapp_port}")

result = json.dumps(entries)

if override == "__UNSET__":
    if env_path.exists():
        lines = env_path.read_text().splitlines()
    else:
        lines = []
    updated = False
    for idx, raw in enumerate(lines):
        if '=' not in raw:
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        name, _, _ = raw.partition('=')
        if name.strip() == 'WS_ALLOWED_ORIGINS':
            if lines[idx].strip() != f"WS_ALLOWED_ORIGINS={result}":
                lines[idx] = f"WS_ALLOWED_ORIGINS={result}"
            updated = True
            break
    if not updated:
        lines.append(f"WS_ALLOWED_ORIGINS={result}")
    env_path.write_text("\n".join(lines) + "\n")

print(result)
PY
) || {
    err "Failed to calculate WS_ALLOWED_ORIGINS";
    exit 1;
  }

  export WS_ALLOWED_ORIGINS="$updated"
}

prepare_ports() {
  local enable_inference="$1"
  local enable_ray="$2"

  RESERVED_PORTS=()

  local api_port
  ensure_port_available "API_PORT" 8000 "API service"
  api_port="$API_PORT"

  local webapp_port
  ensure_port_available "WEBAPP_PORT" 8001 "Django webapp"
  webapp_port="$WEBAPP_PORT"

  local postgres_port
  ensure_port_available "POSTGRES_PORT" 5432 "Postgres database"
  postgres_port="$POSTGRES_PORT"

  local redis_port
  ensure_port_available "REDIS_PORT" 6379 "Redis cache"
  redis_port="$REDIS_PORT"

  local mlflow_port
  ensure_port_available "MLFLOW_PORT" 5000 "MLflow server"
  mlflow_port="$MLFLOW_PORT"

  local vault_port
  ensure_port_available "VAULT_PORT" 8200 "Vault server"
  vault_port="$VAULT_PORT"

  local ollama_port=""
  if (( enable_inference )); then
    ensure_port_available "OLLAMA_PORT" 11434 "Ollama service"
    ollama_port="$OLLAMA_PORT"
  fi

  local ray_http_port=""
  local ray_dashboard_port=""
  local ray_client_port=""
  if (( enable_ray )); then
    ensure_port_available "RAY_HTTP_PORT" 8000 "Ray Serve HTTP"
    ray_http_port="$RAY_HTTP_PORT"
    ensure_port_available "RAY_DASHBOARD_PORT" 8265 "Ray dashboard"
    ray_dashboard_port="$RAY_DASHBOARD_PORT"
    ensure_port_available "RAY_CLIENT_PORT" 10001 "Ray client"
    ray_client_port="$RAY_CLIENT_PORT"
  fi

  synchronise_ws_allowed_origins "$api_port" "$webapp_port"

  local summary="API=${api_port}, Webapp=${webapp_port}, Postgres=${postgres_port}, Redis=${redis_port}, MLflow=${mlflow_port}, Vault=${vault_port}"
  if [[ -n "$ollama_port" ]]; then
    summary+="; Ollama=${ollama_port}"
  fi
  if [[ -n "$ray_http_port" ]]; then
    summary+="; Ray HTTP=${ray_http_port}, Ray dashboard=${ray_dashboard_port}, Ray client=${ray_client_port}"
  fi
  log "Host ports resolved: ${summary}"
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

  # Refresh weak defaults using inline Python
  python3 - "$ENV_FILE" <<'PY'
import secrets, sys
from pathlib import Path
from typing import Dict, Optional

env_path = Path(sys.argv[1])
text = env_path.read_text()
lines = text.splitlines()

def parse_kv(line: str):
  if "=" not in line or line.lstrip().startswith("#"):
    return None
  key, _, value = line.partition("=")
  return key.strip(), value

WEAK_DEFAULTS = {
  "SECRET_KEY": {"dev-secret-change-me", ""},
  "DJANGO_SECRET_KEY": {"django-insecure-change-me", ""},
  "DB_PASSWORD": {"changeme", ""},
  "VAULT_TOKEN": {"dev-root-token", ""},
}

GENERATORS = {
  "SECRET_KEY":       lambda: secrets.token_urlsafe(64),
  "DJANGO_SECRET_KEY":lambda: secrets.token_urlsafe(64),
  "DB_PASSWORD":      lambda: secrets.token_urlsafe(24),
  "VAULT_TOKEN":      lambda: secrets.token_hex(16),
}

values: Dict[str,str] = {}
for line in lines:
  kv = parse_kv(line)
  if kv:
    k, v = kv
    values[k] = v

updates: Dict[str,str] = {}
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
  k, _ = kv
  if k in updates:
    new_lines.append(f"{k}={updates.pop(k)}")
  else:
    new_lines.append(line)

for k, v in updates.items():
  new_lines.append(f"{k}={v}")

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

  local command="$1"; shift || true
  local profiles=()
  local pull_before=0
  local build_images=1
  local passthrough=()

  local enable_inference=0
  local enable_ray=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-ollama)
        profiles+=(inference)
        enable_inference=1
        ;;
      --with-ray)
        profiles+=(ray)
        enable_ray=1
        ;;
      --with-all)
        profiles+=(inference ray)
        enable_inference=1
        enable_ray=1
        ;;
      --profile)
        shift || { err "--profile requires an argument"; exit 1; }
        profiles+=("$1")
        case "$1" in
          inference) enable_inference=1 ;;
          ray)       enable_ray=1 ;;
        esac
        ;;
      --pull)     pull_before=1 ;;
      --no-build) build_images=0 ;;
      -h|--help)  usage; exit 0 ;;
      --) shift; passthrough+=("$@"); break ;;
      *)  passthrough+=("$1") ;;
    esac
    shift || true
  done

  local profile_args=()
  if [[ ${#profiles[@]} -gt 0 ]]; then
    local unique_profiles=()
    while IFS= read -r -d '' p; do
      [[ -n "$p" ]] && unique_profiles+=("$p")
    done < <(printf "%s\0" "${profiles[@]}" | sort -uz)
    for p in "${unique_profiles[@]}"; do
      profile_args+=(--profile "$p")
    done
  fi

  if [[ -n "${COMPOSE_PROFILES-}" ]]; then
    local normalized_profiles="${COMPOSE_PROFILES//,/ }"
    for token in $normalized_profiles; do
      case "$token" in
        inference) enable_inference=1 ;;
        ray)       enable_ray=1 ;;
      esac
    done
  fi

  case "$command" in
    up)
      ensure_env_file
      prepare_ports "$enable_inference" "$enable_ray"
      if (( pull_before )); then
        log "Pulling container images"
        compose "${profile_args[@]}" pull
      fi
      local up_args=(-d)
      if (( build_images )); then up_args+=(--build); fi
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
