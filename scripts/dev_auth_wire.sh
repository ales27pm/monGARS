#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
USERNAME="${USERNAME:-admin}"
PASSWORD="${PASSWORD:-admin}"
OPEN_BROWSER="${OPEN_BROWSER:-0}"

json_escape() {
  python3 - <<'PY'
import json,sys
print(json.dumps(sys.stdin.read()))
PY
}

request_json() {
  local method="$1"; shift
  local url="$1"; shift
  local body="$1"; shift

  curl -sS -X "${method}" "${url}" \
    -H "Content-Type: application/json" \
    --data "${body}" \
    "$@"
}

login() {
  request_json POST "${BASE_URL}/api/v1/auth/login" "$(printf '{"username":%s,"password":%s}' "$(printf '%s' "${USERNAME}" | json_escape)" "$(printf '%s' "${PASSWORD}" | json_escape)")"
}

bootstrap_admin() {
  request_json POST "${BASE_URL}/api/v1/auth/bootstrap-admin" "$(printf '{"username":%s,"password":%s}' "$(printf '%s' "${USERNAME}" | json_escape)" "$(printf '%s' "${PASSWORD}" | json_escape)")"
}

extract_token() {
  python3 - <<'PY'
import json,sys
raw=sys.stdin.read().strip()
try:
    obj=json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)
tok=obj.get("access_token") or obj.get("token") or ""
if isinstance(tok,str):
    print(tok)
else:
    print("")
PY
}

echo "[wire] BASE_URL=${BASE_URL}"
echo "[wire] USERNAME=${USERNAME}"

resp="$(login || true)"
tok="$(printf '%s' "${resp}" | extract_token)"

if [[ -z "${tok}" ]]; then
  echo "[wire] Login failed or no token returned. Attempting to create default admin (only works if no admin exists)..."
  bootstrap_admin || true
  resp="$(login || true)"
  tok="$(printf '%s' "${resp}" | extract_token)"
fi

if [[ -z "${tok}" ]]; then
  echo "[wire] Could not obtain a token."
  echo "[wire] Response was:"
  echo "${resp}"
  exit 1
fi

echo
echo "[wire] ✅ JWT obtained."
echo
echo "Option A (recommended): open this URL once (it will self-scrub the token from the URL):"
echo "  ${BASE_URL}/?jwt=${tok}"
echo
echo "Option B: paste this in the browser console:"
echo "  localStorage.setItem('mongars_jwt', '${tok}'); location.reload();"
echo

if [[ "${OPEN_BROWSER}" == "1" ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${BASE_URL}/?jwt=${tok}" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "${BASE_URL}/?jwt=${tok}" >/dev/null 2>&1 || true
  fi
fi
