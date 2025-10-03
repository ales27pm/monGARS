# Code Audit Summary — 2025-05-20

## Scope
- Executed the full `pytest` suite to surface runtime defects and confirm Ray
  Serve fallbacks (`tests/test_llm_ray.py`, `tests/test_llm_adapter_refresh.py`).
- Focused on the authentication stack and WebSocket flow where credential,
  ticket, and Fernet handling intersect (`monGARS/core/security.py`,
  `monGARS/api/ws_manager.py`).

## Findings
- `SecurityManager` initialises a `CryptContext` with `pbkdf2_sha256` and
  opportunistically enables `bcrypt` when the extension is available. Password
  hashing now works in pure-Python environments while still accepting legacy
  bcrypt hashes.
- WebSocket connections are gated by `WS_ALLOWED_ORIGINS`, `WS_ENABLE_EVENTS`, and
  signed ticket verification before history replay begins.
- Ray Serve requests respect per-endpoint circuit breakers and degrade to the
  local Ollama path when HTTP errors or scaling events occur.

## Remediation
- Documented the password hashing trade-offs inline and in the README to avoid
  regressions when future maintainers revisit algorithm choices.
- Added structured log context (`ws_manager.history_failed`, `llm.ray.*`) so
  operators can triage authentication or inference issues without enabling debug
  logging globally.

## Recommendations
- Pin `passlib` in `requirements.txt` and record backend availability in CI to
  prevent regressions.
- Keep the new Ray Serve OpenTelemetry metrics wired into dashboards and alarms
  so SDK-driven traffic inherits the same visibility as operator workflows.
- Fold SDK contract tests into the audit cycle once the client packages ship to
  ensure credential, ticket, and streaming flows remain hardened.
