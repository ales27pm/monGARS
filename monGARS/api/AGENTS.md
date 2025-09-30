# API Layer Standards

Applies to everything under `monGARS/api/`, including FastAPI routers, request
schemas, WebSocket helpers, and authentication utilities.

## Endpoint Design
- Prefer asynchronous endpoints returning Pydantic response models. Validate
  inputs through `schemas.py` and avoid inline validation logic.
- Register shared dependencies in `dependencies.py` so tests can override them.
  Authentication must flow through `get_current_user` / `get_current_admin_user`
  backed by `core.security.SecurityManager`.
- Keep WebSocket interactions inside `ws_manager.WebSocketManager` to preserve
  connection bookkeeping. Do not interact with `WebSocket` instances directly in
  routes.

## Error Handling
- Translate domain-specific exceptions into `HTTPException` with precise status
  codes. Log unexpected failures with correlation identifiers before re-raising.
- Normalise external URLs, tokens, and identifiers before persisting or caching
  them. Use validators similar to `PeerRegistration` to maintain a consistent
  shape across the API.

## Security Expectations
- Require bearer tokens for every route except `/ready` and `/healthz`. Peer and
  user admin routes must enforce the admin guard.
- Sanitize user input with `core.security.validate_user_input` or equivalent
  helpers before handing payloads to the cognition layer or persistence.

## Testing Checklist
- Update `tests/test_api_chat.py`, `tests/test_api_history.py`,
  `tests/test_peer_communication.py`, and `tests/test_user_management.py` when you
  change route behaviour.
- Use FastAPI's async test clients or WebSocket sessions to mimic production
  flows. Patch heavy integrations via fixtures/monkeypatching rather than editing
  application code for testability.
