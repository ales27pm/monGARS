# API Layer Guidelines

The modules in this folder expose the FastAPI application consumed by the Django
frontend and peer nodes.

## Endpoints & Schemas
- Keep endpoints asynchronous and return Pydantic models defined in
  `web_api.py` (e.g. `ChatResponse`, `MemoryItem`). Use validators for payloads
  to match the strict rules enforced by `tests/test_api_chat.py` and
  `tests/test_api_history.py`.
- Centralise security through `authentication.py` and the dependency providers
  in `dependencies.py`. When you add new dependencies, register them here so
  fixtures in `tests/conftest.py` can override them.
- Protect administrative routes with `get_current_admin_user` and all other
  routes with `get_current_user`. JWT parsing must go through
  `core.security.SecurityManager`—do not duplicate the logic.
- Use `ws_manager.WebSocketManager` for websocket broadcasts. Never write to a
  `WebSocket` directly outside of that helper so connection bookkeeping remains
  consistent with `tests/test_websocket.py`.

## Error Handling & Logging
- Translate domain exceptions into `HTTPException` with meaningful status codes
  (`400` for validation failures, `403` for forbidden, `500` for unexpected
  errors). Log unexpected errors with context before propagating.
- Normalise external URLs and tokens before persisting or caching them. Follow
  the `PeerRegistration` validator example to avoid duplication.

## Testing
- Update `tests/test_api_chat.py`, `tests/test_api_history.py`,
  `tests/test_peer_communication.py`, and `tests/test_user_management.py` when
  touching related endpoints.
- Use FastAPI `TestClient` or async WebSocket clients in tests to mirror
  real-world flows. Provide stubs for heavy dependencies via fixtures or
  monkeypatching.
