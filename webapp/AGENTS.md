# Django Webapp Guidelines

The Django project under `webapp/` provides the operator-facing UI and bridges
to the FastAPI service.

## Views & Services
- Keep views asynchronous. `chat/views.py` already uses `async def` and awaits
  helper calls in `chat/services.py`. Any new view should follow the same
  pattern.
- Delegate outbound HTTP calls to `chat/services.py` so authentication, error
  handling, and logging remain centralised. Services rely on `httpx.AsyncClient`
  and the `FASTAPI_URL` environment variable.
- Load configuration from environment variables or Django settings; avoid hard
  coding URLs and secrets.

## Templates & Static Assets
- Templates live in `chat/templates/chat/`. Keep business logic in views or
  services, using templates for presentation only. Document context keys in view
  docstrings when you add or change template variables.
- When adding JavaScript, use progressive enhancement—ensure core flows remain
  usable without client-side scripting where feasible.

## Settings & Middleware
- Update `webapp/webapp/settings.py` when introducing new apps or middleware.
  Ensure defaults exist for any new environment variables and document them in
  `README.md`.
- Authentication decorators live in `chat/decorators.py`. Reuse `require_token`
  when protecting views instead of duplicating token parsing logic.

## Testing
- Add or update Django-side tests under `tests/test_api_history.py` and
  `tests/test_websocket.py` to reflect UI contract changes. These files already
  exercise the async HTTP bridge.
- When mocking FastAPI responses, use `respx` or `httpx` mocking utilities to
  keep async behaviour realistic.
