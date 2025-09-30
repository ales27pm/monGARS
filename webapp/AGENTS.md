# Django Webapp Standards

These rules cover the Django project in `webapp/`, including async views,
services, templates, and middleware.

## Views & Services
- Keep views asynchronous (`async def`) and delegate outbound HTTP calls to
  `chat/services.py` so authentication, retries, and logging stay centralised.
- Load configuration from environment variables or Django settings; avoid
  hardcoding URLs and secrets.
- Apply the `require_token` decorator for any authenticated route and document
  context variables in view docstrings.

## Templates & Frontend Behaviour
- Keep business logic in views/services. Templates under `chat/templates/chat/`
  should focus on presentation and progressive enhancement.
- When adding JavaScript, ensure graceful degradation for non-JS clients.
  Document any new data attributes or events consumed by scripts.

## Settings & Middleware
- Document new settings in `README.md` and provide safe defaults in
  `webapp/webapp/settings.py`.
- Reuse existing middleware and authentication helpers instead of duplicating
  token parsing logic.

## Testing
- Update `tests/test_api_history.py` and `tests/test_websocket.py` when UI flows
  change. Mock FastAPI interactions with `respx`/`httpx` utilities.
