# Chat App Guidelines

This Django app wraps the FastAPI conversation endpoints and manages UI state
for operators.

## Architecture
- `views.py` should delegate HTTP calls to `services.py` and apply the
  `require_token` decorator from `decorators.py` for any authenticated routes.
- Keep forms and request payloads aligned with the FastAPI contracts in
  `monGARS.api.web_api`. For example, chat messages must respect the same length
  limits enforced by `ChatRequest`.
- When you introduce new context variables for templates, document them in the
  view docstring and update the corresponding template under
  `templates/chat/`.

## Networking & Error Handling
- `services.py` centralises calls to `/token` and `/api/v1/conversation/history`.
  Expand it when adding new API interactions so retries and logging stay
  consistent.
- Catch network exceptions and surface user-friendly error messages (see
  `fetch_history`). Log failures with enough context for operators to diagnose
  connection issues.

## Testing
- Update `tests/test_api_history.py` and `tests/test_websocket.py` when altering
  chat flows, ensuring websocket broadcasts and HTTP history retrieval remain
  in sync.
- Use asynchronous test clients or `httpx` mocking when validating new service
  functions so behaviour mirrors production.
