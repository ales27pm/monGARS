# Conversation Workflow

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This guide walks through how a chat request flows from authentication to
streamed delivery inside monGARS. Use it when debugging the cognition pipeline
or onboarding new contributors.

## 1. Authenticate & Contextualise
1. Clients obtain a JWT via `POST /token` using the OAuth2 password flow. The
   route validates credentials with `SecurityManager`, hashes passwords, and
   signs a JWT containing the subject and admin flag.
2. Subsequent requests depend on FastAPI dependencies:
   - `get_current_user` validates the JWT and returns the payload.
   - `get_current_admin_user` wraps the dependency for admin-only endpoints.
3. The chat endpoint (`/api/v1/conversation/chat`) receives the validated user ID
   and applies it throughout the pipeline for caching, memory access, and
   persistence.

## 2. Validate Inputs & Prepare Channels
1. Payloads are parsed through `ChatRequest`, trimming whitespace, enforcing
   length bounds, and rejecting empty messages.
2. Inputs are sanitised with `validate_user_input`, stripping HTML and missing
   fields before the data reaches LLM integrations.
3. WebSocket subscribers first call `POST /api/v1/auth/ws/ticket` with their JWT
   to obtain a short-lived, signed ticket. They then connect to
   `/ws/chat/?t=<ticket>`; the server verifies the signature with
   `verify_ws_ticket`, enforces the `WS_ALLOWED_ORIGINS` allow-list, and denies
   access outright when `WS_ENABLE_EVENTS` is `false`. After acceptance the
   manager replays recent Hippocampus history and registers the connection for
   downstream broadcasts.

## 3. Orchestrate Cognition
`ConversationalModule.generate_response` coordinates the following stages:
1. **State preparation** – Hippocampus fetches the most recent interactions,
   guarded by per-user asyncio locks. Optional images are captioned via BLIP and
   appended to the query when available.
2. **Curiosity pass** – Curiosity Engine compares the normalised query to recent
   conversation vectors. If knowledge gaps appear, it triggers entity extraction
   and document retrieval (internal service or Iris fallback).
3. **Reasoning hints** – The neuro-symbolic advanced reasoner adds structured
   hints for “why/how” prompts before invoking the LLM.
4. **LLM invocation** – `LLMIntegration` selects Ollama or Ray Serve, wraps calls
   in circuit breakers and retries, rotates endpoints, and caches responses.
   Graceful fallbacks keep the system responsive when optional dependencies are
   missing.
5. **Personality & mimicry** – PersonalityEngine collaborates with the LoRA-based
   StyleFineTuner and MimicryModule to adapt responses using recent interaction
   statistics and stored personality profiles.
6. **Persistence** – `PersistenceRepository` writes structured metadata to the
   SQL store while Hippocampus caches the latest exchange for rapid replay.
7. **Delivery** – The adapted response is returned via HTTP and broadcast to all
   active WebSocket subscribers via `WebSocketManager`'s per-connection queues.

## 4. Cross-Cutting Concerns
- **Security** – Peer messaging and chat routes reuse the same cryptographic
  helpers for JWT validation, Fernet encryption, and sanitisation.
- **Observability** – Structured logs and OpenTelemetry counters mark the start
  and end of each stage (`llm.*`, `curiosity.*`, `evolution_engine.*`).
- **Idle optimisation** – Sommeil Paradoxal monitors the distributed scheduler
  queue and triggers evolution engine cycles when the system is idle.

## 5. Event Bus Integration
The UI and background services consume a typed event bus defined in
`core/ui_events.py`.
- `chat.message` events mirror HTTP responses and power live updates in the
  Django interface.
- `ai_model.response_chunk` / `ai_model.response_complete` events stream LLM
  output without polling.
- `ws.connected`, `history.snapshot`, and `performance.alert` events surface
  connection state, history snapshots, and scheduler warnings for operator
  dashboards.
- `evolution_engine.*` and `sleep_time_compute.*` events keep dashboards informed
  about background optimisation and maintenance cycles.

This layered workflow keeps monGARS resilient: optional modules degrade
gracefully, observability remains rich, and every chat request leaves an auditable
trace across memory, persistence, and streaming channels.
