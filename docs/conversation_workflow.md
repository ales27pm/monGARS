# monGARS Conversation Workflow

This document provides a step-by-step walkthrough of how monGARS turns a user chat request into a conversational response. It follows the path from authentication, through request validation and orchestration in the conversational engine, to persistence and streaming updates.

## 1. Authentication and Session Context

1. A client first exchanges credentials for a bearer token by calling `POST /token` with an `OAuth2PasswordRequestForm`. The FastAPI route validates the username/password pair against `users_db`, hashes the password with the `SecurityManager`, and signs a JWT containing the subject and `admin` claim when applicable.【F:monGARS/api/web_api.py†L38-L66】【F:monGARS/core/security.py†L18-L73】
2. For each authenticated request, FastAPI injects `get_current_user` which reconstructs a `SecurityManager` using the configured secret and algorithm, verifies the JWT, and returns the payload. Admin-only routes wrap this dependency with `get_current_admin_user` to enforce the `admin` flag.【F:monGARS/api/authentication.py†L1-L25】
3. The `/api/v1/conversation/chat` route therefore receives a validated `current_user` payload whose `sub` field becomes the logical user identifier throughout the workflow.【F:monGARS/api/web_api.py†L133-L183】

## 2. Request Validation and WebSocket Setup

1. Incoming chat requests must conform to the `ChatRequest` pydantic model which trims whitespace, enforces non-empty messages, and limits message/session lengths. This prevents overly large payloads and removes incidental formatting issues.【F:monGARS/api/web_api.py†L104-L131】
2. The message and user identifier are sanitized via `validate_user_input`. This helper strips HTML with `bleach`, confirms presence of required fields, and raises `ValueError` for missing data so that the API can return a structured error response.【F:monGARS/core/security.py†L87-L106】
3. In parallel, authenticated WebSocket clients may subscribe to `/ws/chat/` by presenting the same JWT. The server reuses the `SecurityManager` to verify the token, replays the most recent history via the shared `Hippocampus`, and registers the connection with `WebSocketManager` so that future responses can be broadcast live.【F:monGARS/api/web_api.py†L69-L103】【F:monGARS/api/ws_manager.py†L1-L34】【F:monGARS/api/dependencies.py†L1-L16】【F:monGARS/core/hippocampus.py†L1-L43】

## 3. Conversational Pipeline Orchestration

The heart of the workflow resides in `ConversationalModule.generate_response`, which composes multiple cognitive subsystems in sequence.【F:monGARS/core/conversation.py†L26-L145】  The major stages are:

1. **State Preparation**
   * A shared `Hippocampus` provides the five most recent `(query, response)` pairs for context. Memory access is serialized per user through asyncio locks to avoid race conditions.【F:monGARS/core/conversation.py†L91-L99】【F:monGARS/core/hippocampus.py†L19-L43】
   * Optional image bytes are captioned with `ImageCaptioning`. If the BLIP model is available, captions are generated asynchronously and appended to the textual query; otherwise, the module logs a warning and returns the untouched query.【F:monGARS/core/conversation.py†L53-L57】【F:monGARS/core/mains_virtuelles.py†L1-L70】

2. **Knowledge Gap Detection**
   * The `CuriosityEngine` compares the normalized query against recent conversation vectors stored in the SQL-backed `ConversationHistory`. When insufficient overlap is found, it performs entity extraction via spaCy (or a rule-based fallback), consults the knowledge graph driver, and, if necessary, triggers external document retrieval or web search via `Iris`. The resulting supplemental context is appended to the query to better inform the LLM stage.【F:monGARS/core/conversation.py†L59-L99】【F:monGARS/core/cortex/curiosity_engine.py†L1-L94】

3. **Symbolic Reasoning Augmentation**
   * The `AdvancedReasoner` inspects the refined query for interrogatives (e.g., “why”, “how”) and contributes templated explanatory hints. These hints are concatenated to the prompt that will be sent to the LLM.【F:monGARS/core/conversation.py†L68-L100】【F:monGARS/core/neuro_symbolic/advanced_reasoner.py†L1-L13】

4. **LLM Invocation with Reliability Guards**
   * `LLMIntegration` determines whether to call Ollama locally or forward to a Ray Serve deployment, governed by environment variables. Calls are wrapped in a circuit breaker and tenacity retries. Responses are cached in an async TTL cache to avoid repeated work, and multiple fallbacks return descriptive placeholder text when dependencies are unavailable, ensuring the system degrades gracefully rather than throwing errors.【F:monGARS/core/conversation.py†L100-L108】【F:monGARS/core/llm_integration.py†L23-L197】

5. **Personality-Driven Adaptation**
   * `PersonalityEngine` now collaborates with the LoRA-powered `StyleFineTuner` to build and refresh user-specific adapters. Recent interactions trigger fine-tuning on a compact causal language model, and the resulting hidden-state projections drive personality trait updates stored in `UserPersonality`. `AdaptiveResponseGenerator` reuses the same adapters to rewrite responses with the learned tone instead of relying on regex substitutions, while `MimicryModule` continues to merge long- and short-term statistics for additional contextual polish.【F:monGARS/core/conversation.py†L26-L138】【F:monGARS/core/personality.py†L1-L122】【F:monGARS/core/style_finetuning.py†L1-L326】【F:monGARS/core/dynamic_response.py†L1-L197】【F:monGARS/core/mimicry.py†L1-L104】

6. **Persistence and Memory Consolidation**
   * Structured metadata capturing the original, image-augmented, and refined prompts plus the adapted response is saved to the lightweight SQLite schema via `PersistenceRepository`. The repository writes both an `Interaction` row and a matching `ConversationHistory` entry so the `CuriosityEngine` can learn from past conversations, while the same augmented query/response pair is cached in the shared `Hippocampus` for WebSocket replays.【F:monGARS/core/conversation.py†L91-L138】【F:monGARS/core/persistence.py†L10-L31】【F:monGARS/core/hippocampus.py†L19-L43】【F:monGARS/init_db.py†L14-L114】

7. **Output Delivery**
   * `SpeakerService` currently delegates to `Bouche`, which logs and returns the final text. This abstraction makes it easy to swap in a TTS system later. The API then broadcasts the payload to all active WebSocket listeners and returns an HTTP response containing the text, reported confidence, and processing time.【F:monGARS/core/conversation.py†L26-L145】【F:monGARS/core/bouche.py†L1-L10】【F:monGARS/api/web_api.py†L166-L183】【F:monGARS/api/ws_manager.py†L16-L34】

## 4. Cross-Cutting Services

1. **Security & Sanitization** – Beyond JWT handling, `validate_user_input` ensures no raw HTML or missing fields reach the LLM layer. Peer-to-peer routes reuse the same Fernet-based encryption helpers for secure message exchange between nodes.【F:monGARS/core/security.py†L18-L106】【F:monGARS/core/peer.py†L1-L56】
2. **Adaptive Infrastructure** – The instantiated `EvolutionEngine` watches for high CPU, memory, or GPU utilization via `SystemMonitor` and can scale Kubernetes worker deployments or flush caches when required. While not triggered directly in the chat path, it provides operational resilience for the conversation service.【F:monGARS/core/conversation.py†L26-L51】【F:monGARS/core/evolution_engine.py†L1-L94】【F:monGARS/core/monitor.py†L1-L36】
3. **Configuration & Secrets** – `get_settings()` centralizes configuration, optionally fetches overrides from HashiCorp Vault, configures OpenTelemetry exporters, and enforces production requirements such as a non-empty `SECRET_KEY`. These settings feed into security, external service URLs, and optional accelerations like Ray Serve or GPU usage.【F:monGARS/config.py†L1-L149】【F:monGARS/config.py†L151-L222】

## 5. Response Lifecycle Summary

1. **Authenticate** – Obtain JWT via `/token`; FastAPI dependencies validate it on subsequent requests.
2. **Submit** – Send sanitized chat payload to `/api/v1/conversation/chat`.
3. **Orchestrate** – `ConversationalModule` aggregates memory, curiosity, reasoning, LLM output, personality adaptation, and mimicry.
4. **Persist & Notify** – The augmented query/response pair is stored in SQLite and in-memory history, then broadcast to WebSocket subscribers and returned to the caller.
5. **Observe** – Background systems continue to monitor resource usage and maintain secure peer communication channels.

This layered design keeps the chat workflow modular, resilient to missing optional dependencies, and ready for extension across distributed deployments.
