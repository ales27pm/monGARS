# Model Loading & Inference Code Locations

This reference pinpoints the files that orchestrate model configuration, provisioning, and runtime inference so operators can jump directly to the relevant logic when debugging or extending the stack.

## Model Definitions Manifest
- `configs/llm_models.json` lists the logical roles used by the runtime (for example `general` and `coding`), the provider to load them from, and default generation parameters merged into every request.【F:configs/llm_models.json†L1-L28】

## Model Provisioning Manager
- `LLMModelManager` (in `monGARS/core/model_manager.py`) loads profile manifests, resolves overrides, and exposes helpers for retrieving model metadata.【F:monGARS/core/model_manager.py†L136-L210】  The manager caches successful provisioning attempts so repeated calls do not block on redundant downloads.【F:monGARS/core/model_manager.py†L211-L238】
- `_ensure_provider` delegates to provider-specific installers. The Ollama path lists existing weights and pulls missing ones via `asyncio.to_thread(ollama.pull, definition.name)` when auto-download is enabled, returning structured status entries for observability.【F:monGARS/core/model_manager.py†L244-L344】

## Inference Entrypoint
- `LLMIntegration.generate_response` coordinates cache lookups, optional Ray Serve dispatch, and local inference, falling back to the local provider when Ray fails and recording the source of the final response.【F:monGARS/core/llm_integration.py†L779-L885】
- `_call_local_provider` ensures the requested roles are provisioned, logs the dispatch, and either calls `_ollama_call` or activates the slot-based fallback if Ollama is missing or errors out.【F:monGARS/core/llm_integration.py†L609-L698】
- `_ollama_call` wraps `ollama.chat` with retry and circuit breaker protections so transient errors do not surface to users without multiple attempts.【F:monGARS/core/llm_integration.py†L567-L607】
- `_slot_model_fallback` and `_generate_with_model_slot` bridge to the Unsloth-backed PyTorch slot manager, translating Ollama-style options into Hugging Face generation arguments when a local fallback is required.【F:monGARS/core/llm_integration.py†L700-L760】

## Conversation Driver
- `ConversationalModule.generate_response` composes the final prompt (history, curiosity hints, semantic recall), invokes `LLMIntegration.generate_response`, adapts the reply to personality traits, and persists the full interaction payload.【F:monGARS/core/conversation.py†L241-L317】

## Task Invocation
- The Celery task `process_interaction` wraps the conversational module so background workers share long-lived cognition components while serving web requests dispatched from the chat frontend.【F:tasks.py†L1-L26】
