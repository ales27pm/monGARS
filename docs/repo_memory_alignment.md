# monGARS Repository vs Memory Alignment

This document captures the relationship between the modules present in the
`monGARS` repository and the architectural components referenced in the
assistant's long-term project memory. Use it as an onboarding aid when mapping
historical design notes to the current codebase.

## API Layer
- `api/authentication.py` ←→ **Security model** (HMAC tokens with planned JWT
  support).
- `api/dependencies.py` ←→ **Backend scaffolding** utilities for FastAPI.
- `api/web_api.py` ←→ **Bouche** (dialogue engine), `APIService.swift`
  integration points, and browser speech clients.

## Core Services
- `core/conversation.py` ←→ **Bouche + Cortex** orchestration for dialogue
  flows.
- `core/evolution_engine.py` ←→ **Evolution Engine** and Sommeil Paradoxal
  testing loops.
- `core/llm_integration.py` ←→ **monGARS LLM bucket** (currently unpopulated in
  memory).
- `core/logging.py` ←→ **Mémoire Autobiographique** (structured logging and
  historical traceability).
- `core/monitor.py` ←→ **Sommeil Paradoxal diagnostics** and Evolution Engine
  validation checks.
- `core/caching/tiered_cache.py` ←→ **Hippocampus** (short-term cache layered on
  vector embeddings).
- `core/neuro_symbolic/advanced_reasoning.py` ←→ **Tronc** (neuro-symbolic
  curiosity engine experiments).

## Testing Infrastructure
- `tests/integration_test.py` ←→ Phase 1 validation workflows.
- `tests/self_training_test.py` ←→ Sommeil Paradoxal + Evolution Engine
  self-optimization routines.
- `tests/property_test.py` ←→ Cortex + Mimicry behaviour consistency checks.
- `tests/chaos_test.py` ←→ Robustness testing for autonomy and offline-first
  guarantees.

## Platform & Operations
- `init_db.py` ←→ **Hippocampus** semantic memory database provisioning
  (Postgres + pgvector).
- `tasks.py` ←→ **Sommeil Paradoxal** background job scheduling.
- `docker-compose.yml` ←→ Local orchestration for Cortex, Hippocampus, and
  auxiliary services.
- `Dockerfile` ←→ Baseline container specification.
- `k8s/deployment.yaml` ←→ Future distributed deployment plans.
- `k8s/prometheus.yaml` ←→ Monitoring hooks for Sommeil Paradoxal diagnostics.
- `k8s/secrets.yaml` ←→ Runtime secrets (align with planned AES-256 storage).
- `.github/workflows/ci-cd.yml` ←→ Automated validation via Evolution Engine
  pipelines.

## Related Documentation
- `monGARS_structure.txt` and `ROADMAP.md` carry historical notes referenced in
  the memory archive. Update this mapping whenever major architectural changes
  land to keep the knowledge base synchronized.
