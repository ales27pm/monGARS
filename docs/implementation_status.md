# Implementation Status Overview

_Last updated: 2025-03_

This report reconciles the roadmap with the current codebase. Each phase notes
what has shipped, what remains, and any discrepancies between historical plans
and reality.

## Phase 1 – Core Infrastructure (Completed Q1 2025)
- Cortex, Hippocampus, Neurons, and Bouche modules exist with concrete
  orchestrations (`monGARS/core/conversation.py`, `monGARS/core/bouche.py`).
- Memory stack combines in-memory Hippocampus with SQLModel persistence and
  thread-safe locks.
- Docker Compose and Kubernetes manifests provision FastAPI, Postgres, Redis,
  MLflow, Vault, and Ollama for local and cluster deployment.

## Phase 2 – Functional Expansion (Completed Q2 2025)
- Mimicry, Personality, and Adaptive Response engines collaborate to personalise
  chat output. Style fine-tuning now updates LoRA adapters per user session.
- Mains Virtuelles image captioning uses BLIP with guarded imports so CPU-only
  deployments remain operational.
- Iris scraping and Curiosity Engine perform asynchronous retrieval with
  fallbacks to document ingestion when the retrieval service is unavailable.
- Contrary to the original roadmap, tests are no longer placeholders: `chaos_test.py`,
  `property_test.py`, and `self_training_test.py` exercise circuit breakers,
  tiered caching, and self-training workflows.
- `LLMIntegration` issues real Ollama requests with caching, retries, and an
  optional Ray Serve path (still stubbed, see Phase 3).

## Phase 3 – Hardware & Performance Optimisation (In Progress – Target Q3 2025)
- ✅ Worker auto-tuning for Raspberry Pi/Jetson via
  `monGARS.utils.hardware.recommended_worker_count()`.
- ✅ Multi-architecture build scripts (`build_embedded.sh`, `build_native.sh`) and
  corresponding Dockerfiles.
- ✅ Tiered cache metrics exported through OpenTelemetry.
- ✅ Hardened Kubernetes RBAC manifests.
- 🔄 Outstanding: real Ray Serve integration (current implementation logs intent
  but falls back to local inference) and full MNTP training loop (persists
  placeholders today).
- 🔄 Outstanding: version-pin container images in `docker-compose.yml` and expand
  Alembic migrations for newly introduced tables.

## Phase 4 – Collaborative Networking (In Progress – Target Q4 2025)
- Peer registry, encrypted messaging, and admin-guarded endpoints are live.
- DistributedScheduler and Sommeil Paradoxal coordinate idle-time optimisation
  and background jobs.
- Safe optimisation wrappers prevent destructive upgrades by executing changes in
  a sandbox.
- Smarter load-aware scheduling now prioritises lower-risk peers using shared telemetry, and
  nodes broadcast scheduler metrics for collaborative optimisation.

## Phase 5 – Web Interface & API Refinement (Target Q1 2026)
- FastAPI routes for `/token`, `/api/v1/conversation/chat`, `/api/v1/conversation/history`,
  and peer management are implemented with Pydantic validation.
- Django chat UI renders progressive templates and attempts to open a WebSocket
  connection; the backend WebSocket handler still needs completion to match the
  frontend expectations.
- Planned work: consolidate validation rules, replace demo credential stores with
  the database-backed auth flow, and publish polished client SDKs.

## Phase 6 – Self-Improvement & Research (Target Q2 2026)
- ✅ Personality profiles persist via SQLModel and reload into memory-backed
  caches on demand. When new conversations arrive the PersonalityEngine asks the
  style fine-tuning adapters for a fresh analysis, applies the deltas, and
  requeues a persistence task so database rows stay aligned with the latest
  LoRA fingerprints.
- ✅ SelfTrainingEngine captures dataset catalogue versions, curated record
  counts, and training summaries for each simulated run. The trainer stub still
  exercises placeholder loops, but every cycle now emits traceable metadata so a
  real MNTP implementation can consume the same manifests without schema
  changes.
- 🚧 Reinforcement learning experiments remain future work. Open design notes in
  `docs/advanced_fine_tuning.md` describe candidate reward signals and replay
  buffers, but no executable pipeline exists yet.
- ⚠️ Testing coverage for cognition and scheduling modules is substantive, while
  WebSocket fan-out and hardware utility helpers still rely on smoke tests.
  Expand targeted async WebSocket suites and Raspberry Pi/Jetson fakes before
  declaring the phase complete.

## Phase 7 – Sustainability & Longevity (Future)
- Evolution Engine orchestrates diagnostics and safe optimisation cycles but the
  underlying training remains placeholder-heavy.
- Energy-usage reporting, advanced hardware-aware scaling, and cross-node sharing
  of optimisation artefacts are open research topics.

## Key Contradictions & Actions
- **Testing maturity**: Update the roadmap to reflect that chaos/property/self-training
  tests are substantive, not placeholders.
- **LLM2Vec training**: Replace simulated MNTP trainer output with real masked
  next-token training and manifest updates.
- **Ray Serve**: Wire actual HTTP requests and replica rollouts in `LLMIntegration`
  before advertising the integration as complete.
- **WebSockets**: Implement the backend handler to satisfy the Django frontend’s
  connection attempts.
