# monGARS Roadmap

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This roadmap reflects the verified state of the codebase and the milestones
required to reach production readiness.

## Immediate Priorities (Security & Stability)

- ✅ Align JWT algorithm with deployed secrets (HS256 enforced until managed key storage is available).
- ✅ Store runtime secrets in Vault/Sealed Secrets instead of raw `k8s/secrets.yaml`.
- ✅ Update Dockerfiles to run as non-root and add a `.dockerignore` to exclude
  secrets and build artefacts.
- ✅ Replace demo users in `web_api.py` with the database-backed authentication
  flow and migrations; bootstrap now persists accounts without shipping
  in-memory defaults.【F:monGARS/api/authentication.py†L17-L120】【F:monGARS/api/web_api.py†L41-L120】

## Phase 1 – Core Infrastructure (✅ Completed Q1 2025)

- Established core cognition modules (Cortex, Hippocampus, Neurons, Bouche).
- Delivered Docker Compose and Kubernetes manifests for local/cluster deployment.

## Phase 2 – Functional Expansion (✅ Completed Q2 2025)

- Landed Mimicry, Personality, Adaptive Response, and image captioning modules.
- Integrated Iris scraping and Curiosity Engine for external research.
- Replaced placeholder tests with meaningful coverage for circuit breakers,
  caching, and self-training.

## Phase 3 – Hardware & Performance (✅ Completed Q3 2025)

- ✅ Worker auto-tuning for Pi/Jetson (`recommended_worker_count`).
- ✅ Multi-architecture build scripts and cache metrics.
- ✅ Hardened RBAC manifests.
- ✅ Ray Serve HTTP integration with circuit breaking plus MNTP trainer support
  for LoRA and curated adapters.
- ✅ Extend Alembic migrations for the newest SQLModel tables, including legacy
  tables created outside the current ORM layer.
- ✅ Expose Ray Serve success/failure counters via OpenTelemetry (`llm.ray.*`
  metrics emitted by `LLMIntegration`).

## Phase 4 – Collaborative Networking (✅ Completed Q4 2025)

- ✅ Encrypted peer registry, admin-guarded endpoints, and distributed scheduler.
- ✅ Sommeil Paradoxal idle-time optimisation and safe apply pipeline.
- ✅ Implemented load-aware scheduling strategies and shared optimisation telemetry
  across nodes.

## Phase 5 – Web Interface & API Refinement (✅ Completed Q4 2025)

- ✅ FastAPI chat/history/token endpoints with validation.
- ✅ Django chat UI with progressive enhancement.
- ✅ FastAPI WebSocket handler with ticket verification, history replay, and
  streaming guarded by `WS_ENABLE_EVENTS`.
- ✅ Replaced hard-coded credential stores with database-backed auth flows;
  FastAPI no longer seeds demo credentials at startup.【F:monGARS/api/web_api.py†L41-L120】
- ✅ Publish polished SDKs and reference clients with documented release flows.【F:docs/sdk-release-guide.md†L1-L160】【F:docs/sdk-overview.md†L1-L120】

## Phase 6 – Self-Improvement & Research (✅ Completed Q1 2026)

- ✅ Personality profiles persisted via SQLModel with live adapter updates.
- ✅ Self-training cycles produce real adapter artefacts via
  `modules.neurons.training.mntp_trainer.MNTPTrainer` with deterministic fallbacks.
- ✅ Reinforcement-learning research loops run through the evolution
  orchestrator, operator approvals, and long-haul validator with telemetry and
  manifest updates.【F:modules/evolution_engine/orchestrator.py†L360-L440】【F:monGARS/core/long_haul_validation.py†L1-L220】
- ✅ ResearchLongHaulService now schedules multi-replica soak runs and persists
  observability snapshots for dashboards, ensuring reinforcement pipelines stay
  healthy without manual triggers.【F:monGARS/core/research_validation.py†L1-L200】【F:monGARS/core/reinforcement_observability.py†L1-L168】【F:tests/test_research_long_haul_service.py†L1-L200】【F:tests/test_long_haul_validation.py†L200-L320】

## Phase 7 – Sustainability & Longevity (🌱 Future)

- 🚧 Fully integrate evolution engine outputs into routine optimisation cycles.
- 🚧 Automate energy usage reporting and advanced hardware-aware scaling using the
  energy tracker pipeline and reinforcement observability feeds as the baseline
  data source.【F:modules/evolution_engine/energy.py†L1-L160】【F:monGARS/core/reinforcement_observability.py†L1-L168】
- 🚧 Share optimisation artefacts between nodes for faster convergence.

Review [docs/implementation_status.md](docs/implementation_status.md) for a
narrative deep dive into each phase and current discrepancies.
