# Codebase Status Report

> **Last updated:** 2025-10-13 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

## Purpose

This document captures the verified state of the monGARS repository as of the
current audit. It cross-references runtime modules, optional research tooling,
tests, and operations assets so the roadmap can be reconciled with concrete
implementation details.

## Runtime & API Surface

- **FastAPI application** – `monGARS/api/web_api.py` exposes authentication,
  chat, conversation history, and peer-management endpoints with typed
  responses and dependency-injected services.【F:monGARS/api/web_api.py†L63-L200】【F:monGARS/api/web_api.py†L203-L331】
- **Bootstrap users** – demo credentials were removed; the FastAPI lifespan now
  validates overrides without seeding default accounts so deployments rely on
  persisted records exclusively.【F:monGARS/api/web_api.py†L41-L88】
- **WebSocket streaming** – `monGARS/api/ws_manager.py` enforces ticket
  verification, manages per-user rate limiting, and fans out UI events through
  a token-bucket protected broadcaster.【F:monGARS/api/ws_manager.py†L1-L144】【F:monGARS/api/ws_manager.py†L145-L250】
- **RAG, ticketing, and model-management routers** extend the API surface with
  review context enrichment and administrative provisioning workflows, all wired
  through dependency providers declared in `monGARS/api/dependencies.py`.

## Cognition Pipeline

- `monGARS/core/conversation.py` assembles Hippocampus memory, curiosity gap
  detection, neuro-symbolic reasoning, adaptive response generation, mimicry,
  and speech synthesis, persisting every interaction for downstream analysis.【F:monGARS/core/conversation.py†L1-L122】
- The pipeline persists structured interaction metadata and history snapshots
  through `PersistenceRepository`, keeping the chat stack stateless at the API
  layer.【F:monGARS/core/conversation.py†L123-L178】

## Memory & Persistence

- Hippocampus blends in-memory caching with SQL-backed history management and
  Redis integrations, while `monGARS/core/persistence.py` offers transactional
  helpers for user, interaction, and adapter state records.【F:monGARS/core/hippocampus.py†L1-L160】【F:monGARS/core/persistence.py†L1-L160】
- Alembic revisions under `alembic/versions/` align SQLModel schemas with
  production deployments, including the 20250304 migration that backfills legacy
  conversation tables.【F:alembic/versions/20250304_01_align_sqlmodel_tables.py†L1-L200】

## LLM & Serving Layer

- `monGARS/core/llm_integration.py` negotiates between local Ollama inference
  and Ray Serve replicas with circuit breakers, endpoint rotation, TTL caches,
  and OpenTelemetry counters/histograms for `llm.ray.*` metrics.【F:monGARS/core/llm_integration.py†L1-L160】【F:monGARS/core/llm_integration.py†L161-L320】
- Model provisioning and manifest management live in
  `monGARS/core/model_manager.py` and `modules/neurons/registry.py`, ensuring
  adapters are installed and tracked before inference attempts occur.【F:monGARS/core/model_manager.py†L1-L200】【F:modules/neurons/registry.py†L1-L200】

## Research & Training Modules

- The **Evolution Orchestrator** coordinates MNTP training runs, energy usage
  tracking, and manifest updates, raising errors if artefacts fall outside the
  expected output tree.【F:modules/evolution_engine/orchestrator.py†L1-L160】
- `modules/neurons/training/mntp_trainer.py` provides curated dataset handling,
  deterministic linear adapters, and optional LoRA/QLoRA fine-tuning when heavy
  dependencies are available.【F:modules/neurons/training/mntp_trainer.py†L1-L160】【F:modules/neurons/training/mntp_trainer.py†L161-L320】
- Reinforcement-learning research loops are implemented in
  `modules/neurons/training/reinforcement_loop.py`, including adaptive scaling
  strategies, operator approvals, and evolution orchestrator integration for
  manifest rollouts.【F:modules/neurons/training/reinforcement_loop.py†L1-L160】【F:modules/evolution_engine/orchestrator.py†L360-L440】
- `monGARS/core/self_training.py` orchestrates automated self-improvement by
  curating records, persisting anonymised datasets, and launching MNTP training
  runs on a schedule.【F:monGARS/core/self_training.py†L1-L160】【F:monGARS/core/self_training.py†L161-L320】

## Web Operator Console

- The Django chat console delegates HTTP calls to
  `webapp/chat/services.py`, which handles authentication, chat submission, and
  history retrieval against the FastAPI backend with structured error handling.【F:webapp/chat/services.py†L1-L120】
- Templates and async views (see `webapp/chat/views.py`) integrate WebSocket
  tickets and progressive enhancement so operators can monitor and interact with
  conversations without JavaScript dependencies.【F:webapp/chat/views.py†L1-L200】

## Observability & Peer Collaboration

- `monGARS/core/distributed_scheduler.py` exports queue depth, uptime, and
  failure-rate gauges via OpenTelemetry, broadcasting telemetry snapshots to
  peers through `PeerCommunicator` for load-aware routing.【F:monGARS/core/distributed_scheduler.py†L1-L200】【F:monGARS/core/distributed_scheduler.py†L200-L400】
- `monGARS/core/peer.py` encrypts inter-node messages, caches telemetry, and
  supports dynamic bearer token rotation to keep distributed coordination secure
  and observable.【F:monGARS/core/peer.py†L1-L200】【F:monGARS/core/peer.py†L200-L360】
- `ResearchLongHaulService` schedules unattended validation cycles, deduplicates
  concurrent jobs, and captures the latest reinforcement summary for downstream
  consumers, ensuring RL telemetry remains fresh without manual triggers.【F:monGARS/core/research_validation.py†L1-L200】【F:tests/test_research_long_haul_service.py†L1-L200】

## Tests & Guardrails

- The suite covers API contracts (`tests/test_api_chat.py`), scheduler load
  sharing (`tests/test_distributed_scheduler.py`), reinforcement-learning loops
  (`tests/test_reinforcement_loop.py`), and module-specific guardrails to ensure
  TODO/NotImplemented markers do not regress.【F:tests/test_api_chat.py†L1-L200】【F:tests/test_distributed_scheduler.py†L1-L200】【F:tests/test_reinforcement_loop.py†L1-L200】【F:tests/test_incomplete_logic_guardrails.py†L1-L160】
- Property-based and chaos experiments supplement deterministic tests to stress
  caching, persistence, and failure-handling behaviour.【F:tests/property_test.py†L1-L200】【F:tests/chaos_test.py†L1-L160】

## Deployment & Operations

- Docker, Compose, and multi-architecture build scripts sit alongside Kubernetes
  manifests so operators can run the stack on laptops or clusters.【F:Dockerfile†L1-L200】【F:build_native.sh†L1-L160】
- External secret orchestration pulls runtime credentials from Vault using an
  `ExternalSecret`, eliminating raw `Secret` manifests in the repository.【F:k8s/secrets.yaml†L1-L52】
- `scripts/docker_menu.py` provides an interactive orchestrator that rotates
  secrets, resolves port collisions, and manages lifecycle tasks for local
  onboarding.【F:scripts/docker_menu.py†L1-L260】

## Known Gaps & Risks

- **Credential Hardening** – legacy bootstrap accounts were removed from FastAPI;
  audit existing deployments to ensure no environments still rely on the retired
  defaults before rotating secrets.【F:monGARS/api/web_api.py†L41-L88】
- ✅ **Sustainability Dashboards** – energy tracker reports and reinforcement
  observability feeds now converge in
  `models/encoders/sustainability_dashboard.json`, with OpenTelemetry metrics
  (`llm.sustainability.*`) exposing the same data for Grafana and alerting. The
  bridge updates run automatically during long-haul validation so operators have
  unified energy, approval, and replica insights for deployment decisions.【F:monGARS/core/long_haul_validation.py†L123-L520】【F:monGARS/core/sustainability_dashboard.py†L1-L260】
- ✅ **RAG Governance** – retention metadata, automated scrubbing, and documented
  export flows keep curated artefacts compliant as partner integrations scale.【F:docs/rag_dataset_governance.md†L1-L160】

## Roadmap Phase Summary

| Phase                           | Status         | Evidence                                                                                                                                                                                                                                                     |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 – Core Infrastructure         | ✅ Complete    | FastAPI/Django services, persistence, and container assets are in place.【F:monGARS/api/web_api.py†L63-L200】【F:README.md†L1-L120】                                                                                                                         |
| 2 – Functional Expansion        | ✅ Complete    | Adaptive response, mimicry, curiosity, and captioning modules run end-to-end.【F:monGARS/core/conversation.py†L1-L122】【F:monGARS/core/mimicry.py†L1-L200】                                                                                                 |
| 3 – Hardware & Performance      | ✅ Complete    | Scheduler metrics, worker tuning, and Ray Serve integration are implemented.【F:monGARS/utils/hardware.py†L1-L120】【F:monGARS/core/distributed_scheduler.py†L1-L200】【F:monGARS/core/llm_integration.py†L1-L200】                                          |
| 4 – Collaborative Networking    | ✅ Complete    | Peer telemetry, load-aware scheduling, and Sommeil optimisation loops are shipping.【F:monGARS/core/peer.py†L1-L200】【F:monGARS/core/sommeil.py†L1-L160】                                                                                                   |
| 5 – Web/API Refinement          | ✅ Complete    | FastAPI endpoints, WebSocket streaming, and published SDK packages cover partner integrations end-to-end.【F:monGARS/api/web_api.py†L41-L88】【F:docs/sdk-release-guide.md†L1-L160】                                                                         |
| 6 – Self-Improvement & Research | ✅ Complete    | Research long-haul automation, observability snapshots, and multi-replica coverage keep reinforcement loops production-ready.【F:monGARS/core/research_validation.py†L1-L200】【F:tests/test_long_haul_validation.py†L200-L320】 |
| 7 – Sustainability & Longevity  | 🌱 Planned     | Evolution engine and energy tracking are present, and carbon-aware rollout gating now defers high-emission cycles; cross-node artefact sharing and hardware-aware guidance remain open.【F:modules/evolution_engine/orchestrator.py†L260-L360】【F:modules/evolution_engine/energy.py†L1-L160】                           |

## Recommended Next Steps

1. Leverage the sustainability dashboard feed to model cross-node artefact
   reuse, enriching shared metrics with energy savings and reuse ratios.【F:monGARS/core/sustainability_dashboard.py†L200-L320】【F:monGARS/core/long_haul_validation.py†L520-L660】
2. Extend sustainability analytics to include hardware-aware rollouts once
   dashboards surface baseline consumption trends, combining replica insights
   with the new energy stream.【F:modules/evolution_engine/orchestrator.py†L260-L360】【F:docs/implementation_status.md†L180-L200】
