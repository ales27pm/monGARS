# monGARS Roadmap

This roadmap reflects the verified state of the codebase and the milestones
required to reach production readiness.

## Immediate Priorities (Security & Stability)
- 🔐 Align JWT algorithm with deployed secrets (HS256 today). Implement asymmetric
  keys only when the infrastructure supports managed key storage.
- 🔒 Store runtime secrets in Vault/Sealed Secrets instead of raw `k8s/secrets.yaml`.
- 🛡️ Update Dockerfiles to run as non-root and add a `.dockerignore` to exclude
  secrets and build artefacts.
- 👤 Replace demo users in `web_api.py` with the database-backed authentication
  flow and migrations.

## Phase 1 – Core Infrastructure (✅ Completed Q1 2025)
- Established core cognition modules (Cortex, Hippocampus, Neurons, Bouche).
- Delivered Docker Compose and Kubernetes manifests for local/cluster deployment.

## Phase 2 – Functional Expansion (✅ Completed Q2 2025)
- Landed Mimicry, Personality, Adaptive Response, and image captioning modules.
- Integrated Iris scraping and Curiosity Engine for external research.
- Replaced placeholder tests with meaningful coverage for circuit breakers,
  caching, and self-training.

## Phase 3 – Hardware & Performance (🔄 In Progress, Target Q3 2025)
- ✅ Worker auto-tuning for Pi/Jetson (`recommended_worker_count`).
- ✅ Multi-architecture build scripts and cache metrics.
- ✅ Hardened RBAC manifests.
- ✅ Ray Serve HTTP integration with circuit breaking plus MNTP trainer support
  for LoRA and curated adapters.
- 🔄 Extend Alembic migrations for the newest SQLModel tables and expose Ray Serve
  success/failure counters via OpenTelemetry.

## Phase 4 – Collaborative Networking (🔄 In Progress, Target Q4 2025)
- ✅ Encrypted peer registry, admin-guarded endpoints, and distributed scheduler.
- ✅ Sommeil Paradoxal idle-time optimisation and safe apply pipeline.
- 🔄 Implement load-aware scheduling strategies and share optimisation telemetry
  across nodes.

## Phase 5 – Web Interface & API Refinement (🗓 Target Q1 2026)
- ✅ FastAPI chat/history/token endpoints with validation.
- ✅ Django chat UI with progressive enhancement.
- ✅ FastAPI WebSocket handler with ticket verification, history replay, and
  streaming guarded by `WS_ENABLE_EVENTS`.
- 🔄 Replace hard-coded credential stores with database-backed auth flows.
- 🚧 Publish polished SDKs and reference clients.

## Phase 6 – Self-Improvement & Research (🗓 Target Q2 2026)
- ✅ Personality profiles persisted via SQLModel with live adapter updates.
- ✅ Self-training cycles produce real adapter artefacts via
  `modules.neurons.training.mntp_trainer.MNTPTrainer` with deterministic fallbacks.
- 🚧 Explore reinforcement learning loops and advanced scaling strategies.
- 🔄 Expand tests for long-running MNTP jobs, multi-replica Ray Serve rollouts,
  and distributed workflows.

## Phase 7 – Sustainability & Longevity (🌱 Future)
- 🚧 Fully integrate evolution engine outputs into routine optimisation cycles.
- 🚧 Automate energy usage reporting and advanced hardware-aware scaling.
- 🚧 Share optimisation artefacts between nodes for faster convergence.

Review [docs/implementation_status.md](docs/implementation_status.md) for a
narrative deep dive into each phase and current discrepancies.
