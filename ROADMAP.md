# monGARS Detailed Roadmap

monGARS is a modular, privacy-first AI system. The roadmap below outlines major development phases. Items marked with [x] are complete.

## Phase 1 – Core Infrastructure ✅
- Establish the Cortex, Hippocampus, Neurons and Bouche modules.
- Integrate basic memory storage, conversation flow and logging.
- Ship Docker Compose and Kubernetes manifests for local deployment.

## Phase 2 – Functional Expansion ✅
- Add Mimicry for behavioral adaptation and Mains Virtuelles for running user code.
- Expand scraping through Iris and refine the Curiosity Engine.
- Provide initial tests and monitoring hooks.

## Phase 3 – Hardware & Performance Optimization (current)
- Optimize CPU and memory usage for low-resource devices such as Raspberry Pi and Jetson boards.
- Build container images for embedded hardware targets.
- Improve telemetry in `config.py` and finalize tiered caching.
- Harden security policies and RBAC rules.

## Phase 4 – Collaborative Networking (planned)
- Enable peer-to-peer coordination across monGARS instances.
- Introduce a distributed scheduler for cooperative tasks.
- Extend Sommeil Paradoxal to run optimization during idle time.
- Stabilize the Evolution Engine for sandboxed updates.

## Phase 5 – Web Interface & API
- Replace placeholder code in `monGARS/api/web_api.py` with full REST endpoints.
- Finish the Django chat application under `webapp/`.
- Add authentication and permission checks to the web layer.
- Document the API and publish example clients.

## Phase 6 – Self‑Improvement and Research
- Persist personality profiles using a database (see TODO in `personality.py`).
- Expand the self-training engine and integrate real metrics from the Evolution Engine.
- Replace stubbed tests (`chaos_test.py`, `self_training_test.py`, `property_test.py`) with meaningful coverage.
- Explore reinforcement learning for continuous improvement.

This roadmap will evolve alongside the project. Community feedback and contributions are welcome!
