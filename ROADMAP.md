# monGARS Development Roadmap

This roadmap outlines the milestones needed to evolve monGARS from its current prototype stage into a stable and fully featured system. Earlier phases delivered only the architectural skeleton; many core capabilities are still stubs.

## Phase 1 - Core Infrastructure (prototype completed Q1 2025)
- Established the Cortex, Hippocampus, Neurons and Bouche modules with initial APIs.
- Added basic memory storage, conversation flow and logging facilities.
- Provided Docker Compose and Kubernetes manifests for local deployment.
- Ray Serve hooks and evolution logic remained placeholders.

Milestone: architecture scaffolding in place.

## Phase 2 - Functional Expansion (prototype completed Q2 2025)
- Introduced the Mimicry and Mains Virtuelles modules for adaptive behaviour.
- Began scraping via Iris and improved the Curiosity Engine.
- Implemented early tests and monitoring hooks; most tests are still placeholders.
- Simulated LLM output through the LLMIntegration stub.

Milestone: proof-of-concept functionality established, real AI features still missing.

## Phase 3 - Hardware & Performance Optimization (current - target Q3 2025)
- **[Completed]** Finish the LLM2Vec training pipeline and Ray Serve inference integration.
- **[Completed]** Implement the conversation history endpoint.
- **[Completed]** Add encrypted token handling for social media integration.
- **[Completed]** Improve error handling and tests for social posting.
- **[Completed]** Optimize CPU and memory usage for Raspberry Pi and Jetson boards.
  Worker auto-tuning now falls back to logical CPUs when physical core count is
  unavailable.
- **[Completed]** Build container images for embedded hardware targets.
- **[Completed]** Provide a host-optimized build script for Intel i7 developer workstations.
- **[Completed]** Improved embedded build script to push multi-arch images to a registry.
- **[Completed]** Added cache hit/miss metrics with OTEL units and layer labels. PostgreSQL migrations pending.
- **[Completed]** Harden security policies and RBAC rules.

## Phase 4 - Collaborative Networking (planned - target Q4 2025)
**[In Progress]** Enable peer-to-peer coordination with encrypted communication channels. `PeerCommunicator` now dispatches requests concurrently and `/api/v1/peer/message` requires authentication.
- Introduce a distributed scheduler for cooperative tasks across nodes.
- Extend Sommeil Paradoxal for idle-time optimization and auto-updates.
- Stabilize the Evolution Engine for safe sandboxed upgrades.

## Phase 5 - Web Interface & API (target Q1 2026)
- Replace placeholder routes with full REST endpoints and robust input validation.
- Complete the Django chat application with WebSocket support and history view.
- Add authentication, user management and permission checks.
- Publish API documentation and example client libraries.

## Phase 6 - Self-Improvement and Research (target Q2 2026)
- Persist personality profiles using PostgreSQL and SQLAlchemy.
- Expand the self-training engine using real metrics from the Evolution Engine.
- Replace stubbed tests (`chaos_test.py`, `self_training_test.py`, `property_test.py`) with meaningful coverage.
- Explore reinforcement learning and dynamic scaling for continuous improvement.

This roadmap will evolve as monGARS matures. Community feedback and contributions are welcome.
