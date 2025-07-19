# monGARS Development Roadmap

This roadmap outlines the milestones needed to evolve monGARS from its current prototype stage into a stable and fully featured system. Earlier phases delivered only the architectural skeleton; many core capabilities are still stubs.
## Phase 1 - Core Infrastructure (prototype completed Q1 2025)
- Established the Cortex, Hippocampus, Neurons and Bouche modules with initial APIs.
- Added basic memory storage, conversation flow and logging facilities.
- Provided Docker Compose and Kubernetes manifests for local deployment.
- Ray Serve hooks and evolution logic remained placeholders.


## Phase 2 - Functional Expansion (prototype completed Q2 2025)
- Introduced the Mimicry and Mains Virtuelles modules for adaptive behaviour.
- Began scraping via Iris and improved the Curiosity Engine.
- Added initial tests for CircuitBreaker, TieredCache and SelfTrainingEngine alongside monitoring hooks.
- Early LLMIntegration used local Ollama models with caching and a CircuitBreaker; the Ray Serve path remained stubbed.

Milestone: proof-of-concept functionality established, real AI features still missing.

## Phase 3 - Hardware & Performance Optimization (current - target Q3 2025)
- 📝 Framework for LLM2Vec training and Ray Serve inference integration; core logic still stubbed.
- ✅ Implemented the conversation-history endpoint.
- ✅ Added encrypted token handling for social media integration.
- ✅ Improved error handling and tests for social posting.
- ✅ Optimized CPU and memory usage for Raspberry Pi and Jetson boards. Worker auto-tuning now falls back to logical CPUs when physical core count is unavailable.
- ✅ Built container images for embedded hardware targets.
- ✅ Provided a host-optimized build script for Intel i7 developer workstations.
- ✅ Improved embedded build script to push multi-arch images to a registry.
- ✅ Added cache hit/miss metrics with OTEL units and layer labels. PostgreSQL migrations pending.
- ✅ Hardened security policies and RBAC rules.

## Phase 4 - Collaborative Networking (in progress – target Q4 2025)
- 📝 Enable peer-to-peer coordination with encrypted communication channels. `PeerCommunicator` now dispatches requests concurrently and `/api/v1/peer/message` requires authentication.
- ✅ Added peer registration, unregistration and listing endpoints with URL validation and duplicate handling to manage known peers.
- ✅ Introduced a `DistributedScheduler` for cooperative tasks across nodes.
- ✅ Extended Sommeil Paradoxal for idle-time optimization and auto-updates.
- ✅ Added `safe_apply_optimizations` to Evolution Engine for sandboxed upgrades.

- ✅ Added `/api/v1/conversation/chat` endpoint with input sanitization.
- 🚧 Refine existing REST endpoints with robust input validation and error handling.
- 🚧 Backend WebSocket handler to complete the Django chat application and expose a history view.
- 🚧 Authentication, user management and permission checks.
- 🚧 Publication of API documentation and example client libraries.

## Phase 6 - Self-Improvement and Research (target Q2 2026)
- ✅ Persist personality profiles using PostgreSQL and SQLAlchemy.
- 🚧 Implement real training logic in SelfTrainingEngine using Evolution Engine metrics; current cycles simulate versioning only.
- 📝 Expand coverage across remaining modules beyond CircuitBreaker, TieredCache and SelfTrainingEngine.
- 🚧 Explore reinforcement learning and dynamic scaling for continuous improvement.

## Phase 7 - Sustainability & Longevity (future)
- 🚧 Integrate the Evolution Engine into routine optimization cycles.
- 🚧 Automate energy usage reporting and hardware-aware scaling.
- 🚧 Share optimization results between nodes to accelerate improvements.

This roadmap will evolve as monGARS matures. Community feedback and contributions are welcome.
Additional fine-tuning and distributed inference plans are provided in `docs/advanced_fine_tuning.md`.
For detailed module status and current implementations, see [docs/implementation_status.md](docs/implementation_status.md).
