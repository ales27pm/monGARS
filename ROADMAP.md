# monGARS Development Roadmap (Revised)

This roadmap outlines the milestones needed to evolve monGARS from its current prototype stage into a stable and fully featured system. This document reflects the current, verified state of the codebase and prioritizes critical security and stability improvements.

---

### **Immediate Priority: Foundational Security & Stability**

*This is a new priority section to address critical vulnerabilities before further feature development.*

- 🔐 **Fix JWT Algorithm Mismatch:** Change the JWT algorithm from `RS256` to `HS256` to match the symmetric `SECRET_KEY` usage, or implement a full RSA key pair.
- 🔒 **Implement Secure Secret Management:** Replace insecure Kubernetes secrets (`k8s/secrets.yaml`) with a proper secret management solution like HashiCorp Vault or Sealed Secrets.
- 🛡️ **Harden Container Security:** Add a non-root user to `Dockerfile` and `Dockerfile.embedded` to prevent containers from running with root privileges.
- 📝 **Create `.dockerignore` File:** Prevent sensitive files (`.env`, `.git`) and local artifacts from being included in container images.
- 🐛 **Replace Hardcoded Demo Users:** Migrate from the hardcoded user dictionary in `web_api.py` to a proper database-backed authentication system.

---

### **Phase 1 - Core Infrastructure (Completed Q1 2025)**

- ✅ Established the Cortex, Hippocampus, Neurons, and Bouche modules with initial APIs.
- ✅ Added basic memory storage, conversation flow, and logging facilities.
- ✅ Provided Docker Compose and Kubernetes manifests for local deployment.

### **Phase 2 - Functional Expansion (Completed Q2 2025)**

- ✅ Introduced the Mimicry and Mains Virtuelles modules for adaptive behaviour.
- ✅ Implemented web scraping via Iris and improved the Curiosity Engine.
- ✅ Added initial, meaningful tests for `CircuitBreaker`, `TieredCache`, and `SelfTrainingEngine`.
- ✅ `LLMIntegration` now performs actual calls to local Ollama models with caching and a CircuitBreaker.

---

### **Phase 3 - Hardware & Performance Optimization (Current - Target Q3 2025)**

- ✅ Optimized CPU and memory usage for Raspberry Pi and Jetson boards via `recommended_worker_count()`.
- ✅ Built container images for embedded hardware targets (`build_embedded.sh`).
- ✅ Provided a host-optimized build script for developer workstations (`build_native.sh`).
- ✅ Improved embedded build script to push multi-arch images to a registry.
- ✅ Added cache hit/miss metrics with OpenTelemetry.
- ✅ Hardened Kubernetes RBAC rules (`rbac.yaml`).
- 📝 **In Progress:** Finalize PostgreSQL database migrations for all data models.
- 📝 **Planned:** Pin all base images in `docker-compose.yml` to specific versions to ensure build stability.

### **Phase 4 - Collaborative Networking (In Progress – Target Q4 2025)**

- ✅ Implemented encrypted peer-to-peer communication via `PeerCommunicator`.
- ✅ Added peer registration, unregistration, and listing endpoints with admin-only access.
- ✅ Introduced a `DistributedScheduler` for cooperative tasks across nodes.
- ✅ Implemented `SommeilParadoxal` for idle-time optimization.
- ✅ Added `safe_apply_optimizations` to Evolution Engine for sandboxed upgrades.
- 📝 **Planned:** Enhance the `DistributedScheduler` with more advanced task distribution strategies (e.g., load-aware routing).

---

### **Phase 5 - Web Interface & API Refinement (Target Q1 2026)**

- ✅ Basic FastAPI REST API with several endpoints (`/chat`, `/history`, `/token`) is functional.
- ✅ Basic JWT authentication implemented with `/token` issuance.
- ✅ Conversation history retrieval is secured per user.
- ✅ A Django frontend serves the initial chat interface.
- 📝 **In Progress:** Implement the backend WebSocket handler in FastAPI to complete the live chat functionality initiated by the frontend.
- 📝 **Planned:** Replace placeholder input validation with robust, centralized validation in Pydantic models for all API endpoints.
- 📝 **Planned:** Complete the user registration flow by connecting it to the database backend.
- 🚧 **Future:** Publish comprehensive API documentation and example client libraries.

### **Phase 6 - Self-Improvement and Research (Target Q2 2026)**

- ✅ Persistence layer for personality profiles using PostgreSQL and SQLAlchemy is complete.
- ✅ Meaningful test coverage exists for several core modules.
- 📝 **In Progress:** Implement *real* training logic in `SelfTrainingEngine`. The current engine only simulates versioning.
- 📝 **In Progress:** Implement dynamic personality analysis in `PersonalityEngine`. The current module only saves and loads static profiles without analyzing interactions.
- 📝 **Planned:** Expand test coverage to all remaining modules, including WebSockets, hardware utilities, and peer communication.
- 🚧 **Future:** Begin research and implementation of reinforcement learning loops for continuous improvement.

### **Phase 7 - Advanced ML & Distributed Systems (Future)**

- 🚧 **Future:** Implement the full `LLM2Vec` training framework and activate the Ray Serve inference path in `LLMIntegration` for distributed, large-scale inference.
- 🚧 **Future:** Integrate the `EvolutionEngine` into routine, automated optimization cycles based on real performance metrics.
- 🚧 **Future:** Automate energy usage reporting and implement more advanced hardware-aware scaling.
- 🚧 **Future:** Enable nodes to share optimization results via the `PeerCommunicator` to accelerate collective improvements.

