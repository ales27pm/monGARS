# monGARS Report Comparison

This report cross-references the "monGARS: Executive Summary and Technical Architecture" narrative with the current repository to highlight alignment, partial delivery, and gaps.

## Key Takeaways

- The runtime wires modules together inside a monolith; no message broker or plug-and-play deployment flow exists despite the marketing focus on modular microservices.
- Privacy-by-design promises are undermined by outbound web searches, demo credentials, and plaintext persistence.
- Safety, neurosymbolic reasoning, and sandboxing claims are largely aspirational; implemented safeguards focus on JWT hygiene, heuristic hints, and image captioning.

## Methodology

The assessment inspects runtime packages (`monGARS/core`, `monGARS/api`) and optional trainers (`modules/`). Findings cite concrete source files to show where behaviour diverges from the report’s statements.

## Detailed Findings

### Architectural Modularity
Both the FastAPI surface and the conversation orchestrator instantiate dependencies directly within the process, relying on constructor wiring instead of message broker RPC. The orchestrator coordinates modules sequentially, and the conversation module constructs every dependency (LLM, curiosity, mimicry, hippocampus, evolution engine) itself, which contradicts the claimed broker-mediated microservice topology.【F:monGARS/core/orchestrator.py†L21-L185】【F:monGARS/core/conversation.py†L25-L155】 No RabbitMQ or equivalent appears in the runtime path.

### Autonomy and Continual Learning
Curiosity-driven learning exists but is tightly scoped: `CuriosityEngine` checks prior queries, optionally hits the knowledge graph, and escalates to web search via Iris when enough entities are missing.【F:monGARS/core/cortex/curiosity_engine.py†L61-L176】 The background optimiser invoked by Sommeil Paradoxal calls `EvolutionEngine.safe_apply_optimizations`, which currently tunes worker replicas and clears caches; it does not author code patches or perform hypothesis-driven experiments.【F:monGARS/core/sommeil.py†L14-L89】【F:monGARS/core/evolution_engine.py†L254-L360】 Encoder retraining requires manual triggers through the evolution orchestrator and MNTP trainer, so "continuous" improvement still expects operator initiation and succeeds only when heavyweight ML dependencies are installed.【F:modules/evolution_engine/orchestrator.py†L28-L200】【F:modules/neurons/training/mntp_trainer.py†L13-L200】

### Longevity and Efficiency Routines
Sommeil Paradoxal’s idle-time loop waits for the scheduler queue to drain, then reruns the same optimisation routine, publishing UI events for observability. The effect is limited to autoscaling and cache eviction rather than the report’s description of research into best practices or library updates.【F:monGARS/core/sommeil.py†L29-L89】【F:monGARS/core/evolution_engine.py†L254-L313】 Embedding fallbacks also show the system is prepared to run without GPU support but at the cost of deterministic hash-based vectors, which undercuts promises of high-fidelity embeddings on constrained hardware.【F:monGARS/core/neurones.py†L83-L188】

### Data Ingestion and Privacy Posture
Iris performs remote DuckDuckGo lookups for every search query, exposing user text to third-party infrastructure unless operators insert their own proxy. There is no local-only mode beyond the ability to replace the HTTP client.【F:monGARS/core/iris.py†L96-L198】 Simultaneously, the API lifecycle manager seeds demo accounts (`u1`/`u2`) on startup, conflicting with the "self-sufficient local assistant" privacy stance and requiring manual cleanup before deployment.【F:monGARS/api/web_api.py†L58-L174】

### Memory Handling and Vector Security
Conversation history persists plaintext queries and responses with embeddings stored as generic JSON blobs. There is no encryption layer, key management, or pgvector-backed column usage as highlighted in the marketing material.【F:monGARS/db/models.py†L27-L55】 Database initialisation simply enables the `vector` extension—no hardening or secret management is applied.【F:init_db.py†L59-L71】 Persistence helpers append additional plaintext history rows whenever an interaction is saved, further entrenching the lack of at-rest protection.【F:monGARS/core/persistence.py†L79-L126】

### Safety and Alignment Controls
Security modules enforce HS256 JWTs, password hashing, and HTML sanitisation but make no attempt to isolate "critical safety directives" in protected memory. Attempting to configure asymmetric JWT keys raises an error, underscoring the focus on compatibility with existing symmetric secrets rather than novel safety layers.【F:monGARS/core/security.py†L20-L193】

### Mémoire Autobiographique and Auditability
The purported immutable autobiographical memory resolves to a pluggable event bus. By default it keeps events in volatile asyncio queues and only persists anything if operators enable Redis externally, so audit trails disappear on restart—far from the immutable log promised in the report.【F:monGARS/core/ui_events.py†L27-L178】

### Sandboxing and "Mains Virtuelles"
`Mains Virtuelles` is exclusively an image-captioning helper around BLIP; there is no sandboxed code execution, interpreter isolation, or language runtime management. As a result, the "secure sandbox" claim in the report is currently unmet.【F:monGARS/core/mains_virtuelles.py†L19-L97】

### Neurosymbolic Reasoning Depth
The neurosymbolic layer reduces to keyword heuristics that append canned French/English hints based on question words. There is no rule engine, symbolic planner, or integration with the embedding system, so the implementation delivers shallow prompt augmentation instead of the described neuro-symbolic synthesis.【F:monGARS/core/neuro_symbolic/advanced_reasoner.py†L9-L34】

### Collaboration and Secure Networking
`PeerCommunicator` encrypts payloads with Fernet-derived tokens and can broadcast telemetry, partially aligning with the encrypted peer-to-peer vision. However, peers must share secrets out of band, and there is no automated network discovery or governance protocol, leaving secure collaboration largely manual.【F:monGARS/core/peer.py†L24-L211】 The distributed scheduler primarily tracks queue depth and broadcasts metrics; it does not orchestrate cross-node task routing over a broker.【F:monGARS/core/distributed_scheduler.py†L104-L212】

### Module Naming Versus Behaviour
While components carry the marketing names, functionality often diverges. `Mains Virtuelles` handles captions, not sandboxing; the "neurons" module falls back to hash-derived embeddings when ML models are unavailable; and the "evolution engine" focuses on autoscaling rather than experiment tracking. These differences mean the names alone overstate the delivered feature set.【F:monGARS/core/mains_virtuelles.py†L19-L97】【F:monGARS/core/neurones.py†L83-L188】【F:monGARS/core/evolution_engine.py†L254-L360】

## Conclusion
monGARS implements several aspirational ideas—curiosity-driven prompts, idle-time maintenance, encrypted peer messaging—but the majority of the marketing report’s standout capabilities remain incomplete or manually operated. Treat the document as a roadmap rather than an accurate snapshot of the deployed system.
