# Repository ↔ Memory Alignment

Use this mapping to translate long-term project memory references into concrete
code locations inside the monGARS repository. Update the table whenever
architecture or naming conventions change.

## API Layer
| Memory Term | Repository Target | Notes |
| --- | --- | --- |
| Bouche (dialogue API) | `monGARS/api/web_api.py` | REST endpoints for chat, history, peers, and health checks. |
| Security model | `monGARS/api/authentication.py`, `monGARS/core/security.py` | OAuth2 password flow, JWT issuance, Fernet utilities. |
| Backend scaffolding | `monGARS/api/dependencies.py` | Dependency injection helpers for FastAPI routes. |
| Speech/Browser clients | `monGARS/api/ws_manager.py` | WebSocket fan-out with connection bookkeeping. |

## Core Services
| Memory Term | Repository Target | Notes |
| --- | --- | --- |
| Cortex + Bouche orchestration | `monGARS/core/conversation.py` | ConversationalModule coordinating memory, curiosity, LLM, mimicry. |
| Hippocampus | `monGARS/core/hippocampus.py` | In-memory, lock-guarded conversation history. |
| Evolution Engine / Sommeil Paradoxal | `monGARS/core/evolution_engine.py`, `monGARS/core/sommeil.py` | Diagnostics, safe optimisation, idle-time triggers. |
| Mémoire Autobiographique | `monGARS/core/logging.py`, `monGARS/core/ui_events.py` | Structured logging and typed event bus. |
| Tronc (neuro-symbolic reasoning) | `monGARS/core/neuro_symbolic/advanced_reasoner.py` | Heuristic reasoning hints for the LLM pipeline. |

## Testing Infrastructure
| Memory Term | Repository Target | Notes |
| --- | --- | --- |
| Phase validation workflows | `tests/integration_test.py` | End-to-end validation of cognition pipelines. |
| Self-optimisation routines | `tests/self_training_test.py` | Covers self-training engine versioning. |
| Robustness guarantees | `tests/chaos_test.py` | Circuit breaker and failure-injection scenarios. |
| Behaviour invariants | `tests/property_test.py` | Property-based coverage for tiered caching. |

## Platform & Operations
| Memory Term | Repository Target | Notes |
| --- | --- | --- |
| Hippocampus provisioning | `init_db.py` | SQLModel schema definitions and migrations. |
| Background scheduling | `tasks.py`, `monGARS/core/distributed_scheduler.py` | Celery-style helpers and async queues. |
| Container orchestration | `docker-compose.yml`, `Dockerfile*` | Local stack provisioning and build scripts. |
| Kubernetes manifests | `k8s/*.yaml` | Deployments, Prometheus scraping, RBAC, secrets. |

## Documentation
- `monGARS_structure.txt` – canonical directory overview referenced by long-term
  memory entries.
- `ROADMAP.md` – planned milestones and their current status.

Keep this alignment file updated after large refactors so the assistant’s memory
remains trustworthy.
