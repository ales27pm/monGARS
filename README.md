# monGARS

> **Last updated:** 2025-11-25 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

**monGARS** (Modular Neural Guardian for Autonomous Research & Support) is a
privacy-first AI assistant designed for resilient deployment on workstations,
edge appliances, and research clusters. The project combines a FastAPI service,
adaptive cognition pipeline, Django operator console, and optional ML training
modules so contributors can experiment end-to-end without bespoke tooling.

## Table of Contents
1. [Project Snapshot](#project-snapshot)
2. [Project Health](#project-health)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Functional Snapshots](#functional-snapshots)
5. [Service Breakdown](#service-breakdown)
6. [Getting Started](#getting-started)
7. [Configuration Keys](#configuration-keys)
8. [Operational Workflows](#operational-workflows)
9. [Deployment Options](#deployment-options)
10. [Security & Observability](#security--observability)
11. [Documentation Map](#documentation-map)
12. [Contributing](#contributing)

## Project Snapshot
- **Conversational engine** with memory-backed context, curiosity-driven
  research, and style adaptation.
- **Memory architecture** that blends in-memory `Hippocampus`, SQL persistence,
  and Redis/disk caching layers.
- **Evolution engine** to retrain adapters, roll out artefacts, and trigger safe
  optimisations during idle windows.
- **Self-improvement loop** where personality profiles persist via SQLModel and
  refresh through LoRA-style adapters, while SelfTrainingEngine now feeds the
  MNTP trainer for curated batches and materialises deterministic adapters when
  heavyweight ML dependencies are unavailable.
- **Web interface** powered by Django async views that proxy authentication and
  chat history to FastAPI.
- **Hardware-aware operations** with worker tuning for Raspberry Pi/Jetson,
  container build scripts for multiple architectures, and Kubernetes manifests.
- **Repository-aware RAG enrichment** that augments `/ask`, `/review`, and
  compliance flows when `rag_enabled` is set.

## Project Health
- **Phase 5 – Web/API Refinement**: REST, WebSocket, and Django operator flows
  are production-ready, credential bootstrap now relies solely on persisted
  accounts, and the Python/TypeScript SDKs ship with documented release
  tooling.【F:monGARS/api/web_api.py†L41-L120】【F:docs/sdk-release-guide.md†L1-L80】【F:docs/sdk-overview.md†L1-L112】
- **Phase 6 – Self-Improvement & Research**: Automated MNTP self-training and
  reinforcement-learning loops now operate with scheduled long-haul soak tests,
  durable observability snapshots, and operator approvals coordinated by the
  research long-haul service, closing the roadmap milestone.【F:modules/evolution_engine/orchestrator.py†L360-L440】【F:monGARS/core/research_validation.py†L1-L200】【F:tests/test_long_haul_validation.py†L200-L320】【F:tests/test_research_long_haul_service.py†L1-L200】
- **Open Risks**: Prioritise sustainability—extend the energy tracker feeds and
  hardware-aware rollouts into shared dashboards before declaring the
  longevity phase ready for production workloads.【F:modules/evolution_engine/energy.py†L1-L160】【F:docs/codebase_status_report.md†L169-L214】

See [docs/codebase_status_report.md](docs/codebase_status_report.md) for the full
audit of runtime modules, tests, and deployment assets.【F:docs/codebase_status_report.md†L1-L188】

## Architecture at a Glance
![System overview diagram](docs/images/system-overview.svg)

The system is intentionally modular:
- **FastAPI surface (`monGARS.api`)** exposes OAuth2 authentication, chat,
  conversation history, and peer-management endpoints. WebSockets are mediated by
  a dedicated manager so broadcasts remain consistent, and Ray Serve requests are
  issued directly with per-endpoint retries when distributed inference is
  enabled.
- **Core cognition (`monGARS.core`)** orchestrates Hippocampus memory, curiosity
  heuristics, neuro-symbolic reasoning, LLM adapters (Ollama or Ray Serve), and
  personality/mimicry engines.
- **Optional modules (`modules/`)** host the evolution engine, neuron trainers,
  and research tooling. They degrade gracefully when heavy dependencies are
  absent.
- **Django webapp (`webapp/`)** delivers a token-aware operator console with
  progressive enhancement for offline/JS-light scenarios.
- **Persistence & cache layers** span PostgreSQL, Redis, disk snapshots, and an
  adapter manifest that coordinates encoder rollouts across services.

## Functional Snapshots
Each core capability ships with a quick visual reference that you can embed in
slide decks or ops runbooks.

### Conversational Pipeline
![Conversation pipeline](docs/images/conversation-pipeline.svg)
- OAuth2 login issues JWTs consumed by FastAPI dependencies.
- Requests are validated, sanitised, and enriched with memory, curiosity-driven
  research, and neuro-symbolic hints before hitting the LLM.
- Responses are adapted for user personality, persisted, cached, and streamed to
  WebSocket subscribers.

### Evolution & Self-Training Loop
![Evolution engine workflow](docs/images/evolution-engine.svg)
- Diagnostics collect OpenTelemetry counters, system metrics, and curiosity gap
  reports.
- `MNTPTrainer` produces new adapters (LoRA when GPU/PEFT dependencies are
  present, deterministic linear adapters otherwise), publishes metrics to MLflow,
  and updates the shared manifest.
- Deployments notify Ray Serve replicas (when enabled) and trigger idle-time
  optimisation cycles through `SommeilParadoxal`.

### RAG Context Enrichment
- When `rag_enabled=true`, `RagContextEnricher` calls the configured
  `rag_service_url`/`DOC_RETRIEVAL_URL` endpoint.
- Focus areas and references are normalised into typed objects consumed by
  FastAPI review tooling and `/api/v1/review/rag-context`.
- Network hiccups degrade gracefully to an empty enrichment payload so chat and
  review flows remain responsive.

### Operator Console Flow
![Web application flow](docs/images/webapp-flow.svg)
- Django templates render a progressive chat interface and bootstrap WebSockets.
- `chat/services.py` manages token exchange, retries, and history hydration via
  async `httpx` calls.
- FastAPI endpoints provide token issuance, chat, history, and streaming updates
  for the UI.

### Deployment Stack
![Deployment stack](docs/images/deployment-stack.svg)
- Run locally with Python/uvicorn, package via Docker/Compose, or deploy to
  Kubernetes using the provided manifests and RBAC rules.

## Service Breakdown
| Component | Location | Responsibilities |
| --- | --- | --- |
| FastAPI service | `monGARS/api` | Authentication, REST/WebSocket endpoints, dependency wiring |
| Cognition core | `monGARS/core` | Memory, reasoning, LLM integration, adaptive response generation |
| Evolution modules | `modules/` | Adapter training, diagnostics, research tooling |
| Django UI | `webapp/` | Operator-facing chat console and auth proxy |
| Persistence | `init_db.py`, `monGARS/core/persistence.py` | SQLModel schemas, Hippocampus caching, Redis/disk tiers |
| Tooling | `tasks.py`, `build_*.sh`, `docker_menu.py`, `docker-compose.yml`, `k8s/` | Automation, orchestration, deployment manifests |

## Getting Started
### Prerequisites
- Python 3.11+
- Docker & Docker Compose for container workflows
- NVIDIA drivers compatible with CUDA 12.1 when running GPU workloads inside the
  official `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` image
- Optional: GPU drivers for Ollama or Torch-based adapters

### Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # update secrets + URLs
python init_db.py      # applies Alembic migrations and prepares persistence tables
uvicorn monGARS.api.web_api:app --host 0.0.0.0 --port 8000 --reload
```

- Quick path: `scripts/install_test_dependencies.sh` installs the runtime and
  test dependencies defined in `requirements.txt` so `pytest` can start
  collecting modules without import errors.

### Multi-stage training & export pipeline
- Use [`mongars_multistage_pipeline.py`](mongars_multistage_pipeline.py) to
  orchestrate dataset generation, module-level SFT, LLM2Vec adaptation, and
  export in one CLI. The runbook in
  [`docs/mongars_multistage_pipeline.md`](docs/mongars_multistage_pipeline.md)
  lists the stage flags, state/resume behaviour, and troubleshooting tips.

### Docker Compose Stack
```bash
python scripts/docker_menu.py
```

The interactive orchestrator introduces a terminal menu with options to deploy
(`up --build -d`), start without rebuilding, stop, restart, inspect status,
tail logs, rebuild images, pull upstream images, open a shell inside any
service, and destroy the stack. Additional tooling now:

- Creates `.env` from `.env.example` on first run and rotates weak defaults for
  JWT, Django, Postgres, and Vault secrets.
- Layers `docker-compose.searxng.yml` automatically when
  `SEARCH_SEARX_ENABLED=true`, wiring the FastAPI/Django containers to the
  internal `http://searxng:8080` endpoint while exposing the host-accessible
  interface on `http://localhost:${SEARXNG_PORT:-8082}`.
- Scans for port collisions across API, webapp, SearxNG, Postgres, Redis,
  MLflow, Vault, Ollama, and Ray endpoints. Busy ports are replaced with the
  next available value, persisted back to `.env`, reflected in
  `WS_ALLOWED_ORIGINS`, and applied to the SearxNG base URLs so CLI tools keep
  talking to the correct host without manual edits.
- Lets you toggle optional profiles before startup. Enable the Ollama profile
  to launch the local LLM runtime or the Ray profile to expose the Ray head
  node (`${RAY_DASHBOARD_PORT:-8265}`), Ray Client API
  (`${RAY_CLIENT_PORT:-10001}`), Serve HTTP gateway (`${RAY_HTTP_PORT:-8005}`),
  and a dedicated worker port range (`${RAY_MIN_WORKER_PORT:-20000}`-
  `${RAY_MAX_WORKER_PORT:-20100}`) that avoids collisions with the Ray client
  endpoint by default.
- Provides guided diagnostics (`option 12`) that optionally auto-remediate
  missing `.env` entries, weak secrets, stale ports, and Compose syntax issues
  before you deploy.
- Offers an auto-heal routine (`option 13`) that restarts unhealthy services
  after a successful diagnostic pass so the stack can recover without manual
  container juggling.
- Exposes an environment summary (`option 14`) with sensitive values redacted to
  support incident response without leaking secrets in shared terminals.
- Captures Compose failures, runs lightweight triage (status snapshots +
  diagnostics), and suggests remediation such as port regeneration or network
  rebuilds when commands exit unsuccessfully.

For automated pipelines you can continue to call Docker Compose directly:

```bash
docker compose -f docker-compose.yml --project-name mongars up -d
```

#### Docker Compose v1 compatibility

Docker Engine 27+ removes the legacy `ContainerConfig` field that Compose v1
(`docker-compose` 1.29.x) relied on when reconciling named volumes. If you see
errors such as `'ContainerConfig'` while recreating services like Postgres,
Redis, Vault, or MLflow, run the compatibility wrapper included with the
repository:

```bash
scripts/compose_up.sh up -d
```

The script pins `DOCKER_API_VERSION=1.43` so the engine exposes the legacy
schema, disables BuildKit for parity with Compose v1 expectations, and
auto-detects whether the modern `docker compose` plugin is available before
falling back to the classic binary. You can pass any Compose arguments through
unchanged, for example `scripts/compose_up.sh down --volumes`.

The refreshed Compose topology keeps the same service names but now uses
anchors to share environment blocks, consistent health checks, and optional
profiles for inference (`ollama`) and distributed serving (`ray-head`,
`rayserve`). Application images default to
`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, but the Dockerfiles accept a
`PYTORCH_IMAGE` build argument so you can bump CUDA/PyTorch releases without
editing manifests. The GPU overlay sets `PYTORCH_IMAGE_GPU` to
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` and publishes the build under
`MONGARS_GPU_IMAGE` to avoid clobbering CPU-only tags. Both variants ship with
Git, Git LFS, FFmpeg, the `fr_core_news_sm` spaCy model, and transformer tooling
preinstalled so containers can pull adapters, process multimedia context, and
invoke LLMs out of the box.

### Django Operator Console
```bash
cd webapp
export DJANGO_SECRET_KEY="change-me"
export FASTAPI_URL="http://localhost:8000"
python manage.py migrate
python manage.py runserver 0.0.0.0:8001
```
Browse to `http://127.0.0.1:8001/chat/login/` and authenticate with a FastAPI
user/token.

## Configuration Keys
| Setting | Description |
| --- | --- |
| `SECRET_KEY` | Required for JWT signing and Fernet encryption. Never leave empty. |
| `API_PORT` / `WEBAPP_PORT` | Host ports exposed for the FastAPI service and Django operator console. |
| `POSTGRES_PORT`, `REDIS_PORT`, `MLFLOW_PORT`, `VAULT_PORT`, `OLLAMA_PORT`, `RAY_HTTP_PORT`, `RAY_DASHBOARD_PORT`, `RAY_CLIENT_PORT` | Host bindings for stateful and optional services managed by Compose. |
| `RAY_MIN_WORKER_PORT`, `RAY_MAX_WORKER_PORT` | Lower/upper bounds for the Ray head worker port range. Defaults prevent overlaps with `RAY_CLIENT_PORT`. |
| `DB_PASSWORD` | Password applied to the Postgres user `mongars`; rotated automatically by the deploy script when left as `changeme`. |
| `USE_RAY_SERVE` / `RAY_SERVE_URL` | Enable distributed inference and point at Ray Serve HTTP endpoints (defaults to `http://rayserve:8000/generate`). |
| `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`, `DJANGO_DEBUG_HOSTS`, `FASTAPI_URL` | Settings used by the Django operator console when running inside Compose. When `DJANGO_ALLOWED_HOSTS` is unset the console now trusts `localhost`, loopback addresses, `0.0.0.0`, and Compose provided `WEBAPP_HOST`/`HOST` values automatically. `DJANGO_DEBUG_HOSTS` appends comma-separated hostnames/IPs that should be trusted automatically when `DJANGO_DEBUG=true`. |
| `MONGARS_IMAGE` / `MONGARS_GPU_IMAGE` | Tags applied to the CPU and GPU application builds. Using distinct values prevents rebuilds for one profile from overwriting the other. |
| `PYTORCH_IMAGE` / `PYTORCH_IMAGE_GPU` | Base images consumed by the Dockerfiles; override to align with local CUDA driver stacks without editing manifests. |
| `RAY_HEAD_IMAGE`, `RAY_HEAD_IMAGE_GPU`, `RAY_SERVE_IMAGE`, `RAY_SERVE_GPU_IMAGE` | Control the Ray head/Serve container tags for CPU and GPU deployments. Pair with the new build args to align with cluster driver versions. |
| `OLLAMA_HOST` | URL for the Ollama runtime; defaults to the local container (`http://ollama:11434`). |
| `LLM_MODELS_CONFIG_PATH` | Path to the JSON manifest listing model profiles and download preferences. |
| `LLM_MODELS_PROFILE` | Name of the profile within `LLM_MODELS_CONFIG_PATH` used for inference defaults. |
| `LLM_MODELS_AUTO_DOWNLOAD` | When `true`, missing local models are pulled automatically if the provider supports it. |
| `LLM_GENERAL_MODEL` / `LLM_CODING_MODEL` | Optional overrides for the conversational and coding model names. |
| `DOCUMENT_RETRIEVAL_URL` | Endpoint for external research invoked by the curiosity engine. |
| `RAG_ENABLED` / `RAG_SERVICE_URL` / `RAG_REPO_LIST` / `RAG_MAX_RESULTS` | Toggle repository-aware enrichment, direct the enrichment service, scope repositories, and clamp the number of references. |
| `WORKER_DEPLOYMENT_NAME` / `WORKER_DEPLOYMENT_NAMESPACE` | Kubernetes deployment targeted by the evolution engine when scaling. |
| `VAULT_URL` / `VAULT_TOKEN` | Vault dev server address and bootstrap token for local testing. |
| `WS_ALLOWED_ORIGINS` / `WS_ENABLE_EVENTS` | Comma-separated list of origins allowed to open WebSocket sessions and a feature flag to disable streaming entirely. |
| `REDIS_URL`, `DATABASE_URL` | Connection strings for cache and persistence layers. |
| `OPEN_TELEMETRY_EXPORTER` | Optional metrics exporter configuration for observability pipelines. |
| `TRAINING_PIPELINE_ENABLED` / `TRAINING_CYCLE_INTERVAL_SECONDS` / `TRAINING_CYCLE_JITTER_SECONDS` | Control whether the background evolution training workflow runs and how often cycles are executed. |
| `TRAINING_PIPELINE_USER_ID` / `TRAINING_PIPELINE_VERSION_PREFIX` | Identify background training cycles in telemetry and manifest updates. |

Document any new environment variable or feature flag in this table and the
relevant module docstrings. Model profiles live in `configs/llm_models.json` and
can be extended to register additional Ollama models or alternate providers.

### Model Management & Provisioning
- **Inspect configuration** – `GET /api/v1/models` (admin token required) returns
  the active profile, normalised model metadata, and available profiles defined
  in `configs/llm_models.json`.
- **Trigger provisioning** – `POST /api/v1/models/provision` (admin token) calls
  into the shared `LLMModelManager` to download or validate the models backing
  each role. Supply `{"roles": ["general", "coding"], "force": true}` to
  re-sync specific models.
- **CLI** – `python -m scripts.provision_models --roles general coding` runs the
  same provisioning workflow from the shell. Pass `--json` to emit machine
  readable output for automation.

### Exporting Ollama Models
- **Prerequisite** – Install the [Ollama](https://ollama.com/) runtime locally so
  the `ollama` CLI is available on your `PATH`.
- **Export workflow** – Run the helper to persist an Ollama-managed model to a
  portable `.bin` artefact:

  ```python
  from pathlib import Path

  from monGARS.mlops.exporters import export_to_ollama

  export_path = export_to_ollama("llama3", Path("./exports"))
  print(f"Model exported to {export_path}")
  ```

- **Verification** – The command above creates an `.bin` file (for example
  `exports/llama3.bin`). Use standard filesystem tooling to confirm the file is
  present before distributing it to air-gapped or offline environments.

## Operational Workflows
- **Testing**: `pytest -q` (631 tests in ~107 seconds) for unit/integration coverage. Run
  `pytest -k <pattern>` while iterating and `pytest --maxfail=1` during triage; execute `npm run test` when frontend assets or SDKs change.
- **Static analysis**: `black . && isort .` before committing. Add type hints when
  editing public APIs.
- **Provisioning**: `python -m scripts.provision_models --json` ensures Ollama
  weights for the active profile are present before running integration tests.
- **Evolution cadence**: The background pipeline launched from `main.py` uses
  `monGARS.mlops.training_pipeline.training_workflow` to trigger adapter refresh
  cycles. Tune cadence and identifiers with the `TRAINING_PIPELINE_*`
  environment variables documented above.
- **Database migrations**: define new SQLModel tables in `init_db.py`, generate an
  Alembic migration, and document schema changes.
- **Worker tuning**: `monGARS.utils.hardware.recommended_worker_count()` auto-tunes
  Uvicorn worker counts on constrained hardware—no manual flags needed on Pi or
  Jetson devices.

## Deployment Options
- **Local**: `uvicorn monGARS.api.web_api:app` for rapid development.
- **Container**: `build_native.sh` (x86_64) or `build_embedded.sh` (multi-arch)
  produce images aligned with `Dockerfile` / `Dockerfile.embedded`.
- **Kubernetes**: manifests in `k8s/` cover deployments, RBAC, Prometheus
  scraping, and secrets. Update RBAC when introducing new controllers.
- **Ray Serve**: follow [docs/ray_serve_deployment.md](docs/ray_serve_deployment.md)
  to provision inference clusters with graceful fallbacks.
- **RAG**: enable enrichment by setting `rag_enabled=true`, configuring
  `rag_service_url`, and following
  [docs/rag_context_enrichment.md](docs/rag_context_enrichment.md).

## Security & Observability
- Restrict WebSocket origins via `WS_ALLOWED_ORIGINS` and terminate TLS at your
  ingress layer.
- Enable per-user rate limiting by setting `WS_RATE_LIMIT_MAX_TOKENS` and
  `WS_RATE_LIMIT_REFILL_SECONDS`.
- Review OpenTelemetry metrics (`llm.*`, `cache.*`, `evolution_engine.*`) and
  structured logs for operational insight. MLflow captures training runs when the
  evolution engine is active.
- Store secrets in Vault or a compatible secret manager. Avoid committing `.env`
  files or sample secrets.

## Documentation Map
Start with the curated [docs/index.md](docs/index.md) hub for an audience-focused
navigation of every runbook. The tables below highlight frequently referenced
entries for quick access. Keep the documentation set dynamic by running
`python scripts/update_docs_metadata.py` before opening a PR; the script refreshes
`Last updated` banners across all Markdown files using Git history.

### Architecture & Core Services

| Topic | Location |
| --- | --- |
| System diagrams and component walkthrough | [docs/architecture/module_interactions.md](docs/architecture/module_interactions.md) |
| Conversation pipeline deep dive | [docs/conversation_workflow.md](docs/conversation_workflow.md) |
| Model configuration & provisioning | [docs/model_management.md](docs/model_management.md) |
| Repository vs. memory mapping | [docs/repo_memory_alignment.md](docs/repo_memory_alignment.md) |

### Operations & Delivery

| Topic | Location |
| --- | --- |
| Visual deployment orchestrator | [docs/deployment_automation.md](docs/deployment_automation.md) |
| Ray Serve deployment guide | [docs/ray_serve_deployment.md](docs/ray_serve_deployment.md) |
| Workflow automation reference | [docs/workflow_reference.md](docs/workflow_reference.md) |
| Codebase status audits | [docs/codebase_status_report.md](docs/codebase_status_report.md) |
| Documentation maintenance checklist | [docs/documentation_maintenance.md](docs/documentation_maintenance.md) |

### Research & SDKs

| Topic | Location |
| --- | --- |
| Implementation status & roadmap alignment | [docs/implementation_status.md](docs/implementation_status.md) |
| Advanced fine-tuning and distributed inference plan | [docs/advanced_fine_tuning.md](docs/advanced_fine_tuning.md) |
| RAG context enrichment | [docs/rag_context_enrichment.md](docs/rag_context_enrichment.md) |
| Dolphin-X1 chat/embedding reuse | [docs/dolphin_x1_chat_embeddings.md](docs/dolphin_x1_chat_embeddings.md) |
| SDK overview & release workflow | [docs/sdk-overview.md](docs/sdk-overview.md), [docs/sdk-release-guide.md](docs/sdk-release-guide.md) |
| API specification & client stubs | [docs/api/](docs/api/README.md) |

Future milestones continue to live in [ROADMAP.md](ROADMAP.md).

## Contributing
Read the scoped contribution rules in `AGENTS.md` files before changing code.
Keep pull requests focused, document behavioural changes, and include a testing
summary in every PR. When adding optional integrations, provide fallbacks and
update documentation alongside the implementation.
