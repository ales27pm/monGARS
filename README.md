# monGARS

monGARS (Modular Neural Agent for Research and Support) is a privacy-first AI system designed for experimentation on low-resource hardware. The project bundles Python services, a Django web interface and container orchestration files so contributors can run or deploy the stack with minimal setup.

## Features

- **Conversational engine** powered by `LLMIntegration` with optional Ray Serve
  integration (URL configured via `RAY_SERVE_URL`). When `USE_RAY_SERVE` is set
  the service streams adapter metadata from `LLM_ADAPTER_REGISTRY_PATH`
  (`models/encoders` by default) so new LLM2Vec adapters are propagated to Ray
  replicas without manual restarts.
- **Memory management** through the in-memory `Hippocampus` and persistent storage via `PersistenceRepository`.
- **Adaptive behaviour** using the `MimicryModule`, `PersonalityEngine` and `AdaptiveResponseGenerator`.
- **Web scraping** utilities provided by `Iris` for retrieving external context.
- **Tiered caching** (memory, Redis and disk) with graceful fallback handling.
- **Self‑training and monitoring** via `SelfTrainingEngine` and `SystemMonitor`.
- **Evolution Engine** for autonomous diagnostics and code optimization.
- **Web interface** implemented with Django (located in `webapp/`).
  Conversation history is displayed on load using Django's `json_script` helper
  to safely embed data. Responses appear immediately after sending a message and
  the dark mode preference persists across sessions.
- **Automatic worker tuning** for Raspberry Pi and Jetson devices via `recommended_worker_count()`.
- **Worker deployment settings** configurable through `WORKER_DEPLOYMENT_NAME` and `WORKER_DEPLOYMENT_NAMESPACE` environment variables used by the Evolution Engine.
- **Robust core detection** falls back to logical CPUs if physical cores cannot be determined.
- **Encrypted peer-to-peer messaging** via `PeerCommunicator` for basic node coordination.
- **Peer registry endpoints** `/api/v1/peer/register`, `/api/v1/peer/unregister` and `/api/v1/peer/list` now require an admin token.
- **User management** with `/api/v1/user/register` to create new accounts. Invalid credentials return clear error responses.
- **Conversation endpoints** `/api/v1/conversation/chat` to generate responses
  with sanitized input and `/api/v1/conversation/history` to retrieve past
  interactions.
- **Distributed task scheduling** handled by `DistributedScheduler` to share work between peers.
- **Idle-time optimization** through `SommeilParadoxal` which triggers upgrades when the system is quiet.
- **Safe optimization cycles** executed by `EvolutionEngine.safe_apply_optimizations`.

A high level component overview can be found in `monGARS_structure.txt`.

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for local development)
- Optional: GPU drivers for Nvidia containers

### Installation

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or run the helper script:
   ```bash
   ./setup.sh
   ```
2. Copy `.env.example` to `.env` and adjust configuration values.
3. Initialize the database:
   ```bash
   python init_db.py
   ```
4. Launch services:
   ```bash
   docker-compose up
   ```
   This starts PostgreSQL, Redis, MLflow, Vault and an Ollama model server alongside the application.

### Running the Application

The main entry point is `main.py` which bootstraps monitoring tasks and exposes the FastAPI app under `monGARS.api.web_api`. During development you can run:

```bash
python main.py
```

On Raspberry Pi or Jetson boards the number of Uvicorn workers is
automatically reduced using `monGARS.utils.hardware.recommended_worker_count`.
The helper checks physical cores first and falls back to logical CPUs when
necessary.

The `/api/v1/peer/message` endpoint allows nodes to exchange encrypted messages
for basic coordination. This authenticated POST route accepts a JSON body of the
form `{"payload": "<encrypted>"}`.
Peers can register themselves via `/api/v1/peer/register`, remove themselves with
`/api/v1/peer/unregister` and retrieve the current list of nodes from
`/api/v1/peer/list`. These registry routes now enforce admin permissions.

Unit and integration tests are located in the `tests/` directory. Execute them with:

```bash
pytest
```

Code style is enforced using `black` and `isort` as outlined in `AGENTS.md`.

### Running the Django Web UI

The chat frontend shipped in `webapp/` can be launched alongside the FastAPI
service for manual testing or demos:

1. Install Django dependencies (already listed in `requirements.txt`).
2. Export the minimal environment expected by `webapp/webapp/settings.py`:

   ```bash
   export DJANGO_SECRET_KEY="change-me"  # required, no default provided
   export FASTAPI_URL="http://localhost:8000"  # point to the FastAPI process
   ```

3. Apply migrations and start the development server on an open port, e.g.:

   ```bash
   cd webapp
   python manage.py migrate
   python manage.py runserver 0.0.0.0:8001
   ```

4. Visit `http://127.0.0.1:8001/chat/login/` to authenticate. The login form
   proxies credentials to FastAPI's `/token` endpoint, stores the issued JWT in
   the session, and then redirects to the chat view. When JavaScript is
   disabled, conversations can still be submitted thanks to the progressive
   enhancement fallbacks implemented in `chat/index.html`.

## Security Hardening Quick Wins

- **Restrict WebSocket origins.** Set the `WS_ALLOWED_ORIGINS` environment variable
  to the exact frontend origins that should open WebSocket sessions
  (comma-separated or JSON list), for example
  `WS_ALLOWED_ORIGINS="https://app.example.com,https://admin.example.com"`.
- **Terminate TLS at the edge.** Serve `wss://` traffic through your ingress
  controller (Caddy, Nginx, Traefik, etc.) so certificates and TLS negotiation
  stay outside of the application container.
- **Apply rate limiting.** When a reverse proxy limit is not available, enable the
  built-in per-user token bucket by setting
  `WS_RATE_LIMIT_MAX_TOKENS` and `WS_RATE_LIMIT_REFILL_SECONDS` to throttle event
  fan-out directly in `ws_manager.py`.

## Deployment

Production deployments can be containerised via the provided `Dockerfile`. Kubernetes manifests live in `k8s/` for cluster environments. Adjust resource limits and RBAC rules as required for your infrastructure.
Use `./build_embedded.sh` with Docker Buildx to build and push multi-architecture images for Raspberry Pi and Jetson using `Dockerfile.embedded`. Provide a repository name (e.g. `user/image:tag`) and ensure you're logged in to your container registry.
The `build_native.sh` helper builds an optimized x86_64 image tuned for fast compiles on a typical Intel i7 workstation.

## Contributing

Please read `AGENTS.md` for the contribution guide. Pull requests should be focused, include relevant tests and ensure `pytest` passes before submission. Documentation updates are encouraged whenever behaviour changes.

## Roadmap

Development milestones are tracked in `ROADMAP.md`. Upcoming phases include hardware optimization, collaborative networking and a full web API. A detailed plan for fine-tuning and distributed inference is available in `docs/advanced_fine_tuning.md`.

Community feedback and contributions are welcome!
