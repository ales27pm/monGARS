# Ray Serve Deployment Guide

This guide outlines how to provision Ray Serve for monGARS, configure the
application, and understand fallback behaviour when the cluster is unavailable.
It assumes Ray 2.x, Python 3.11, and optional Docker/Kubernetes familiarity.

## 1. Provision a Ray Cluster
1. Install Ray with Serve extras:
   ```bash
   pip install "ray[serve]" "ray[default]"
   ```
2. Start the head node (expose the dashboard if you need remote access):
   ```bash
   ray start --head --dashboard-host=0.0.0.0
   ```
3. Join worker nodes to the head node:
   ```bash
   ray start --address='<HEAD_NODE_IP>:6379'
   ```
4. Verify cluster health with `ray status`.

### Docker Compose (local)

When running the stack with Docker Compose, the default configuration now uses
the CPU-only Ray image so environments without NVIDIA drivers can launch the
head node successfully. Set `COMPOSE_PROFILES=ray` before invoking `docker
compose up` to include the Ray services. If you _do_ have GPUs available,
override the image by exporting `RAY_HEAD_IMAGE=rayproject/ray:2.9.3-py311-cu121`
or by using `docker-compose.gpu.yml`, which reintroduces the GPU device
reservations. Without that override, Docker will surface `could not select
device driver "" with capabilities: [[gpu]]` errors because the default engine
cannot satisfy the GPU reservation that Ray previously required.

For Kubernetes deployments, reuse the manifests/Helm chart produced from
`modules/ray_service.py` so replica configuration matches the application.

## 2. Deploy a Serve Application
1. Use the provided deployment in `modules/ray_service.py`. It loads Ollama or
   adapter artefacts, streams events, and exposes an HTTP endpoint compatible
   with `LLMIntegration.generate_response` payloads.
2. Submit the deployment:
   ```bash
   python -m modules.ray_service --deploy
   ```
   or package your deployment script (e.g. `serve_app.py`) exposing a
   `deployment = LLMDeployment.bind(...)` handle mirroring the schema returned by
   Ray responses in tests (`tests/test_llm_ray.py`).
3. Confirm the deployment is healthy:
   ```bash
   serve status
   ```
   The status should report `HEALTHY` for your application.

## 3. Configure monGARS
Set the following environment variables (e.g. in `.env` or Kubernetes secrets):

| Variable | Purpose |
| --- | --- |
| `USE_RAY_SERVE` | Enable (`true`) or disable (`false`) the integration. Defaults to `true` when a URL is provided. |
| `RAY_SERVE_URL` | Comma-separated list of Serve HTTP endpoints, e.g. `http://ray-head:8000/generate`. |
| `RAY_CLIENT_TIMEOUT` / `RAY_CLIENT_CONNECT_TIMEOUT` | Adjust client-level timeouts for long-running requests. |
| `RAY_CLIENT_MAX_CONNECTIONS` / `RAY_CLIENT_MAX_KEEPALIVE` | Tune connection pooling. |
| `RAY_SCALING_STATUS_CODES` / `RAY_SCALING_BACKOFF` / `RAY_MAX_SCALE_CYCLES` | Control retry strategy while replicas scale up. |
| `LLM_ADAPTER_REGISTRY_PATH` | Optional override for the adapter manifest directory shared with Ray Serve. |

Restart FastAPI workers after changing configuration so `LLMIntegration`
reloads its settings and adapter metadata.

## 4. Fallback Behaviour
`LLMIntegration` degrades gracefully when Ray is unreachable:
- When `USE_RAY_SERVE=false` or `RAY_SERVE_URL` resolves to an empty list, the
  service uses the local Ollama providers.
- For scaling responses (HTTP 503/409 by default), the client performs
  exponential backoff driven by `RAY_SCALING_BACKOFF`, rotates through the
  configured endpoints, and honours `Retry-After` headers.
- After exhausting retries or when the Ray client cannot initialise, the
  integration logs the failure and routes the request through the local provider
  chain (`llm.ray.fallback_local`).

Monitor structured logs (`llm.ray.*`) and add OpenTelemetry counters for success,
error, and fallback paths when you extend the integration with additional
telemetry.

## 5. Rolling Out New Adapters
- Self-training runs update manifests in `LLM_ADAPTER_REGISTRY_PATH` and emit
  checksums. Ray Serve deployments detect these changes and reload weights when
  the evolution engine triggers refresh events.
- Use `python -m scripts.provision_models` to ensure Ollama baselines are
  available before Ray replicas start, preventing cold-start cascades.
- When deploying across clusters, synchronise the adapter directory (e.g. via
  object storage) so Ray replicas observe the same manifest versions as the API
  pods.

## 6. Database Preparation
Apply Alembic migrations before bootstrapping Ray Serve so the API layer and
background workers have the expected persistence schema:

1. Configure `DATABASE_URL` (and related pool settings) in the environment or
   `.env` file.
2. Run `python init_db.py` to apply migrations. The script now drives Alembic
   directly and converts existing `conversation_history.vector` columns to
   JSON/JSONB so embeddings remain portable across environments.
3. Confirm the migration by inspecting the latest revision:
   ```bash
   alembic current
   ```
   To roll back in staging, execute `alembic downgrade 20250108_03`.

The conversion away from the `pgvector` column type is backward-compatible; the
new JSON payloads preserve existing embedding data while avoiding hard
dependencies on the extension in environments that lack it.

## 7. Observability & Telemetry

- `LLMIntegration` publishes `llm.ray.requests`, `llm.ray.failures`,
  `llm.ray.scaling_events`, and `llm.ray.latency` metrics via OpenTelemetry.
- Forward these counters/histograms to your metrics backend (Prometheus, OTLP
  collector, etc.) and alert when failure ratios or latency percentiles drift.
- Combine Ray metrics with scheduler gauges to understand whether throttling
  originates from inference, queuing, or upstream cognition workloads.
