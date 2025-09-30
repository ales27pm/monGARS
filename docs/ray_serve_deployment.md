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

## 2. Deploy a Serve Application
1. Package your deployment script (e.g. `serve_app.py`) exposing a
   `deployment = LLMDeployment.bind(...)` handle that mirrors the payload emitted
   by `LLMIntegration.generate_response`.
2. Submit the deployment:
   ```bash
   serve deploy serve_app:deployment
   ```
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

Restart FastAPI workers after changing configuration so `LLMIntegration` reloads
its settings.

## 4. Fallback Behaviour
`LLMIntegration` degrades gracefully when Ray is unreachable:
- When `USE_RAY_SERVE=false` or `RAY_SERVE_URL` resolves to an empty list, the
  service uses the local Ollama providers.
- For scaling responses (HTTP 503/409 by default), the client performs
  exponential backoff and rotates through the configured endpoints before
  declaring a failure.
- After exhausting retries or when the Ray client cannot initialise, the
  integration logs the failure and routes the request through the local provider
  chain.

Monitor structured logs for events such as `llm.ray.disabled`,
`llm.ray.backoff.*`, and `llm.ray.request.failure` to understand which path is in
use. Add OpenTelemetry counters if you extend the integration with additional
fallback logic.
