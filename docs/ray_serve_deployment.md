# Ray Serve Deployment Guide

This guide describes how to provision Ray Serve for monGARS, configure the
application to use it, and understand the fallback behaviour when Ray is not
available. It assumes you have Docker (optional), Python 3.11, and Ray
installed on the target hosts.

## 1. Provision Ray on the control plane

1. Install Ray with the extras required for Serve and HTTP support:
   ```bash
   pip install "ray[serve]" "ray[default]"
   ```
2. Start a head node. Use the `--dashboard-host` flag if you need to expose the
   dashboard beyond localhost:
   ```bash
   ray start --head --dashboard-host=0.0.0.0
   ```
3. Add worker nodes by pointing them at the head node's address. Replace
   `<HEAD_NODE_IP>` with the externally reachable IP or DNS name of the head
   node:
   ```bash
   ray start --address='<HEAD_NODE_IP>:6379'
   ```

## 2. Deploy the Serve application

1. Package your deployment script (for example `serve_app.py`) that exposes a
   `deployment` handle accepting the same payload shape produced by
   `LLMIntegration.generate_response`.
2. Use the Serve CLI to submit the deployment:
   ```bash
   serve deploy serve_app:deployment
   ```
3. Confirm the deployment is healthy:
   ```bash
   serve status
   ```
   The output should show your application in the `HEALTHY` state.

## 3. Configure monGARS

monGARS discovers the Ray Serve endpoints through environment variables:

* `USE_RAY_SERVE`: set to `true` to keep Ray enabled. If omitted the
  integration defaults to `true` when `RAY_SERVE_URL` is present.
* `RAY_SERVE_URL`: comma-separated list of Ray Serve HTTP endpoints, e.g.
  `http://ray-head:8000/generate`. When omitted monGARS falls back to
  `http://localhost:8000/generate` if Ray is enabled.
* `RAY_CLIENT_TIMEOUT`, `RAY_CLIENT_CONNECT_TIMEOUT`,
  `RAY_CLIENT_MAX_CONNECTIONS`, `RAY_CLIENT_MAX_KEEPALIVE` allow tuning the
  HTTP client Ray Serve uses for inference.
* `RAY_SCALING_STATUS_CODES` and `RAY_SCALING_BACKOFF` control the response
  codes that trigger exponential backoff before retrying requests while Ray is
  scaling replicas.
* `RAY_MAX_SCALE_CYCLES` limits the number of scaling retries attempted before
  falling back to an alternative provider.

Set these variables in your `.env` file or deployment environment. Restart the
FastAPI workers after updating the configuration so `LLMIntegration` reloads the
settings.

## 4. Fallback behaviour when Ray is unavailable

`LLMIntegration` automatically falls back to the local provider chain when Ray
Serve is unreachable or disabled:

* If `USE_RAY_SERVE=false` or `RAY_SERVE_URL` resolves to an empty list, Ray is
  disabled and monGARS uses the local Ollama models when available.
* When Ray is enabled but an HTTP request fails with scaling status codes (503,
  409 by default), the integration performs exponential backoff and rotates
  through the configured endpoints before declaring a failure.
* If all retries fail or the Ray client cannot be initialised, the integration
  logs the failure and invokes the local provider path so requests still
  succeed.

Review the log stream for events named `llm.ray.disabled`, `llm.ray.backoff.*`,
`llm.ray.request.failure`, and `llm.cache.*` to monitor the active execution
path. These metrics help you confirm when the application is using Ray and when
it has fallen back to local inference.

