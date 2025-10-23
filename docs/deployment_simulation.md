# Deployment Simulation Runbook

The deployment simulator provides a fast, deterministic pre-flight check for
operators who want to validate configuration before promoting a build to
staging or production. It inspects FastAPI configuration, Docker Compose
manifests, and Kubernetes deployments without requiring access to Docker or a
cluster. This lets CI pipelines and local development environments catch
misconfigurations early.

## Usage

Run the simulator from the repository root:

```bash
python scripts/deployment_simulator.py
```

The command prints a summary of detected issues and exits with a non-zero code
when errors are present. Use `--strict` to treat warnings as failures, or
`--json` to emit machine-readable output for CI workflows.

```bash
python scripts/deployment_simulator.py --strict --json
```

The simulator accepts additional Compose and Kubernetes manifests via
`--compose` and `--k8s` flags. For example, to validate both the production and
GPU Compose stacks:

```bash
python scripts/deployment_simulator.py \
  --compose docker-compose.yml \
  --compose docker-compose.gpu.yml
```

## Checks Performed

- **Application settings**: loads `monGARS.config.Settings`, validates JWT
  configuration, and flags missing or ephemeral `SECRET_KEY` values before a
  production rollout.
- **Docker Compose manifests**: verifies that every service defines an `image`
  or `build`, detects duplicate host ports, surfaces missing `env_file`
  references, and warns about bind mounts that point at absent directories.
- **Kubernetes deployments**: ensures each container declares an image,
  highlights missing resource requests/limits, and reports missing readiness or
  liveness probes that could stall rollouts.

## When to Run

- Prior to updating `docker-compose` or `k8s` manifests.
- After changing configuration defaults in `monGARS.config.Settings`.
- Inside CI pipelines to block merges that would crash-loop on deployment.

The simulator supplements, but does not replace, full integration tests such as
`pytest` and end-to-end smoke tests. Use it alongside those suites to achieve a
comprehensive deployment readiness check.
