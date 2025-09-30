# monGARS Contributor Playbook

This document defines the global expectations for any change proposed to the
repository. Every nested `AGENTS.md` adds requirements that build on top of
these rules—read the file in your target directory before you edit code.

## Supported Toolchain
- **Python runtime**: 3.11. Keep `requirements.txt` aligned with the effective
  production set of packages when you bump or add dependencies.
- **Formatting**: run `black` and `isort` on every patch. The default
  configuration is provided by `pyproject.toml`—do not supply custom flags.
- **Typing**: prefer typing constructs from `collections.abc` and annotate every
  new public function or method. Preserve existing annotations when you refactor
  code.

## Configuration & Secrets
- Access settings through `monGARS.config.get_settings()` so cached configuration
  stays consistent. Never fetch environment variables directly inside
  application code unless you are writing bootstrap logic in `config.py`.
- Secrets must be injected via environment variables or Vault. Avoid committing
  `.env` files, credentials, URLs with embedded keys, or any sample values that
  resemble production data.

## Quality Gates
- **Tests**: `pytest -q` must pass locally. Targeted runs (`pytest -k ...`) are
  acceptable while iterating, but run the full suite before submitting a PR.
- **Long-running suites**: execute `chaos_test.py`, `integration_test.py`, and
  `self_training_test.py` before cutting a release branch or when your changes
  touch distributed scheduling, self-training, or infrastructure code paths.
- **Observability**: keep structured logging (`logger.info("event", extra={...})`)
  and OpenTelemetry metrics consistent. When you add new long-lived services,
  expose high-level success/failure counters.

## Documentation Discipline
- Update `README.md`, `docs/`, and `monGARS_structure.txt` whenever you change
  behaviour, deployment steps, or architecture diagrams. Every feature flag or
  new environment variable must be documented where operators will find it.
- Capture context for consequential changes in this file so future contributors
  understand why conventions evolved.

## Deployment Expectations
- `docker-compose.yml` provisions FastAPI, Postgres, Redis, MLflow, Vault, and a
  local Ollama server. Keep service names, ports, and volume mounts in sync when
  modifying APIs or storage paths.
- `build_native.sh` and `build_embedded.sh` must mirror the Dockerfiles. If you
  tweak build args, update both scripts and note the change in `README.md`.
- Kubernetes manifests live under `k8s/`. Extend RBAC rules (`rbac.yaml`) when
  you add controllers or background workers, and document new environment
  variables alongside the manifest update.

## Pull Request Protocol
1. Keep PRs scoped to a coherent change set. Link to ROADMAP items or issues when
   you advance them.
2. Include a testing summary (`pytest`, linting, type checks, etc.) in the PR
   description. If you skip a gate, explain why.
3. Squash commits on merge unless the history is intentionally split for release
   notes.
