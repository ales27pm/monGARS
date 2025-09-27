# monGARS Contribution Guide

This repository powers **monGARS**, a modular AI assistant composed of a FastAPI
service (`monGARS.api`), an orchestration and cognition layer (`monGARS.core`),
a Django frontend (`webapp`), and optional research modules under `modules/`.
Use this document together with the scoped `AGENTS.md` files deeper in the tree.

## Environment & Tooling
- Python 3.11 is the primary runtime. `requirements.txt` mirrors the current
  production stack; keep it in sync with any dependency changes.
- Format with `black` and organise imports with `isort` before committing.
  Both tools are configured implicitly through the project defaults.
- Type hints are required for new or modified public functions. Prefer
  `collections.abc` generics over concrete types for callables and iterables.
- Central configuration lives in `config.py` and `.env` files loaded by
  `monGARS.config.get_settings`. Never hard-code credentials or URLs.

## Tests & Quality Gates
- Run `pytest -q` locally. The suite exercises FastAPI (`tests/test_api_*.py`),
  cognition (`tests/test_hippocampus.py`, `tests/test_personality.py`), and the
  optional ML flows (`tests/test_mntp_trainer.py`). Add coverage for new code.
- Use `pytest -k <keyword>` to focus on a subset of tests while iterating.
  `chaos_test.py`, `integration_test.py`, and `self_training_test.py` cover
  longer paths—run them before release branches.
- Keep fixtures isolated—see `tests/conftest.py` for hippocampus resets and
  monkeypatch patterns.

## Documentation Expectations
- Update `README.md`, `docs/`, or `monGARS_structure.txt` whenever behaviour or
  deployment expectations change. Match the tone and structure already present.
- Record significant architecture changes or recurring incidents in this file
  so future updates can reference the rationale.

## Deployment & Operations
- `docker-compose.yml` spins up the FastAPI stack with Postgres and Redis
  locally. `build_native.sh` and `build_embedded.sh` produce x86_64 and ARM
  images; keep their flags aligned with the Dockerfiles.
- Kubernetes manifests live in `k8s/`. When adding services, extend the RBAC
  rules in `rbac.yaml` and document any new environment variables.
- Background workers reference `DistributedScheduler` and peer messaging over
  `/api/v1/peer/*`. Ensure any new long-running process emits structured logs
  (JSON or key=value) for the observability pipeline.

## Pull Requests
1. Keep PRs focused and describe both implementation and testing in detail.
2. Run `black`, `isort`, and `pytest` before requesting review.
3. Squash commits when merging unless the history is intentionally split for
   release notes.
4. Reference relevant ROADMAP items or issues in the PR description when you
   progress them.
