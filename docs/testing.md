# Testing Guidance

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The monGARS quality gates span Python services, the Django operator console,
the React Native client, and container smoke tests. This guide keeps the
instructions dynamic by recording the current matrix, runtime expectations, and
the command variants that engineering and operations rely on.

## End-to-end test matrix

| Layer | Command | Expected duration | Notes |
| --- | --- | --- | --- |
| Backend (FastAPI, cognition, evolution engine) | `pytest` | ~110 s | Runs 631 tests covering API contracts, curiosity heuristics, reinforcement loops, and adapter manifests. Mirrors the CI `python-tests` job. |
| Backend lint & formatting | `make lint` or `ruff check` / `black --check` (see `pyproject.toml`) | ~35 s | CI drives these via the `python-quality` job after scope detection. |
| Frontend (Django assets) | `npm test -- --watch=false` inside `webapp/` | ~45 s | Executes Jest with coverage. CI additionally runs `npm run build` to confirm the production bundle compiles. |
| React Native client | `npm test -- --watch=false` inside `mobile-app/` | ~60 s | Uses the shared Jest config, TypeScript `--noEmit`, and ESLint (mirrors the `mobile-quality` workflow job). |
| TypeScript lint (root + mobile) | `npm run lint` (repository root) | ~25 s | Validates shared tooling and Django UI assets. React Native lint executes via the `mobile-quality` job. |
| Container smoke test | `docker compose -f docker-compose.yml up --build --abort-on-container-exit` followed by `pytest --maxfail=1 --disable-warnings -k "not long"` inside the app container | ~6 min | Mirrors the CI Docker job. Use after dependency upgrades or Dockerfile edits. |

> Tip: The scope detection in `.github/workflows/ci.yml` skips irrelevant jobs on
> pull requests. When working locally, prefer the granular commands above instead
> of `npm test` / `pytest` combos that rebuild dependencies repeatedly.

## Recommended local workflows

### Full backend verification
```bash
pytest
```

Use `pytest -q` for quieter logs and `pytest -k <expression>` to focus on a
subset during debugging. When touching persistence or Hippocampus internals,
run `pytest tests/test_persistence.py tests/test_conversation.py` to surface
regressions quickly.

### Frontend & mobile spot checks
```bash
pushd webapp
npm test -- --watch=false
npm run lint
popd

pushd mobile-app
npm test -- --watch=false
npm run lint
npm run typecheck
popd
```

`npm run typecheck` wraps `tsc --noEmit` and matches the workflow job. Run it
whenever you touch shared typings or API clients.

### Integration sweeps before a release
```bash
make lint
pytest
npm run lint
pushd webapp && npm test -- --watch=false && popd
pushd mobile-app && npm run lint && npm test -- --watch=false && npm run typecheck && popd
```

Capture the outputs in the pull request summary. Reviewers expect to see the
exact command plus the final status.

## Coverage and reporting expectations
- The CI `python-tests` job fails if coverage drops below **84%** overall. When
  adding new modules under `monGARS/core` or `modules/`, include targeted tests
  to preserve the threshold.
- Frontend and mobile jobs upload Jest coverage artefacts. When the coverage
  delta is meaningful (±1%), note it in the PR description so release managers
  can track UI test health.
- Use `coverage html` locally if you need to inspect missed lines before
  resubmitting a change.

## Troubleshooting playbook
- **Hanging backend tests** – confirm Redis/PostgreSQL containers are not
  running from a previous Compose session. The default suite mocks external
  services, so lingering sockets can block teardown.
- **Out-of-memory during Jest runs** – export `NODE_OPTIONS="--max-old-space-size=4096"`
  before invoking `npm test` on lower-memory machines.
- **Mobile TypeScript errors** – run `npm run clean` inside `mobile-app/` to
  remove Metro caches, then re-run `npm install`.
- **Docker smoke-test failures** – inspect the combined logs captured in
  `deployment.log` when using `scripts/full_stack_visual_deploy.py` or run
  `docker compose logs` after the smoke test exits.

Keep this document updated whenever commands, thresholds, or workflow
behaviour changes. The tables are the fastest way for operators to verify that
local execution matches CI, so stale data quickly causes confusion.

## Keeping this guide current
- Update the command table whenever CI adds or removes jobs. Reference the
  workflow name and job ID in the pull request so reviewers can validate the
  change.
- When coverage thresholds or counts move, edit both the bullets above and
  [docs/implementation_status.md](implementation_status.md) to keep release
  readiness dashboards accurate.
- Note the exact command outputs you ran in the PR description; this creates an
  audit trail that mirrors the expectations in the
  [Documentation Maintenance Checklist](documentation_maintenance.md).
