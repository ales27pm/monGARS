# Workflow reference

> **Last updated:** 2025-10-23 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The repository ships two GitHub workflows that back day-to-day delivery:
`CI Quality Gate` (`.github/workflows/ci.yml`) and `Generate signing CSR`
(`.github/workflows/Csr.yml`). This note documents what each job enforces,
when it runs, and how operators can interact with the dispatch inputs.

## CI Quality Gate

### Scope detection and fan-out

The `changes` job uses `dorny/paths-filter@v3` to compute which portions of
the tree a pull request touches. It records booleans for Python, the Django
web UI, the React Native client, Docker assets, docs, and workflow files,
then writes a quick summary to the run log. Downstream jobs use those
outputs so that documentation-only pull requests skip heavy mobile
installs, while pushes to `main` still execute the full gate for safety.

### Python pipeline

Python quality gates execute in three layers:

1. **`python-quality`** installs Black, isort, and Flake8, verifying style
   and lint in a cached Python 3.11 environment.
2. **`python-typecheck`** installs the full backend requirements plus
   `mypy`, runs `pip check`, and performs static analysis with the
   configuration stored in `pyproject.toml`.
3. **`python-tests`** installs the backend stack, executes
   `coverage run -m pytest`, generates XML and HTML coverage artefacts, and
   attaches the coverage report to the job summary. The coverage gate is
   calibrated to fail when overall backend coverage falls below 84%, giving
   teams space to backfill historical gaps while still flagging regressions.
   Artefacts include the `.coverage` file, XML coverage report, HTML report,
   and the pytest cache for troubleshooting failures.

Every pull request that touches backend, workflow, docker, documentation,
or frontend assets triggers these steps, while pushes to `main` always run
all three.

### Web UI checks

`node-quality` provisions Node.js 18 with npm caching, runs `npm ci` with
fund/audit disabled, lints `webapp/static/js` via ESLint, runs Jest in CI
mode with coverage enabled, and builds the production chat bundle. The job
publishes both the generated `webapp/static/js/chat.js` file and the Jest
coverage output so reviewers can inspect the bundle that shipped.

### React Native checks

`mobile-quality` mirrors the frontend job for the React Native client under
`mobile-app/`. It runs `npm ci`, executes ESLint, performs a strict TypeScript
`tsc --noEmit`, and runs the Jest suite in CI mode with coverage enabled.
Coverage output and any generated JUnit XML artefacts upload for diagnosis
without blocking the workflow when Jest decides not to emit the files.

### Repository inventory

`inventory` waits for the quality gates to finish, installs the backend
stack once, and calls `.tools/inventory.py` against the PR and base tree.
When differences exist it posts a trimmed diff (200 lines) into the job
summary, creating a paper trail of new modules, FastAPI routes, and
configuration keys. Runs on push events to update the baseline record even
when no diff exists.

### Docker runtime validation

`docker-build` runs after the quality jobs on pushes and manual dispatches
that touch backend, docker, workflow, or frontend assets. It builds the
runtime stage of `Dockerfile` with `pytorch/pytorch:2.2.2-cpu`, then performs
an abbreviated `pytest --maxfail=1 --disable-warnings -k "not long"` inside
the container. The smoke test ensures the packaged environment still boots
pytest without re-running the entire suite twice.

## Generate signing CSR

The dispatchable CSR workflow now accepts full distinguished-name metadata,
including organization, organizational unit, locality, state, country, and
optional Subject Alternative Names. Operators can choose RSA (2048/3072/4096
bits) or EC keys (prime256v1 or secp384r1). A Python helper renders the
OpenSSL config and subject string, ensuring SAN entries get encoded as
extensions.

The job emits SHA-256 digests for both the CSR and key, writes the full CSR
text to `certs/signing.csr.txt`, and appends the subject, key algorithm, key
parameters, SANs, and public-key fingerprint to the run summary. Artefacts
ship as a single bundle with a three-day retention policy so secrets do not
linger indefinitely.

Refer back to `docs/deployment_automation.md` for guidance on how the Docker
image and CI gate feed into the visual deployment orchestrator.
