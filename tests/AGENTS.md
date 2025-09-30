# Testing Standards

The `tests/` directory uses `pytest` with async fixtures and heavy
monkeypatching. Follow the project-wide tooling rules plus the specifics below.

## Structure & Conventions
- Async tests must declare `pytest.mark.asyncio` and rely on fixtures from
  `conftest.py`. Never create event loops manually.
- Reset shared state (`hippocampus._memory`, caches, global singletons) inside
  fixtures to keep tests independent.
- Stub external dependencies (Torch, spaCy, HTTP clients) with lightweight fakes
  or patching utilities.

## Coverage Expectations
- Cover both success and failure paths for HTTP endpoints, WebSockets, and
  orchestration flows.
- Long-running paths (`chaos_test.py`, `integration_test.py`,
  `self_training_test.py`) are mandatory before releases or infrastructure-heavy
  changes.
- Keep property-based tests scoped and fast by constraining strategy sizes when
  adding new properties.

## Tooling
- Default command: `pytest -q`.
- Focused runs: `pytest -k <pattern>`.
- Fail-fast triage: `pytest --maxfail=1`.
- Generate coverage reports with `pytest --cov=monGARS --cov=modules` when
  requested by maintainers.
