# Testing Guidelines

The suite uses `pytest` with heavy reliance on async fixtures and monkeypatching
for external services.

## Structure & Conventions
- Async tests must use `pytest.mark.asyncio` and rely on fixtures defined in
  `conftest.py`. Avoid creating event loops manually—`pytest-asyncio` handles
  lifecycle management.
- Reset shared state in fixtures. For example, tests touching the hippocampus
  clear `hippocampus._memory` and `_locks`. Follow that pattern whenever you add
  new singletons.
- Patch heavy dependencies (`torch`, `spacy`, HTTP clients) with lightweight
  stubs. See `test_api_chat.py` and `test_mntp_trainer.py` for reference.

## Coverage Expectations
- Exercise both success and failure paths. For endpoints, ensure you cover
  authentication, validation errors, and happy paths (`test_api_chat.py`,
  `test_user_management.py`).
- Long-running components like the distributed scheduler or self-training flows
  have dedicated tests (`test_distributed_scheduler.py`, `self_training_test.py`).
  Update them when modifying related behaviour.
- Property-based checks in `property_test.py` validate stability of core
  invariants. Keep them fast by constraining strategy sizes if you introduce new
  properties.

## Tooling
- Run `pytest -q` before submitting patches. Use `pytest -k <name>` to scope
  runs while iterating and `pytest --maxfail=1` when triaging regressions.
- Generate coverage reports with `pytest --cov=monGARS --cov=modules` when
  requested by maintainers.
