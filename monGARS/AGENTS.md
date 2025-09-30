# monGARS Package Standards

These rules apply to everything under `monGARS/`, including FastAPI surfaces,
core cognition services, persistence helpers, and shared utilities. Follow the
root playbook in `/AGENTS.md` and treat the guidelines below as additive.

## Configuration & Dependency Injection
- Always resolve configuration with `monGARS.config.get_settings()` and share the
  cached object. Constructing new settings instances breaks feature-flag parity
  across services.
- Inject collaborators via constructors or dependency providers in
  `monGARS.api.dependencies`. Avoid module-level singletons so tests can replace
  implementations with fakes.

## Optional Integrations
- Gate heavy imports (Torch, Ray, transformers, GPU tooling) behind availability
  checks. Provide functional fallbacks that log a warning and degrade gracefully.
- Place adapters or provider abstractions in `monGARS.utils` or dedicated modules
  so callers can select optional backends without editing orchestration logic.

## Logging & Telemetry
- Use `logging.getLogger(__name__)` and emit structured dictionaries. Redact
  `user_id`, tokens, and other sensitive identifiers before logging.
- When adding retries or background loops, surface counters/timers through
  OpenTelemetry to keep observability consistent across the stack.

## Persistence & State
- Route database access through `core.persistence.PersistenceRepository` and keep
  schemas defined in `init_db.py`. New tables require a matching Alembic
  migration and documentation in `docs/`.
- Prefer the caching utilities in `core.caching` and `core.hippocampus` instead
  of ad-hoc dictionaries. Clear caches explicitly in tests.

## Testing Expectations
- Update targeted tests when you modify behaviour:
  - `tests/test_api_*.py` for FastAPI changes
  - `tests/test_hippocampus.py`, `tests/test_personality.py`, `tests/test_peer_communication.py`
    for cognition and messaging updates
  - `tests/test_mntp_trainer.py` and `tests/test_evolution_engine.py` for ML and
    orchestration adjustments
- Provide fixtures or monkeypatches for external calls so the suite remains
  hermetic and fast.
