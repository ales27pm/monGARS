# monGARS Package Guidelines

This package contains the FastAPI surface, cognition pipelines, persistence
layers, and shared utilities. Respect the repository-wide standards described in
the root `AGENTS.md` and apply these specifics when working under `monGARS/`.

## Configuration & Settings
- Always fetch configuration via `monGARS.config.get_settings()`. Modules such as
  `core.conversation` and `utils.hardware` rely on the cached settings object;
  creating new instances manually can desynchronise feature flags.
- Validate environment-dependent inputs early. For example, `core.security`
  raises on invalid JWT settings—follow that approach for new secrets or API
  tokens.

## Dependency Management
- Avoid global singletons. Inject collaborators through constructors or via the
  providers in `monGARS.api.dependencies` so tests can swap implementations. The
  `ConversationalModule` and WebSocket manager demonstrate the preferred pattern.
- Keep optional integrations (OpenAI, Torch, etc.) behind feature checks in
  `utils` or dedicated adapters. Do not import them at module load time without
  fallbacks.

## Logging & Observability
- Use module-level loggers (`logging.getLogger(__name__)`) and include context
  such as `user_id`, `session_id`, or model identifiers. Sensitive fields must be
  redacted before logging.
- Emit structured dictionaries when raising or propagating errors so FastAPI and
  background workers can surface actionable telemetry.

## Persistence & State
- Route database interactions through `core.persistence.PersistenceRepository`
  and schema definitions in `init_db.py`. If you need new tables, add SQLModel
  definitions there and update the Alembic migrations.
- When caching, prefer the utilities in `core.caching` and `core.hippocampus`
  rather than ad-hoc dictionaries.

## Testing
- Updates here usually require adjustments under `tests/`: see
  `test_hardware_utils.py`, `test_hippocampus.py`, `test_peer_communication.py`,
  `test_personality.py`, and `test_token_security.py` for reference patterns.
- Provide fixtures or monkeypatches when you introduce new external calls so the
  suite remains hermetic and fast.
