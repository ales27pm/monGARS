# Core Engine Guidelines

Components here coordinate perception, memory, reasoning, and output for
monGARS. Most public methods are asynchronous and orchestrate multiple
subsystems.

## Architectural Patterns
- Keep orchestrators thin. `conversation.ConversationalModule` composes
  Hippocampus, LLM integration, curiosity, mimicry, and persistence services.
  New collaborators should expose async interfaces and be injected through the
  constructor for testability.
- Reuse shared services: `persistence.PersistenceRepository` for storage,
  `hippocampus.Hippocampus` for memory, `distributed_scheduler.DistributedScheduler`
  for background work, and `peer.PeerCommunicator` for messaging. Avoid
  duplicating logic already encapsulated in these modules.
- When adding cognitive behaviours (e.g. new curiosity heuristics or sleep
  cycles), update module-level docstrings to capture the rationale and mention
  any tunable constants.

## Asynchrony & Performance
- Preserve async workflows end-to-end. Methods such as
  `ConversationalModule.generate_response`, `MimicryModule.adapt_response_style`,
  and `SommeilParadoxal.run_cycle` are awaited across the stack. Use
  `asyncio.to_thread` if you must call blocking libraries.
- Leverage caching helpers under `core/caching/` and avoid long-lived globals.
  Clear caches in tests via fixtures (see `tests/test_tiered_cache.py`).

## Logging & Security
- Use structured logs (`logger.info("event", extra={...})` style) and include
  correlation identifiers (user ID, session ID, request ID). Never log secrets or
  raw tokens—redact sensitive values manually before emitting logs.
- Security-sensitive flows must rely on `core.security.SecurityManager` and
  `core.security.validate_user_input` to prevent inconsistent sanitisation.

## Testing
- Update relevant tests under `tests/` when modifying this package:
  conversation entry points surface through `test_api_chat.py`, while lower-level
  modules are covered by `test_hippocampus.py`, `test_personality.py`,
  `test_sommeil.py`, `test_distributed_scheduler.py`, and
  `test_peer_communication.py`.
- Provide deterministic behaviour in tests by monkeypatching randomness (see
  `test_social_media.py`) and clearing shared state in fixtures.
