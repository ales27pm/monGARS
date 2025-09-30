# Core Engine Standards

`monGARS/core/` contains the cognition pipeline, memory abstractions, scheduling
primitives, and adaptive behaviour modules. Extend the root and package
guidelines with the rules below.

## Architectural Patterns
- Keep orchestrators thin. Inject collaborators through constructors and expose
  asynchronous interfaces (`async def`) so tests can patch behaviours.
- Reuse existing services (`Hippocampus`, `PersistenceRepository`,
  `DistributedScheduler`, `PeerCommunicator`) instead of introducing parallel
  state machines.
- Document new behaviours in module docstrings, including tuning knobs or
  heuristics, so the reasoning behind defaults is preserved.

## Asynchrony & Performance
- Preserve async flows end-to-end. Wrap blocking libraries with
  `asyncio.to_thread` or queue them on background executors.
- Use caching utilities in `core/caching` and clear them in fixtures. Avoid global
  state that survives between tests.

## Logging, Metrics, and Safety
- Emit structured logs with correlation identifiers (user, session, request).
  Redact secrets and tokens manually before logging.
- Publish high-level counters or gauges through OpenTelemetry when adding new
  loops, retries, or backoff strategies.
- Handle optional ML dependencies defensively—log unavailability and return safe
  fallbacks rather than raising unless the caller explicitly requires a hard
  failure.

## Testing Expectations
- Update targeted tests when cognition changes:
  - `tests/test_hippocampus.py`, `tests/test_personality.py`,
    `tests/test_mimicry.py`, `tests/test_sommeil.py`
  - `tests/test_evolution_engine.py`, `tests/test_distributed_scheduler.py`,
    `tests/test_peer_communication.py`
- Mock randomness, time, and heavy integrations to keep the suite deterministic
  and fast. Reset shared memory/state within fixtures.
