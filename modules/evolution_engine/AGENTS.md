# Evolution Engine Standards

`modules/evolution_engine/` coordinates adapter retraining and operational
self-healing.

## Orchestration Principles
- Keep `EvolutionOrchestrator` focused on sequencing. Instantiate trainers from
  `modules.neurons.training` and return artifact metadata; avoid mutating global
  state.
- Represent pipeline stages with explicit methods (validation, training,
  deployment) so tests can exercise them individually.
- Wrap external calls with targeted exception handling. Log the action, propagate
  unexpected errors, and ensure callers receive actionable messages.

## Configuration
- Default configs should point to `configs/training/mntp_mistral_config.json`.
  Accept overrides via constructor arguments for deterministic tests and batch
  jobs.
- Document new config keys and update the README/roadmap when pipeline defaults
  change.

## Testing
- Extend `tests/test_evolution_engine.py` to cover new stages, success/failure
  cases, and artifact validation.
- Replace `MNTPTrainer` with fakes in fixtures when unit testing so tests remain
  fast and hermetic.
