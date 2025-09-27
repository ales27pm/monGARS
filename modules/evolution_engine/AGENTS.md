# Evolution Engine Guidelines

`EvolutionOrchestrator` in `orchestrator.py` coordinates encoder refresh
pipelines that invoke the MNTP trainer and persist artifacts under
`models/encoders/`.

- Treat the orchestrator as a pure coordinator. It should assemble trainers from
  `modules.neurons.training` and return artifact paths without mutating global
  state beyond logging.
- Keep pipeline stages explicit. When you add steps (e.g. validation, metrics
  export), expose them as dedicated methods so they can be unit-tested and
  chained from `trigger_encoder_training_pipeline`.
- Wrap external calls with precise error handling. Reuse the logging pattern in
  the current implementation—log before launching, record the final artifact
  path, and re-raise unexpected exceptions so the caller can respond.
- Configuration defaults should continue to point at
  `configs/training/mntp_mistral_config.json`. Accept overrides via constructor
  arguments for reproducibility in tests and automation.

## Testing
- Update `tests/test_evolution_engine.py` when you alter pipeline behaviour.
  The test currently asserts that the returned path exists and that trainer
  methods are invoked; expand it to cover new side effects or error modes.
- Use `pytest` fixtures to replace `MNTPTrainer` with a fake implementation when
  validating orchestration logic. Keep temporary directories isolated under
  `tmp_path`.
