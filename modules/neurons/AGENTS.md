# Neuron Utilities Guidelines

The `neurons` package houses embedding and training helpers used by the
conversational stack and the evolution engine. `core.py` exposes lightweight
runtime helpers, while `training/` hosts long-running jobs such as the MNTP
trainer.

- Keep imports optional. Follow the pattern in `training/mntp_trainer.py` where
  `torch`, `datasets`, and `llm2vec` are wrapped in a `try/except` block and
  stored as module-level fallbacks. Never import heavy frameworks at module load
  time without guards.
- Trainers must accept output directories and config paths. Persist artefacts
  using the helper methods already present (`_save_config`, `_save_placeholder`).
  Any new file layout should be documented in the docstring and tests.
- When handling real ML runs, prefer streaming logs instead of large return
  payloads. Keep metrics dictionaries small and serialisable.
- Add new configuration schemas under `configs/training/` and validate them in
  `_load_config` or equivalent entry points.

## Testing
- Extend `tests/test_mntp_trainer.py` for new trainer behaviours. Patch missing
  dependencies and assert that fallbacks (like placeholder adapters) are
  emitted when requirements are unavailable.
- Use `pytest.mark.parametrize` for different config permutations so we catch
  validation errors early.
- Keep generated artefacts under the provided temporary directory (`tmp_path`)
  to avoid polluting the repository.
