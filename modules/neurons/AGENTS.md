# Neuron Utilities Standards

These rules apply to `modules/neurons/` and its training helpers.

## Runtime Behaviour
- Import heavy ML frameworks lazily. Follow the `training/mntp_trainer.py`
  pattern where missing dependencies result in informative fallbacks rather than
  crashes.
- Accept output directories, manifests, and config paths as explicit arguments so
  callers can orchestrate training runs from tests or automation scripts.
- Persist artefacts via helper methods (e.g. `_save_config`, `_save_placeholder`)
  and document file layouts in docstrings.

## Logging & Metrics
- Stream progress logs instead of returning large payloads. Include model name,
  dataset, loss values, and output paths when available.
- Emit metrics to OpenTelemetry when you introduce long-running jobs so the
  evolution engine can track performance.

## Testing
- Update `tests/test_mntp_trainer.py` or create new async tests for trainers.
- Use `pytest.mark.parametrize` for config permutations and keep generated files
  confined to the provided temporary directory (`tmp_path`).
