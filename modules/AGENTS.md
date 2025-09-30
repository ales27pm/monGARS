# Optional Modules Standards

The `modules/` namespace hosts optional subsystems (e.g. evolution engine,
training utilities). They must remain importable on lightweight deployments.

## Design Principles
- Guard heavy dependencies (Torch, datasets, GPU tooling) behind runtime feature
  checks. Provide clear log messages and return placeholders when features are
  unavailable.
- Accept configuration via explicit parameters or config objects. Reuse JSON
  bundles from `configs/` instead of reading environment variables directly.
- Emit logs with `logging.getLogger(__name__)` and include identifiers such as
  model name, dataset, and output directory.
- Document new subpackages by adding scoped `AGENTS.md` files and updating
  `monGARS_structure.txt`.

## Testing
- Extend existing tests when behaviour changes:
  - `tests/test_evolution_engine.py` for orchestration
  - `tests/test_mntp_trainer.py` for neuron trainers
- Seed randomness and patch heavy imports to keep tests deterministic and fast.
