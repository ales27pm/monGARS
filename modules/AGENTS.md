# Modules Directory Guidelines

The `modules` namespace contains optional subsystems that extend the core
capabilities of monGARS. Today this includes the evolution engine controller and
research-grade neuron trainers. Everything here must remain importable even
when heavy ML dependencies are missing.

## Design Principles
- Guard optional packages (e.g. `torch`, `llm2vec`, `datasets`) with runtime
  availability checks as demonstrated in
  `modules/neurons/training/mntp_trainer.py`. Tests rely on these guards to run
  without GPU libraries.
- Pass configuration explicitly. Subpackages should accept file system paths or
  config objects instead of reading environment variables directly. Reuse the
  JSON configs in `configs/training/` when possible.
- Emit logs through `logging.getLogger(__name__)`. Include key identifiers such
  as model name, dataset, and output directory to aid long-running experiment
  debugging. Avoid `print` except for CLI entry points.
- Document new subpackages in `monGARS_structure.txt` and create a scoped
  `AGENTS.md` describing their internal conventions.

## Testing
- Extend `tests/test_evolution_engine.py` when changing orchestrator behaviour
  (e.g. new pipeline stages or artifact paths).
- Mirror the dependency stubbing strategy from `tests/test_mntp_trainer.py` for
  trainers and neuron utilities. Heavy imports should be patched with
  lightweight fakes.
- Keep tests deterministic by seeding random modules in fixtures when you
  introduce stochastic algorithms.
