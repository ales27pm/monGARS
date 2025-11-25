# monGARS multi-stage pipeline runbook

> **Last updated:** 2025-11-25

This runbook explains how to operate `mongars_multistage_pipeline.py`, the
end-to-end CLI that assembles dataset generation, supervised fine-tuning (SFT),
LLM2Vec adaptation, an optional retrieval-augmented fine-tuning (RAFT) hook,
and export. Use it to reproduce the training/export flow described in the
repository without stitching together individual scripts.

## Prerequisites
- Python 3.11 with the repository installed (`pip install -e .` or inside the
  repo root) so imports under `monGARS.mlops` resolve.
- Access to the LLM2Vec repository with `experiments/run_mntp.py` and
  `experiments/run_simcse.py` available on disk (only when running the
  `--llm2vec` stage).
- A Hugging Face model identifier for the base model (e.g.
  `mistralai/Mistral-7B-v0.2`) and an optional HF token for private weights
  (only when running `--sft` and/or `--export`).
- Sufficient GPU VRAM for Unsloth/QLoRA training (defaults assume ~24GB). If
  dependencies are missing locally, run `scripts/install_test_dependencies.sh`
  to install the runtime/test stack before launching the pipeline.

## Quick start
Run from the repository root so the relative paths resolve. Include only the
flags for the stages you plan to execute:

```bash
python mongars_multistage_pipeline.py \
  --run-id myexperiment \
  --repo-root . \
  --build-datasets --sft --llm2vec --export \
  --base-model mistralai/Mistral-7B-v0.2 \
  --mntp-config path/to/mntp_config.json \
  --simcse-config path/to/simcse_config.json \
  --llm2vec-root /path/to/llm2vec
```

### Stage-specific runs
- Generate datasets only: `python mongars_multistage_pipeline.py --run-id demo --build-datasets`
- Resume SFT after datasets exist: `python mongars_multistage_pipeline.py --run-id demo --sft --base-model <id>`
- Apply LLM2Vec to existing adapters: `python mongars_multistage_pipeline.py --run-id demo --llm2vec --mntp-config <file> --simcse-config <file> --llm2vec-root <path>`
- Export current adapters to GGUF: `python mongars_multistage_pipeline.py --run-id demo --export --base-model <id>`
- Run export only (relying on previous state): `python mongars_multistage_pipeline.py --run-id demo --export --base-model <id>`

## How the stages behave
- **Dataset generation**
  - Invokes `scripts/consolidated_french_dataset_pipeline.py` for instruction,
    reasoning, conversation, and retrieval datasets. Outputs land under
    `runs/<run-id>/french_datasets`.
  - Runs internal static analysis (LLM usage + module interactions) to build
    per-module instruction datasets under `runs/<run-id>/internal_datasets`,
    split into `train.jsonl` and `val.jsonl` per module.
- **SFT**
  - Calls `monGARS.mlops.pipelines.unsloth.run_unsloth_finetune` per module,
    writing LoRA adapters and wrapper bundles in `runs/<run-id>/<module>/`.
  - Hyperparameters (LoRA rank/alpha/dropout, epochs, batch size, VRAM budget)
    come from CLI flags so runs stay reproducible.
- **LLM2Vec**
  - Executes MNTP then SimCSE using the provided configs and LLM2Vec repo. Each
    module receives `mntp/` and `simcse/` subdirectories with the adapted
    checkpoints; `current_model_dir` moves to the SimCSE output for downstream
    steps.
- **RAFT (optional)**
  - Stubbed by default; call with `--raft` only after you add retrieval-aware
    fine-tuning logic to `perform_raft` inside
    `mongars_multistage_pipeline.py`.
- **Export**
  - Exports each module’s `current_model_dir` to GGUF under
    `runs/<run-id>/<module>/gguf/` via
    `monGARS.mlops.exporters.export_gguf` (or a custom shell command if
    `--export-cmd` is provided).

## State and resumability
- Every run writes `runs/<run-id>/state.json` that tracks dataset paths,
  per-module train/val counts, adapter output paths, and the most recent
  `current_model_dir` per module.
- Re-running with new stage flags reads the existing state to skip completed
  work. If `state.json` is missing or corrupted, regenerate datasets with
  `--build-datasets` before continuing. The CLI validates stage-specific
  requirements (e.g., `--base-model` for SFT/export and LLM2Vec paths for the
  LLM2Vec stage) before starting work.
- Outputs remain isolated per `--run-id`; use different run IDs to compare
  hyperparameters without clobbering artefacts.

## Troubleshooting
- **Missing modules/import errors**: ensure the virtual environment is active
  and run `scripts/install_test_dependencies.sh` if `pytest` previously failed
  on missing optional packages.
- **LLM2Vec scripts not found**: verify `--llm2vec-root` points to a checkout
  that contains `experiments/run_mntp.py` and `experiments/run_simcse.py`.
- **Hugging Face permission errors**: supply `--hf-token` when using private
  models or gated datasets.
- **GPU memory pressure**: reduce `--batch-size`, `--lora-r`, or `--lora-alpha`,
  or increase `--vram-budget-mb` to align with available hardware.

## Related references
- Pipeline implementation: [`mongars_multistage_pipeline.py`](../mongars_multistage_pipeline.py)
- French dataset flow: [`docs/french_dataset_pipeline.md`](french_dataset_pipeline.md)
- LLM2Vec configs: tracked per module under `runs/<run-id>/<module>/simcse/`
  and the source configs you pass via `--mntp-config`/`--simcse-config`
