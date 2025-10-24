# monGARS Unsloth Pipeline Outputs

> **Last updated:** 2025-10-14 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This directory hosts the LoRA adapters, merged checkpoints, and wrapper module
produced by the `unsloth-trainer` Docker Compose profile. The runtime services
mount this path so newly trained artefacts become available without rebuilding
images.

## Layout
- `run/` – default output directory passed to the training pipeline. Contains
  `chat_lora/` adapters, metadata, and optional `merged_fp16/` checkpoints.
- `adapter_manifest.json` – automatically refreshed via
  `scripts/run_mongars_llm_pipeline.py finetune --registry-path …` so inference
  components can detect the latest revision.
- `latest` – symlink pointing at the most recent adapter directory when the
  manifest is updated.

## Usage
1. Trigger a training run:
   ```bash
   docker compose run --rm --profile training unsloth-trainer
   ```
2. Inspect `run/run_metadata.json` for dataset provenance, metrics, and
   evaluation results.
3. Deploy the refreshed adapters by restarting the API or Ray Serve services if
   they cache model state in memory.

> **Note:** Heavy checkpoints (merged FP16, GGUF, AWQ) remain ignored by Git via
> the local `.gitignore`. Upload distributable builds to object storage rather
> than committing them to the repository.
