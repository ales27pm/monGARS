# French Dataset Pipeline

> **Last updated:** 2025-11-23

The [`scripts/consolidated_french_dataset_pipeline.py`](../../scripts/consolidated_french_dataset_pipeline.py) entry point provides a production-ready ingestion pipeline for French instruction and retrieval datasets. It enforces French-only processing, layers quality and license checks, and emits validated exports plus run metadata for governance.

## Capabilities
- Loads multiple Hugging Face datasets with checkpointing, deduplication, and sampling safeguards tailored for French corpora, then orchestrates validation and export steps via a single `DatasetPipeline.run()` flow.
- Applies license validation, PII detection, and configurable quality thresholds before persisting merged splits and derived prompt/completion exports for Unsloth-style fine-tuning.
- Supports crash recovery with periodic state checkpoints, resumable deduplication, and optional cloud uploads when `boto3` is available.

## Prerequisites
- Python 3.11 with the new dependencies added to `requirements.txt`/`setup.py`: `datasets`, `datasketch`, `numpy`, `tqdm`, `boto3` for optional cloud delivery, and `flask` for the optional dashboard server.
- Access to Hugging Face datasets storage (local cache or configured download directory) and sufficient disk space for temporary checkpoints.

To enable the optional Flask dashboard, install dependencies (e.g., `pip install -r requirements.txt`), provide a `--dashboard_port` value, and ensure the port is reachable from your operator workstation. The dashboard renders real-time instruction/retrieval counts, deduplication statistics, memory usage, and per-source breakdowns.

## Running the pipeline
1. Create an output directory and invoke the script with the desired quotas and toggles, for example:
   ```bash
   python scripts/consolidated_french_dataset_pipeline.py \
     --output_dir /data/pipelines/french-merge \
     --dashboard_port 8080 \
     --max_per_dataset 75000 \
     --enable_checkpointing \
     --enable_validation \
     --unsloth_export_name unsloth_prompt_completion.jsonl
   ```
2. Monitor progress via the rotating log at `dataset_pipeline.log` inside the output directory; console output also surfaces progress bar updates and configuration banners.
3. Resuming a run uses `--resume` together with the checkpoint and dedup state files stored alongside the outputs; enable `--force_download` when underlying datasets change and cached shards must be refreshed.

## Outputs
- Final train/validation artifacts plus prompt/completion exports in the output directory.
- Metadata JSON summarising dataset provenance, applied filters, and any PII/license findings to simplify downstream auditing.
- Optional uploads to a configured object store when `--cloud_storage` includes an AWS target and `boto3` is installed.
