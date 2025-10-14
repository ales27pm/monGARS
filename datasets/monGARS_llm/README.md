# monGARS LLM Alignment Dataset

This dataset captures high-signal instruction/response pairs tailored to monGARS so fine-tuning runs can reason about the platform's cognition pipeline, API surface, memory layout, and operational runbooks.

## Contents
- `monGARS_llm_dataset.jsonl` – Full JSON Lines corpus with `prompt`, `response`, and `metadata` objects. Records include a `split` key indicating whether they belong to train or validation subsets.
- `monGARS_llm_train.jsonl` / `monGARS_llm_val.jsonl` – Pre-split training and validation partitions ready for direct ingestion.
- `generation_report.json` – Summary statistics emitted by the generator script (timestamp, counts, split sizes, source breakdown).
- `scripts/generate_monGARS_llm_dataset.py` – Reproducible generator that rebuilds the dataset from source material and curated templates.

## Usage
1. Regenerate the dataset to capture the latest APIs and module graph:
   ```bash
   python scripts/generate_monGARS_llm_dataset.py
   ```
   Pass `--val-ratio` to adjust the validation split and `--seed` for deterministic shuffling when creating alternative partitions.
2. Consume either the combined file or the dedicated `train`/`val` artifacts with your preferred fine-tuning or RAG bootstrap tooling.
3. Track versioning by checking the `generation_report.json` timestamp, commit hash, and `split_counts` metadata.

## Docker Integration

The default Docker Compose deployment now mounts the dataset into `/app/datasets/monGARS_llm`
and publishes it through the curated catalog consumed by the self-training pipeline.

- `models/datasets/curated/catalog.json` registers run `monGARS_llm_20251014` and points to
  `datasets/monGARS_llm/monGARS_llm_dataset.jsonl` so `collect_curated_data()` immediately
  returns the monGARS-aligned prompts when containers boot.
- `docker-compose.yml` binds both the raw dataset directory (read-only) and the curated catalog
  path into the application services, ensuring fine-tuning or evaluation jobs observe any
  refreshed exports without rebuilding the image.

### Unsloth + LLM2Vec training service

Run the full fine-tuning and wrapper export pipeline inside Docker with the dataset split
provided here:

```bash
docker compose run --rm --profile training unsloth-trainer
```

The service executes `scripts/run_mongars_llm_pipeline.py finetune` against
`monGARS_llm_train.jsonl` (with `monGARS_llm_val.jsonl` for evaluation), merges adapters,
and refreshes the adapter manifest under `models/encoders/monGARS_unsloth`. Override any
hyperparameter by exporting environment variables before running, for example:

```bash
export UNSLOTH_BATCH_SIZE=2
export UNSLOTH_EPOCHS=3
docker compose run --rm --profile training unsloth-trainer
```

All API-facing services mount `./models/encoders` so the refreshed adapters and wrapper module
are immediately visible to the runtime once training completes.

To replace the dataset at runtime, regenerate the JSONL files and rerun `docker compose up` so
the mounted catalog and artifacts are refreshed inside the containers.

Each record includes both a `category` label (architecture, operations, module_reference, api, alignment, …) and a `split` designation so you can sub-sample for tasks like architecture Q&A, API automation, self-optimization, or reliability runbooks while maintaining evaluation parity.
