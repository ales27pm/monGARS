# monGARS Unsloth Fine-Tuning Dataset

This directory contains a curated JSONL dataset optimised for adapting `dphn/Dolphin-X1-8B` to monGARS-specific communication patterns. Each record pairs a prompt describing a system situation with a completion that encodes the canonical response structure monGARS should follow.

## File Overview

- `monGARS_unsloth_dataset.jsonl` – 12 prompt/completion pairs covering:
  - FastAPI chat entrypoint hand-offs, sanitisation, and telemetry publishing.
  - Multimodal enrichment steps inside `ConversationalModule.generate_response` including captioning, curiosity detection, reasoning, and semantic recall.
  - Post-LLM adaptation via personality, mimicry, speech synthesis, persistence, hippocampus memory storage, and evolution-engine sampling.
  - Operational messaging for prompt budgeting, event bus failures, self-training requests, peer telemetry isolation, and evolution diagnostics.

## Usage

Run the Unsloth pipeline with the local dataset by pointing `--dataset-path` at this file:

```bash
python dolphin_x1_unsloth_pipeline.py \
  --dataset-path datasets/unsloth/monGARS_unsloth_dataset.jsonl \
  --output-dir outputs/dolphin-x1-mongars \
  --epochs 3 \
  --train-fraction 1.0
```

The script automatically tokenises `prompt`/`completion` pairs, preserving the structured protocol markers (`[[monGARS-protocol]]`) so the fine-tuned model learns to emit the expected sections during inference.

## Validation

Loaders in `monGARS.mlops.dataset.prepare_local_instruction_dataset` assert the presence of `prompt` and `completion` keys and convert them into masked token sequences that respect the Dolphin chat template, ensuring this dataset is immediately consumable by the existing training pipeline.
