# Dataset Curation Guide

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This guide explains how to triage and curate the datasets emitted by the monGARS Deep Scanner.

## 1. Review the QA report

1. Open `report.md` and inspect the dataset totals, top source files, and warnings.
2. Confirm that the Québécois French detection ratios align with expectations for user-facing content.
3. Skim the sample payloads provided for each dataset to spot formatting issues or misclassified
   records quickly.

## 2. Validate provenance

1. Load `provenance.csv` into your spreadsheet tool of choice.
2. Filter by `dataset` and review a random sample of entries per dataset.
3. Cross-check the `source_file` and line ranges inside the repository to verify correctness.
4. Flag any outliers (e.g., line ranges that do not contain text, or unexpected file types) for
   extractor tuning.

## 3. Promote high-quality SFT records

1. In `sft_dataset.jsonl`, prioritise instructions that clearly state a user intent and responses that
   demonstrate helpful, safe assistant behaviour.
2. Use the `_meta.type` tag to cluster similar examples (e.g., `doc_dialog` vs `python_prompt_dialog`).
3. For French-Canadian coverage, filter by `_meta.qc_fr_ca = true` to evaluate linguistic accuracy and
   cultural tone.
4. Export your curated subset to `data/sft_final.jsonl` (or your preferred location) and document any
   manual edits you apply.

## 4. Curate agent handoff flows

1. Sort `agent_handoff_dataset.jsonl` by `_meta.type` to group workflow steps, Docker actions, and other
   structured outputs.
2. Ensure each instruction has a corresponding output object (valid JSON) describing the expected handoff.
3. When necessary, augment the output with clarifying metadata (e.g., environment variables) while keeping
   the structure faithful to the source file.
4. Save the curated dataset to `data/agent_handoff_final.jsonl`.

## 5. Prepare the embeddings corpus

1. Deduplicate paragraphs by hashing the `text` field (the scanner already filters exact duplicates across
   datasets when hashes match).
2. Remove any residual boilerplate or legal notices that do not contribute to the downstream embedding task.
3. Export the cleaned paragraphs to `data/embeddings_final.jsonl`.

## 6. Track adjustments

Document all manual curation steps (filters applied, files ignored, examples rewritten) in a changelog so
future scans can incorporate the improvements.
