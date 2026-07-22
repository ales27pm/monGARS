# monGARS Deep Scanner

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The monGARS Deep Scanner is a local-only CLI that inspects a repository (or a zipped copy),
extracts user-facing artefacts, and assembles three datasets alongside provenance and QA reports.

## Quick start

```bash
make install
make run INPUT=. OUT=output
```

The command writes JSONL datasets, a provenance CSV, a QA report, and logs to the specified
`OUT` directory. Use `make dryrun` to preview the number of files that will be scanned without
producing artefacts.

## CLI usage

```bash
python -m tools.monGARS_deep_scan.deep_scan --input <path> [--out output/]
                                         [--allow-network]
                                         [--max-lines N] [--jobs N]
                                         [--dry-run]
                                         [--qc-terms qc_terms.txt]
                                         [--include-ext EXT,...]
                                         [--exclude-dir DIR,...]
```

- `--input` accepts either a directory or a local `.zip` archive.
- `--out` selects the output directory (default: `output/`).
- `--allow-network` toggles optional network access; it is disabled by default.
- `--max-lines` skips files whose line count exceeds the threshold (default: 50k).
- `--jobs` defines the worker pool size. By default the scanner uses up to `2 * CPU` cores (capped at 8).
- `--dry-run` validates configuration and lists the number of files that would be processed.
- `--qc-terms` supplies a newline-separated list of Québécois French keywords.
- `--include-ext` overrides the default extension allow-list.
- `--exclude-dir` augments the directory skip list.

Swift sources are scanned by default. Documentation comments and prompt
templates feed the embedding/SFT corpora, while canonical `ToolDefinition`
declarations produce structured agent records with file and line provenance.

## Outputs

Running the scanner generates:

- `sft_dataset.jsonl`
- `agent_handoff_dataset.jsonl`
- `embeddings_corpus.jsonl`
- `provenance.csv`
- `report.md`
- `logs/scan.log`

Each dataset record contains `_meta` provenance fields, including the source file, line range,
type label, and Québécois French detection flag.

## Development workflow

1. Install dependencies with `make install`.
2. Run unit tests via `make test` (or `pytest -q tests/test_extractors_unit.py tests/test_end_to_end.py`).
3. Execute the CLI with `make run` or `python -m tools.monGARS_deep_scan.deep_scan ...`.

All modules live under `tools/monGARS_deep_scan/`, and accompanying tests reside in `tests/`.

## Native agent manifest and improvement bundle

`agent_manifest_pipeline` derives monGARS's agent contract directly from the
current Swift `AgentToolCatalog`, `AgentToolValidation`, `AgentIntentRouter`,
and selected `configs/llm_models.json` profile. It fails closed unless the current native
contract contains 53 unique tools, 26 approval-protected tools, and 22 complete
intent routes. The generator is local-only and uses fixed timestamps,
content-derived IDs, canonical JSON, and hash-based validation splits.
The complete native validator is digest-pinned, including normalization,
common schema checks, semantic helpers, and cross-field rules: changing its
Swift implementation requires an explicit review of the mirrored dataset
contract.

```bash
python -m tools.monGARS_deep_scan.agent_manifest_pipeline \
  --root . \
  --out /tmp/mongars-agent-bundle
```

Pass `--runtime-audit report.json` more than once, or point it at a directory,
to turn failed runtime scenarios into bounded REM repair samples. JSON inputs
must expose explicit `failures`, `violations`, `repairSamples`, `results`,
`scenarios`, or `tests` records. Text inputs accept Lumen-style
`❌ Training eval:` blocks or one strict `FAIL|ERROR|PASS <scenario>: <detail>`
record per line. Malformed, outcome-free, oversized, or unsupported evidence
aborts the whole build; it is never silently skipped. Lumen evidence-layer
ownership is preserved, static checks cannot masquerade as live runtime proof,
and bounded repair text is scrubbed recursively for common secrets and personal
data. Raw/legacy JSON and text summaries can still generate repairs, but only a
machine-readable `e2eTestReport` envelope that explicitly owns live scenarios
can close the improvement loop. Mixed envelopes retain only live-marked rows,
and each retained row must carry a consistent outcome plus an accepted Lumen
model-evidence event with concrete runtime semantics.

The output contains:

- `AgentBehaviorManifest.json`, exact tool schema cards, routing/grounding
  cards, eval scenarios, and runtime repair samples;
- deterministic SFT and DPO train/validation lanes for Cortex, Executor,
  Mouth/Bouche, Mimicry, and REM;
- `dataset_manifest.json`, `provenance.csv`, `artifact_hashes.json`, and a
  self-hashed artifact index; and
- one `improvement_loop/gap_report.json` linking audit evidence to repair IDs
  and the next bounded action.

Outputs are reproducible build products, not source files. Generate them in a
scratch or ignored directory and do not commit them. An existing output is
preserved unless `--replace` is explicitly supplied; replacement is staged and
published atomically.
