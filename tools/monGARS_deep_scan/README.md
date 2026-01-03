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
