# RAG Dataset Governance

The curated datasets produced by `SelfTrainingEngine` now include structured
governance metadata and automated scrubbing checks so that partner exports stay
compliant. This document captures the retention policy, the validation pipeline,
and the day-to-day operator procedures.

## Metadata Contract

Every dataset version registered in `models/datasets/curated/` now records the
following fields:

| Field | Description |
| --- | --- |
| `provenance` | Identifies which pipeline produced the dataset (`self-training` by default). |
| `sensitivity` | Data classification tag surfaced to downstream consumers. |
| `retention_days` | Maximum age before the dataset must be re-reviewed. |
| `reviewed_by` / `reviewed_at` | Reviewer identity (defaults to automation) and timestamp. |
| `export_window_days` / `export_window_ends_at` | Window during which exports are allowed before a re-review is required. |
| `tags` | Extra governance tags applied to the dataset entry. |

The metadata is built in `models/datasets/governance.py` and stored as part of
the catalog entry returned by `DatasetCatalog.register`.【F:models/datasets/governance.py†L41-L132】【F:models/datasets/catalog.py†L17-L95】

Configuration comes from new settings in `monGARS.config.Settings`, allowing
operators to tune provenance labels, retention windows, and review identities
without code changes.【F:monGARS/config.py†L198-L233】

## Automated Scrubbing Checks

When a curated batch is persisted, the governance layer re-scans every record
before it is registered in the catalog. The checks currently include:

1. **PII Detection** – reuses the sanitiser patterns to spot e-mail, phone,
   payment, IP, and UUID fragments that should have been redacted. Any matches
   mark the dataset as quarantined.【F:models/datasets/sanitizer.py†L11-L92】【F:models/datasets/governance.py†L134-L196】
2. **Record Count Validation** – verifies the number of JSONL entries matches
   what the pipeline recorded during batching.【F:models/datasets/governance.py†L171-L196】
3. **Retention & Export Windows** – flags datasets whose expiry or export window
   has elapsed so they can be re-reviewed before partner distribution.【F:models/datasets/governance.py†L96-L133】
4. **Metadata Completeness** – fails fast if required reviewer information is
   missing or malformed.【F:models/datasets/governance.py†L86-L109】

Compliance results are persisted alongside the metadata and exposed through the
training summaries so operators have visibility without digging into catalog
files.【F:monGARS/core/self_training.py†L233-L272】

## Operator Playbook

### Onboarding a New Repository

1. Configure provenance and tags for the partner via environment variables or
   `.env` overrides (`RAG_CURATED_DEFAULT_PROVENANCE`,
   `RAG_CURATED_DEFAULT_TAGS`).【F:monGARS/config.py†L198-L233】
2. Trigger a self-training cycle or run `SelfTrainingEngine._run_training_cycle`
   in a controlled environment to produce a curated batch.【F:monGARS/core/self_training.py†L65-L178】
3. Confirm the resulting dataset shows `"status": "approved"` in its compliance
   payload before enabling export to the partner workspace.【F:tests/self_training_test.py†L1-L200】

### Handling Takedown or Expiry Requests

1. Locate the dataset run ID in `models/datasets/curated/catalog.json` and
   inspect the stored metadata to confirm expiry or export window details.【F:models/datasets/catalog.py†L39-L95】
2. If the dataset is expired or quarantined, remove the directory and notify the
   partner; the next training cycle will regenerate compliant batches.
3. Update retention settings if contractual terms change, then rerun the next
   training cycle so the new window is applied.

### Export Procedure

1. Prior to sharing, rerun the governance evaluation with
   `DatasetGovernance.evaluate_dataset` to confirm the compliance status is
   still `approved`.
2. If violations are reported, address them (by re-sanitising data or updating
   metadata) before distributing any artefacts. Export is blocked while the
   dataset remains quarantined.

## Observability

Governance results are logged with structured metadata
(`curated.dataset.governance_passed` / `.governance_failed`), enabling alerting
or dashboards to track quarantine events over time.【F:models/datasets/governance.py†L198-L219】
