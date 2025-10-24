# Advanced Fine-Tuning & Distributed Inference Plan

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This roadmap details the steps required to take the existing MNTP trainer,
curated-dataset pipeline, and Ray Serve deployment from "operational" to
"production-hardened" status. It is cross-linked from the
[docs/index.md](index.md) hub under both Developer Essentials and Research & ML
Tooling so owners can keep the backlog dynamic as implementation details land.

## Current capabilities snapshot
| Area | Source of truth | Status |
| --- | --- | --- |
| Adapter training orchestration | `modules/neurons/training/mntp_trainer.py` | ✅ Deterministic adapters on CPU-only hardware, optional LoRA/QLoRA when dependencies are installed. |
| Self-training triggers | `monGARS/core/self_training_engine.py` | ✅ Schedules batches, persists anonymised datasets, and records metrics in MLflow. |
| Ray Serve integration | `monGARS/core/llm_integration.py` | ✅ Endpoint rotation, retries, OpenTelemetry counters/histograms for `llm.ray.*`. |
| Adapter manifest publishing | `modules/neurons/registry.py` | ✅ Emits manifests under `models/encoders/` and notifies Ray Serve refresh hooks. |
| Governance & approvals | `monGARS/core/operator_approvals.py` | ✅ Reinforcement and adapter rollouts require explicit operator sign-off. |

## Current Capabilities
- `LLMIntegration` streams responses from local Ollama models and performs real
  Ray Serve round-trips with endpoint rotation, scaling-aware retries, adapter
  manifest tracking, and OpenTelemetry counters/histograms for `llm.ray.*`
  metrics (`monGARS/core/llm_integration.py`).
- `SelfTrainingEngine` batches curated conversation records, persists anonymised
  datasets, and launches `modules.neurons.training.mntp_trainer.MNTPTrainer` to
  train either deterministic linear adapters or LoRA/QLoRA weights depending on
  dependency availability.
- The evolution engine writes adapter manifests under
  `models/encoders/`, emits MLflow metrics, and triggers Ray Serve refresh hooks
  when new artefacts land.

## Telemetry anchors to monitor
| Signal | Source of truth | Purpose |
| --- | --- | --- |
| `llm.ray.*` counters | `monGARS/core/llm_integration.py` | Confirms retries, fallbacks, and latency histograms stay healthy across releases. |
| Adapter manifest checksum | `modules/neurons/registry.py` | Ensures rollout notifications reflect the artefact actually deployed to Ray Serve. |
| MNTP training summaries | `modules/neurons/training/mntp_trainer.py` + MLflow | Tracks dataset provenance, loss curves, and evaluation hooks for each run. |
| Operator approval events | `monGARS/core/operator_approvals.py` | Validates that safety checks block low-confidence adapters before rollout. |
| Dataset sanitiser audit log | `scripts/dataset_sanitiser.py` (planned) | Captures redaction results and export history for compliance reviews. |

## Strategic Goals
1. **Dataset hygiene**
   - Expand the dataset catalogue retained under `models/datasets/curated/` with
     provenance metadata (source peer, confidence, anonymisation timestamp).
   - Provide redaction/scrubbing utilities that operators can run before
     exporting datasets for offline experimentation.
2. **Training loop hardening**
   - Add evaluation harnesses (perplexity, regression tasks) that run after
     MNTP training and surface metrics in MLflow/OpenTelemetry.
   - Support resumable training by persisting optimiser state and learning-rate
     scheduler checkpoints.
   - Parallelise curated linear adapter training to speed up deterministic
     fallbacks on CPU-only hardware.
3. **Distributed inference operations**
   - Feed the emitted Ray Serve metrics (`llm.ray.requests`, `llm.ray.failures`,
     etc.) into dashboards and alerting alongside scheduler telemetry.
   - Publish a hardened Helm chart referencing `modules/ray_service.py` so teams
     can deploy inference clusters alongside monGARS core services.
   - Automate adapter rollouts by wiring SelfTrainingEngine summaries to the Ray
     Serve deployment refresh endpoint.
4. **Operational guardrails**
   - Implement policy checks that require human approval when confidence metrics
     drop below configurable thresholds during training.
   - Provide rollback tooling that reinstates previous adapter manifests and
     notifies Ray replicas.
   - Update `docs/implementation_status.md`, [docs/index.md](index.md), and the main README whenever new
     training or deployment capabilities land so the roadmap remains accurate.

## Immediate Next Steps
- Extend `tests/test_llm_ray.py` with scenarios that assert OpenTelemetry
  counters after Ray retries/fallbacks.
- Ship a dataset sanitiser CLI under `scripts/` that mirrors
  `models.datasets.sanitize_record` and emits audit logs.
- Document expected hardware profiles for LoRA fine-tuning and provide guidance
  for GPU memory pinning in `modules/neurons/training/mntp_trainer.py`.

## Update cadence
- Mirror every change into [docs/index.md](index.md) and the
  [Documentation Maintenance Checklist](documentation_maintenance.md) so release
  managers see which backlog items shifted.
- Review this plan at the end of each research sprint. If a goal ships, move it
  into the capability snapshot with links to the associated tests, modules, and
  telemetry dashboards.
- When updating roadmap priorities, cross-reference
  `docs/implementation_status.md` and mention the delta in the PR summary to keep
  reviewers aware of downstream documentation that needs edits.

When you ship one of these backlog items, update the capability snapshot table
above and cross-link the relevant tests or modules so the plan stays current.
