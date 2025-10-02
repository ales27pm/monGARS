# Advanced Fine-Tuning & Distributed Inference Plan

This roadmap details the steps required to take the existing MNTP trainer,
curated-dataset pipeline, and Ray Serve deployment from "operational" to
"production-hardened" status.

## Current Capabilities
- `LLMIntegration` streams responses from local Ollama models and performs real
  Ray Serve round-trips with endpoint rotation, scaling-aware retries, and
  adapter manifest tracking (`monGARS/core/llm_integration.py`).
- `SelfTrainingEngine` batches curated conversation records, persists anonymised
  datasets, and launches `modules.neurons.training.mntp_trainer.MNTPTrainer` to
  train either deterministic linear adapters or LoRA/QLoRA weights depending on
  dependency availability.
- The evolution engine writes adapter manifests under
  `models/encoders/`, emits MLflow metrics, and triggers Ray Serve refresh hooks
  when new artefacts land.

## Strategic Goals
1. **Dataset Hygiene**
   - Expand the dataset catalogue retained under `models/datasets/curated/` with
     provenance metadata (source peer, confidence, anonymisation timestamp).
   - Provide redaction/scrubbing utilities that operators can run before exporting
     datasets for offline experimentation.
2. **Training Loop Hardening**
   - Add evaluation harnesses (perplexity, regression tasks) that run after
     MNTP training and surface metrics in MLflow/OpenTelemetry.
   - Support resumable training by persisting optimiser state and learning-rate
     scheduler checkpoints.
   - Parallelise curated linear adapter training to speed up deterministic
     fallbacks on CPU-only hardware.
3. **Distributed Inference Operations**
   - Expose Ray Serve health counters (`llm.ray.*`) via OpenTelemetry and
     integrate them with the existing scheduler telemetry.
   - Publish a hardened Helm chart referencing `modules/ray_service.py` so teams
     can deploy inference clusters alongside monGARS core services.
   - Automate adapter rollouts by wiring SelfTrainingEngine summaries to the Ray
     Serve deployment refresh endpoint.
4. **Operational Guardrails**
   - Implement policy checks that require human approval when confidence metrics
     drop below configurable thresholds during training.
   - Provide rollback tooling that reinstates previous adapter manifests and
     notifies Ray replicas.
   - Update docs/implementation_status.md and the main README whenever new
     training or deployment capabilities land.

## Immediate Next Steps
- Extend `tests/test_llm_ray.py` with scenarios that assert OpenTelemetry counters
  after Ray retries/fallbacks.
- Ship a dataset sanitiser CLI under `scripts/` that mirrors
  `models.datasets.sanitize_record` and emits audit logs.
- Document expected hardware profiles for LoRA fine-tuning and provide guidance
  for GPU memory pinning in `modules/neurons/training/mntp_trainer.py`.
