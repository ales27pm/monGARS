# Advanced Fine-Tuning & Distributed Inference Plan

This roadmap details the steps required to move from placeholder adapters to a
production-ready masked next-token training loop and distributed inference.

## Current Capabilities
- `LLMIntegration` streams responses from local Ollama models with retry logic
  and a stubbed Ray Serve branch.
- `SelfTrainingEngine` batches conversation data but simulates training cycles,
  incrementing version metadata without touching weights.
- The evolution engine writes placeholder artefacts to `models/encoders/` and
  emits manifest updates that Ray Serve could consume in the future.

## Strategic Goals
1. **Dataset Hygiene**
   - Aggregate conversation data from `PersistenceRepository` and export
     anonymised corpora.
   - Maintain a versioned dataset catalogue (`models/datasets/`) with reproducible
     preprocessing scripts (PII stripping, deduplication, quality filters).
2. **Training Loop Completion**
   - Implement masked next-token prediction in `modules/neurons/training/mntp_trainer.py`
     with support for LoRA/QLoRA adapters.
   - Stream metrics to MLflow and OpenTelemetry for visibility into convergence.
   - Persist adapter weights, tokenizer assets, and metadata alongside manifest
     updates.
3. **Distributed Inference Activation**
   - Replace the Ray Serve stub in `LLMIntegration` with real HTTP requests,
     health checks, and exponential backoff tuned for replica autoscaling.
   - Ship Helm charts or K8s manifests under `k8s/` for Ray clusters, including
     secrets management and resource limits.
   - Keep Ray Serve replicas synchronised with adapter manifests published by the
     evolution engine.
4. **Operational Guardrails**
   - Enforce dataset lineage logging to trace which conversations generated which
     adapters.
   - Provide manual rollback tooling that reverts to the last known-good adapter.
   - Update `docs/implementation_status.md` and the main `README.md` whenever new
     training features land.

## Immediate Next Steps
- Wire the MNTP trainer into `SelfTrainingEngine` so scheduled runs execute real
  training instead of placeholders.
- Add regression tests covering Ray Serve round-trips using a lightweight mock
  server.
- Publish preprocessing scripts and document retention policies for captured
  conversation data.
