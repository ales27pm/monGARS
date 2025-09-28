# Advanced Fine-Tuning and Distributed Inference Roadmap

This document outlines the planned steps for incorporating robust fine-tuning workflows, pretraining data curation, and distributed inference into **monGARS**.
For current implementation details, see [implementation_status.md](implementation_status.md).

## Current Architecture Overview

The project is structured into modular components such as `Hippocampus`, `Neurons`, `Bouche`, and `Cortex` as described in `monGARS_structure.txt`. The FastAPI web layer defines endpoints including conversation history retrieval in `monGARS/api/web_api.py`.

`LLMIntegration` provides model access with optional Ray Serve support (see `monGARS/core/llm_integration.py`). The `SelfTrainingEngine` batches data for later training though it currently simulates updates without modifying models (`monGARS/core/self_training.py`).

## Roadmap for Advanced Fine-Tuning

1. **Data Collection and Cleaning**
   - Aggregate conversation data stored by `PersistenceRepository` and user interactions in `Hippocampus`.
   - Implement preprocessing scripts to remove personally identifiable information and low-quality text.
   - Maintain a versioned dataset catalog under `models/datasets/`.
2. **Fine-Tuning Workflow**
   - Extend `MNTPTrainer` to perform real masked next token prediction training.
   - Integrate metrics logging via MLflow for each training run.
   - Store resulting adapters in `models/encoders/` and load them through `LLMIntegration`.
     Each training run now updates `models/encoders/adapter_manifest.json`, allowing
     the runtime to broadcast the newest adapter version to inference services.
3. **Distributed Inference**
   - Enable the Ray Serve path in `LLMIntegration` with actual requests to a cluster endpoint.
     The `modules/ray_service.py` module exposes a production-ready `deploy_ray_service`
     helper along with an `LLMServeDeployment` class that consumes the adapter manifest
     and keeps replicas synchronised with new weights.
   - Provide Helm charts under `k8s/` for scalable deployment.
   - Add monitoring hooks using OpenTelemetry metrics already present in `TieredCache`.

## Current Machine Learning Status

- **LLM Integration**: Models are served locally through Ollama with an optional
  Ray Serve backend. When Ray is enabled, `LLMIntegration` automatically attaches
  adapter metadata from `models/encoders/adapter_manifest.json` to every request,
  ensuring the cluster reloads freshly trained LLM2Vec adapters without manual
  coordination.
- **Self-Training Engine**: Present but performs simulated training cycles, recording new version numbers without altering model weights.

## Next Steps

1. Finalize the real training loop in `MNTPTrainer` and connect it to the `SelfTrainingEngine`.
2. Activate Ray Serve integration for large-scale inference and document deployment procedures.
3. Publish preprocessing guidelines and establish regular dataset refresh intervals.
4. Update `ROADMAP.md` as milestones are achieved.
