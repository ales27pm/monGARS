# Model Configuration & Provisioning

> **Last updated:** 2025-11-25 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The LLM runtime is configured through a manifest located at
`configs/llm_models.json`. Profiles group related model roles (for example
`general` and `coding`) and are loaded via `LLMModelManager`. Each entry can
define provider-specific parameters, automatic download preferences, and an
optional description.

```json
{
  "profiles": {
    "default": {
      "models": {
        "general": {
          "name": "dolphin-x1",
          "provider": "ollama",
          "auto_download": true,
          "parameters": {
            "top_p": 0.9,
            "num_predict": 512
          }
        },
        "coding": "dolphin-x1-llm2vec"
      }
    }
  }
}
```

## Operational Interfaces

### FastAPI Endpoints
- `GET /api/v1/models` (admin token required) returns the active profile
  including resolved overrides and a list of all available profiles discovered
  in the manifest.
- `POST /api/v1/models/provision` accepts an optional `roles` array and `force`
  flag, ensuring local providers (such as Ollama) have the required weights.
  Response payloads enumerate the provisioning action taken for each role.

Endpoints are registered automatically via `monGARS.api.model_management` and
surface in the generated OpenAPI specification (`docs/api/openapi.json`).

### Command-Line Utility
`python -m scripts.provision_models` wraps the same provisioning workflow for
local development, CI pipelines, or remote maintenance. Options include:

```bash
python -m scripts.provision_models --roles general coding --force --json
```

- `--roles` filters provisioning to specific roles (default: all roles defined
  by the active profile).
- `--force` bypasses caching so models are revalidated even if previously
  ensured.
- `--json` emits machine-readable output, useful for scripting.

Logs are emitted under the `scripts.models.provision` namespace with the roles
and actions taken, allowing operators to forward results into structured log
pipelines.

### Automated monGARS fine-tuning bundle
- `python -m scripts.build_monGARS_llm_bundle` orchestrates LoRA training for
  `dphn/Dolphin-X1-8B`, consuming either the curated
  `datasets/unsloth/monGARS_unsloth_dataset.jsonl` file or a HuggingFace dataset
  identifier, exporting adapters, and writing an LLM2Vec-ready wrapper alongside
  enriched run metadata so operators can redeploy the artefacts without
  additional scripting. The CLI defaults to the repository’s `models/encoders`
  registry but accepts `--registry-path` for alternate manifests and `--json` to
  emit a machine-readable run summary once training completes. Refer to
  [`scripts/build_monGARS_llm_bundle.py`](../scripts/build_monGARS_llm_bundle.py)
  for the authoritative implementation.【F:scripts/build_monGARS_llm_bundle.py†L103-L195】【F:scripts/build_monGARS_llm_bundle.py†L498-L648】
- Dataset hygiene is surfaced before any GPU work: the command validates the
  JSONL payload, reports duplicate ratios, prompt/completion statistics, and
  writes `dataset_summary.json` next to the adapters. Use `--dry-run` to emit the
  summary (optionally as JSON via `--json`) without launching a fine-tune, which
  is ideal for pre-flight reviews in CI.【F:scripts/build_monGARS_llm_bundle.py†L279-L590】
- Configuration profiles can be captured in JSON or YAML and loaded through
  `--config`; CLI flags still win when both are supplied. Run labels provided via
  `--run-name` are propagated into `run_metadata.json` and the adapter manifest,
  enabling downstream schedulers to tag promotion candidates or bundle specific
  experiments.【F:scripts/build_monGARS_llm_bundle.py†L205-L399】【F:scripts/build_monGARS_llm_bundle.py†L498-L590】
- Optional evaluation datasets (`--eval-dataset-id` or `--eval-dataset-path`)
  and batch sizing knobs roll straight into the Unsloth pipeline, and the
  manifest metrics now include both dataset statistics and evaluation scores so
  consumers such as Ray Serve can sanity-check provenance automatically when the
  registry refreshes.【F:scripts/build_monGARS_llm_bundle.py†L530-L632】

## Runtime Behaviour

- `LLMIntegration` calls `LLMModelManager.ensure_models_installed()` before
  resolving task-specific adapters and now relies on
  `monGARS.core.llm_integration.UnifiedLLMRuntime` for both chat and
  embeddings. The singleton loads the Dolphin-X1 checkpoint defined by
  `settings.unified_model_dir` once, constructs both the LLM2Vec encoder and
  `AutoModelForCausalLM` with matching tokenizers, and exposes synchronous
  `generate`/`embed` helpers backed by 4-bit NF4 quantisation whenever CUDA is
  available.
- When Ray Serve is enabled, adapter manifests stored under
  `settings.llm_adapter_registry_path` keep API workers and Ray replicas aligned
  with the latest training artefacts.
- Provisioning reports are cached per-role during a process lifetime to avoid
  redundant downloads when multiple requests arrive concurrently.

### Inference Backend & Fallback Path

- Model definitions are still sourced from `configs/llm_models.json`, but roles
  now primarily control sampling defaults (`temperature`, `top_p`,
  `max_new_tokens`) that are merged with the quantisation settings exposed under
  `settings.model.*`. Each task routes directly to
  `UnifiedLLMRuntime.generate(...)`, ensuring conversational traffic, RAG
  reranking, and embeddings share the same tokenizer and adapter weights.
- The runtime automatically applies 4-bit NF4 quantisation when CUDA is
  available and falls back to BF16/FP32 execution on CPU hosts. All loading and
  inference runs through `asyncio.to_thread(...)` so event loops remain
  responsive even when synchronous callers trigger local inference.
- Legacy Ollama/Unsloth slots have been removed. If the runtime fails to load
  or serve a request, `LLMRuntimeError` bubbles up to orchestrators and API
  handlers, which surface structured errors and fall back to deterministic
  responses instead of attempting to invoke a second provider.

### Embedding Strategy

- Embeddings reuse the same Dolphin-X1 checkpoint that powers chat. The
  `dolphin_llm2vec_pipeline.py` workflow saves LoRA adapters, exports an
  LLM2Vec-compatible wrapper, and annotates deterministic embedding options so
  downstream services can reload the weights without guessing parameters.
  Operators should consult the wrapper metadata before adjusting pooling or
  token limits to keep search vectors aligned with chat behaviour.【F:dolphin_llm2vec_pipeline.py†L40-L139】
- The training pipeline persists tokenizer metadata, preferred chat sampling
  defaults, and artifact layout (tokenizer directory, adapter subdirectory, and
  merged FP16 snapshot) in `wrapper_config.json`. The exporter deep-merges those
  details into `wrapper/config.json`, stamps a manifest version, and lets
  operators override the base model identifier when publishing to external
  registries.【F:dolphin_llm2vec_pipeline.py†L88-L139】【F:scripts/export_llm2vec_wrapper.py†L1-L229】
- The generated wrapper module exposes `generate` and `embed` helpers via
  Hugging Face Transformers, ensuring the same tokenizer and hidden-state layout
  drive both conversational and retrieval workloads. The embedding path uses
  mean pooling with deterministic attention-mask weighting and optional
  normalisation configured by the manifest.【F:scripts/export_llm2vec_wrapper.py†L57-L168】
- `UnifiedLLMRuntime.generate`/`embed` keep conversational responses and stored
  vectors aligned with the tokenizer, adapters, and pooling strategy described
  in `build_unified_dolphin_x1_enhanced.py`. Operators can still export the
  embedding service independently via `scripts/run_llm2vec_service.py`, which
  offers the same quantisation and device-pinning knobs for HTTP deployments.【F:scripts/run_llm2vec_service.py†L1-L204】

### Quantisation Controls & VRAM Troubleshooting

- Quantisation knobs have moved into `settings.model.*`. Operators can toggle
  `quantize_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_compute_dtype`, and
  `bnb_4bit_use_double_quant` to match the hardware profile. These settings are
  consumed directly by `UnifiedLLMRuntime` so both the generator and LLM2Vec
  encoder share the same precision strategy.
- Structured logs (`llm_runtime_load`, `llm_generate`, `llm_embed`) now include
  token counts and device metadata, making it easy to spot when requests fall
  back to CPU or when quantisation is disabled. Review these logs alongside the
  existing `scripts.diagnose_unsloth` output to correlate memory pressure with
  runtime behaviour.
- When memory pressure persists, reduce the generation target exposed via
  `settings.model.max_new_tokens` or lower the per-role overrides in
  `configs/llm_models.json`. The runtime enforces these bounds uniformly across
  orchestrators so aggressive defaults no longer leak through specific code
  paths.

## Extending Profiles

1. Add a new profile or role in `configs/llm_models.json`. Use either the
   shorthand string notation or the detailed object schema shown above.
2. Update `LLM_MODELS_PROFILE` (or `llm_models_profile` in settings) to point to
   the desired profile.
3. Trigger provisioning via the CLI or API to download any newly referenced
   models.
4. Document custom providers or parameters alongside the profile definition so
   future maintainers understand dependency requirements.
