# Model Configuration & Provisioning

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
          "name": "dolphin3",
          "provider": "ollama",
          "auto_download": true,
          "parameters": {
            "top_p": 0.9,
            "num_predict": 512
          }
        },
        "coding": "dolphin3-llm2vec"
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

## Runtime Behaviour

- `LLMIntegration` calls `LLMModelManager.ensure_models_installed()` before
  dispatching to Ollama, ensuring the active profile is available even after
  container rebuilds.
- When Ray Serve is enabled, adapter manifests stored under
  `settings.llm_adapter_registry_path` keep API workers and Ray replicas aligned
  with the latest training artefacts.
- Provisioning reports are cached per-role during a process lifetime to avoid
  redundant downloads when multiple requests arrive concurrently.

### Inference Backend & Fallback Path

- Model definitions are sourced from `configs/llm_models.json`. The default
  profile declares two roles—`general` and `coding`—that both target the
  Dolphin 3 family via the Ollama provider. Each entry includes provider
  options (`temperature`, `top_p`, `num_predict`, etc.) that are merged with the
  runtime settings pulled from `monGARS.config.get_settings()`.
- `LLMModelManager` currently only instantiates Ollama-backed definitions. Any
  manifest entry referencing an unsupported provider is skipped, and a warning
  is logged so operators can reconcile the mismatch before traffic is routed to
  that role.
- During request handling, `LLMIntegration` invokes `ollama.chat` through
  `_ollama_call`. User prompts are sent as chat messages, and Ollama receives
  the consolidated generation options from the model definition and global
  settings. Responses stream back as text, which is surfaced to the caller and
  emitted on the UI event bus.
- If Ollama is unavailable—because the socket connection fails or the client is
  missing—the integration falls back to a local PyTorch slot via
  `ModelSlotManager`. The slot loads the configured HuggingFace-compatible
  checkpoint, applies the same temperature and nucleus sampling parameters, and
  returns decoded text. Structured logs capture the transition so on-call teams
  understand when the system is running on the degraded path.

### Unsloth Optimisations & VRAM Troubleshooting

- Local fallback slots rely on Unsloth's `FastLanguageModel` to provide 4-bit
  loading, quantisation-aware LoRA adapters, and gradient checkpointing. These
  hooks are applied when `ModelSlotManager` initialises a slot; if Unsloth is
  missing, slot acquisition fails early with an actionable error.
- `LLMIntegration.initialize_unsloth()` attempts to patch PyTorch at startup so
  fused kernels and quantisation paths are ready before any slot is acquired. A
  structured log (`llm.unsloth.patched`) confirms whether the patch succeeded.
  The helper reports a 70% VRAM reduction baseline for the reference Dolphin 3
  adapter.
- Use `python -m scripts.diagnose_unsloth` to inspect whether the optimisation
  was applied and to capture current CUDA memory headroom. The command now emits
  an extended JSON payload that records Python, PyTorch, and Unsloth versions
  alongside per-device VRAM usage (free, reserved, and allocated bytes plus
  utilisation ratios). Add `--all-devices` to iterate over every visible GPU,
  `--force` to re-apply the patch, or `--no-cuda` to skip GPU inspection when
  debugging on shared hosts. Supply `--min-free-gib` and `--min-free-ratio`
  thresholds to tune when OOM risk should be flagged; the CLI now summarises the
  highest observed risk level and surfaces remediation steps (context length
  tuning, offload thresholds, gradient accumulation, allocator defragmentation)
  for each device.
- Review the CLI output when the patch fails: the `environment.unsloth` section
  surfaces the installed package location and version, while
  `environment.torch` confirms whether CUDA support is compiled in. Missing or
  mismatched wheels often explain persistent OOM conditions even after the
  patch reports success.
- If you continue to hit OOM conditions even with Unsloth enabled, reduce the
  `max_seq_length` configured for each slot (defaults to 2048 tokens) or lower
  the `offload_threshold` so snapshots are taken earlier. Both parameters are
  accepted by `ModelSlotManager` and can be tuned where slots are instantiated
  (for example in worker startup code) to keep VRAM usage under the budget of
  8–12 GB consumer GPUs.
- When fine-tuning LoRA adapters, prefer gradient accumulation over larger
  per-device batches so that Unsloth's quantisation savings are not offset by
  training-time activations. The diagnostics script highlights `allocated` vs
  `reserved` VRAM; a high reserved fraction with low allocation indicates that
  fragmentation or peer memory pools are the bottleneck rather than model
  weights alone.

## Extending Profiles

1. Add a new profile or role in `configs/llm_models.json`. Use either the
   shorthand string notation or the detailed object schema shown above.
2. Update `LLM_MODELS_PROFILE` (or `llm_models_profile` in settings) to point to
   the desired profile.
3. Trigger provisioning via the CLI or API to download any newly referenced
   models.
4. Document custom providers or parameters alongside the profile definition so
   future maintainers understand dependency requirements.
