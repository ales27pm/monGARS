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
          "name": "dolphin-mistral:7b-v2.8-q4_K_M",
          "provider": "ollama",
          "auto_download": true,
          "parameters": {
            "top_p": 0.9,
            "num_predict": 512
          }
        },
        "coding": "qwen2.5-coder:7b-instruct-q6_K"
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

## Extending Profiles

1. Add a new profile or role in `configs/llm_models.json`. Use either the
   shorthand string notation or the detailed object schema shown above.
2. Update `LLM_MODELS_PROFILE` (or `llm_models_profile` in settings) to point to
   the desired profile.
3. Trigger provisioning via the CLI or API to download any newly referenced
   models.
4. Document custom providers or parameters alongside the profile definition so
   future maintainers understand dependency requirements.
