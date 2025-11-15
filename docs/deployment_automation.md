# Full-Stack Deployment Automation

> **Last updated:** 2025-02-16 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The `scripts/full_stack_visual_deploy.py` helper orchestrates every
prerequisite required to bring the monGARS research stack online. It
wraps environment bootstrapping, dependency installation, optional
SearxNG provisioning, and container launch into a guided, visual
terminal experience. The script is wired into the
[`docs/index.md`](index.md) hub under Operations so engineers can spot
updates at a glance.

## Visual flow

```mermaid
flowchart TD
    A[Validate prerequisites] --> B[Prepare .env]
    B --> C[Install Python backend]
    C --> D[Install root npm packages]
    D --> E[Install React Native dependencies]
    E --> F{Provision SearxNG?}
    F -- yes --> G[Render SearxNG config]
    F -- no --> H[Skip SearxNG]
    G --> I[Build & launch docker-compose]
    H --> I[Build & launch docker-compose]
```

## Usage

```bash
python scripts/full_stack_visual_deploy.py
```

The wizard renders each step with a bordered block, streams command
output for visibility, and writes a combined transcript to
`deployment.log`. Use `--project-root` to target a checked-out
repository elsewhere, `--skip-searx` to omit SearxNG provisioning, or
`--only` with one or more step keys to focus on a subset (for example,
`--only python containers`).

The installer performs the following actions:

1. **Validate prerequisites** – ensures `python3`, `pip`, `npm`,
   `node`, `docker`, and a Compose binary (`docker compose` or
   `docker-compose`) are present.
2. **Prepare environment file** – copies `.env.example` to `.env` if it
   does not already exist, generates secure secrets for FastAPI,
   Django, PostgreSQL, and SearxNG, adds opinionated SearxNG keys
   (`SEARCH_SEARX_*`, including `SEARCH_SEARX_INTERNAL_BASE_URL` so
   containers point at `http://searxng:8080`), and keeps
   `SEARXNG_BASE_URL`/`SEARCH_SEARX_BASE_URL` aligned with
   `SEARXNG_PORT` when operators customise the exposed port.
3. **Install Python backend** – creates `.venv`, upgrades `pip`, and
   installs `requirements.txt` using the virtual environment interpreter.
4. **Install web dependencies** – runs `npm install` in the repository
   root for shared tooling and the Django operator console assets.
5. **Install React Native dependencies** – runs `npm install` inside
   `mobile-app/` and executes `npx pod-install` on macOS when CocoaPods
   is available.
6. **Provision SearxNG** – writes `configs/searxng/settings.yml` if it
   does not exist so the orchestrator can talk to the bundled
   SearxNG instance.
7. **Launch containers** – executes `docker compose pull`, `build`, and
   `up -d` with `docker-compose.searxng.yml` layered on top of
   `docker-compose.yml` when SearxNG is enabled, mirroring the
   production-oriented profile used by `scripts/docker_menu.py` option 11.
   The overlay pins `searxng/searxng:2025.10.24-2cdbbb249` because the
   upstream 2024.6.9 series was withdrawn from Docker Hub, ensuring
   `docker compose pull` succeeds during automated deployments.

Each step key matches the `--only` flag values: `prerequisites`, `env`,
`python`, `root-node`, `mobile`, `searxng`, and `containers`.

## Fine-tune, deploy, and launch in one pass

Operators who need to roll a freshly tuned adapter directly into the
local docker stack can rely on
`scripts/finetune_deploy_launch.py`. The helper stitches together the
existing Unsloth fine-tuning pipeline, manifest refresh, docker-compose
deployment, and a browser launch for the Django console.

```bash
python scripts/finetune_deploy_launch.py \
  --model-id dphn/Dolphin3.0-Llama3.1-8B \
  --dataset-path datasets/unsloth/monGARS_unsloth_dataset.jsonl
```

Key behaviour:

- Caches the base model snapshot ahead of training using
  `huggingface_hub.snapshot_download`, storing artefacts under
  `models/base_snapshots/<model>/` by default. Provide `--model-cache-dir`,
  `--model-revision`, or authentication via `--hf-token`/`HF_TOKEN` to
  customise the fetch.
- Reuses the `EnvFileManager` logic from the visual deployer to create a
  `.env` file and rotate secrets when `--skip-env` is omitted.
- Invokes `monGARS.mlops.pipelines.run_unsloth_finetune` with the same
  defaults exposed by `scripts/run_mongars_llm_pipeline.py`, writing
  artefacts under `models/encoders/monGARS_unsloth/runs/<timestamp>/` by
  default.
- Ensures 4-bit quantisation is tracked in the adapter manifest and labels
  the export with `llm2vec_export=enabled`, making it clear that the wrapper
  produced by `write_wrapper_bundle` supports both chat and embedding flows.
- Persists adapter metadata through
  `modules.neurons.registry.update_manifest` so FastAPI and background
  workers pick up the new revision without manual edits.
- Executes `docker compose pull`, `build`, and `up -d` (with optional
  extra compose overlays supplied via `--compose-file`) before polling
  `http://localhost:8000/healthz` and
  `http://localhost:8001/chat/login/` for readiness.
- Opens `http://localhost:8001/chat/` in the default browser unless
  `--no-browser` is supplied, making it clear when the UI is ready for
  validation.

Use `--skip-pull` to avoid pulling images during local iteration, set
`--dataset-id` when training from Hugging Face instead of a local JSONL
file, point `--registry-path` at a custom encoder directory when testing
isolated manifests, and supply `--skip-model-download` if the base model is
already present in the Transformers cache.

## React Native build tips

The automation script prepares dependencies but does not start Metro or
native builds. After it finishes, run the following from `mobile-app/`:

```bash
npm run start           # Launch the Metro bundler
npm run ios -- --device # Build for a connected iOS device or simulator
npm run android         # Build for Android emulator/device
```

On macOS, ensure you have installed the Xcode command-line tools and
run `sudo xcode-select --switch /Applications/Xcode.app`. Android
builds require Android Studio with the SDK manager configured for API
34 or newer.

## Manual override switches

The wizard exits immediately on the first failure when `--non-interactive`
 is supplied. Without the flag it prompts to continue or abort on errors.
To diagnose Compose-specific problems, rerun the container step only:

```bash
python scripts/full_stack_visual_deploy.py --only containers --non-interactive
```

Logs remain in `deployment.log`, ensuring parity between interactive
runs and automated CI usage. Attach that log to incident tickets so the
operations team can compare it with the CI artefacts produced by the
`docker-build` workflow job described in [docs/workflow_reference.md](workflow_reference.md).

## Keeping this runbook current
- Update the step descriptions when `scripts/full_stack_visual_deploy.py`
  changes; mention new CLI flags in both this document and the script docstring.
- Verify the Mermaid flow mirrors the order in `_build_steps()` after any edit.
- When Compose files or secrets management workflows move, cross-link the
  updated locations and bump the "Last updated" banner.
