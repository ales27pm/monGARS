#!/usr/bin/env python3
"""End-to-end automation from fine-tuning to docker deployment and browser launch."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.neurons.registry import update_manifest  # noqa: E402
from monGARS.mlops.artifacts import build_adapter_summary  # noqa: E402
from monGARS.mlops.pipelines import run_unsloth_finetune  # noqa: E402
from scripts.full_stack_visual_deploy import (  # noqa: E402
    DeploymentError,
    EnvFileManager,
)

LOGGER = logging.getLogger("monGARS.finetune_deploy")

DEFAULT_MODEL_ID = "dphn/Dolphin3.0-Llama3.1-8B"
DEFAULT_DATASET_PATH = Path("datasets/unsloth/monGARS_unsloth_dataset.jsonl")
DEFAULT_EVAL_DATASET_PATH = Path("datasets/monGARS_llm/monGARS_llm_val.jsonl")
DEFAULT_MODEL_CACHE_ROOT = Path("models/base_snapshots")
DEFAULT_REGISTRY_PATH = Path("models/encoders/monGARS_unsloth")
DEFAULT_WEBAPP_URL = "http://localhost:8001/chat/"
DEFAULT_WEB_HEALTH_URL = "http://localhost:8001/chat/login/"
DEFAULT_API_HEALTH_URL = "http://localhost:8000/healthz"
DEFAULT_COMPOSE_FILE = Path("docker-compose.yml")


def _sanitise_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "-")


def _default_model_cache_dir(model_id: str) -> Path:
    return DEFAULT_MODEL_CACHE_ROOT / _sanitise_model_id(model_id)


class AutomationError(RuntimeError):
    """Raised when a pipeline step fails irrecoverably."""


class StepStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(slots=True)
class Step:
    key: str
    title: str
    handler: Callable[
        ["FineTuneDeploymentAutomation"], tuple[StepStatus, str | None] | StepStatus
    ]
    description: str


@dataclass(slots=True)
class StepResult:
    key: str
    title: str
    status: StepStatus
    message: str | None = None


class FineTuneDeploymentAutomation:
    """Coordinate fine-tuning, container deployment, and browser launch."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = args.project_root
        self.registry_path = args.registry_path
        self.model_cache_dir = args.model_cache_dir
        if not hasattr(self.args, "compose_files"):
            self.args.compose_files = []
        self.env_manager = EnvFileManager(self.project_root, LOGGER)
        self._compose_cache: Sequence[str] | None = None
        self._finetune_results: dict[str, object] | None = None
        self._manifest_path: Path | None = None
        self._model_snapshot_path: Path | None = None

    def run(self) -> int:
        results: list[StepResult] = []
        for step in self._build_steps():
            LOGGER.info("Starting step", extra={"step": step.key, "title": step.title})
            print(f"\n=== {step.title} ===")
            print(step.description)
            try:
                outcome = step.handler(self)
            except (AutomationError, DeploymentError) as exc:
                LOGGER.error("Step failed", exc_info=True)
                results.append(
                    StepResult(step.key, step.title, StepStatus.FAILED, str(exc))
                )
                self._render_summary(results)
                return 1
            except Exception:  # noqa: BLE001 - defensive guard to surface tracebacks
                LOGGER.exception(
                    "Unexpected error during step", extra={"step": step.key}
                )
                results.append(
                    StepResult(
                        step.key, step.title, StepStatus.FAILED, "unexpected error"
                    )
                )
                self._render_summary(results)
                return 1
            else:
                status, message = self._normalise_outcome(outcome)
                LOGGER.info(
                    "Step finished", extra={"step": step.key, "status": status.value}
                )
                results.append(StepResult(step.key, step.title, status, message))
        self._render_summary(results)
        return 0

    # --- Step handlers -------------------------------------------------

    def step_validate_tooling(self) -> tuple[StepStatus, str | None]:
        missing: list[str] = []
        for command in ("python3", "pip", "docker"):
            if shutil.which(command) is None:
                missing.append(command)
        compose = self._compose_invocation()
        if not compose:
            missing.append("docker compose")
        if missing:
            raise AutomationError(
                "Missing required commands: " + ", ".join(sorted(set(missing)))
            )
        return StepStatus.SUCCESS, None

    def step_prepare_env(self) -> tuple[StepStatus, str | None]:
        if self.args.skip_env:
            return StepStatus.SKIPPED, "Environment preparation skipped by flag"
        try:
            self.env_manager.ensure_env_file()
            self.env_manager.ensure_secure_defaults()
        except DeploymentError as exc:  # re-raise with pipeline context
            raise AutomationError(str(exc)) from exc
        return StepStatus.SUCCESS, None

    def step_download_model(self) -> tuple[StepStatus, str | None]:
        if self.args.skip_model_download:
            return StepStatus.SKIPPED, "Model download skipped by flag"
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "huggingface_hub unavailable; skipping explicit model download",
                extra={"error": str(exc)},
            )
            return StepStatus.SKIPPED, "huggingface_hub not installed"

        cache_dir = self.model_cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        token = (
            self.args.hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        kwargs: dict[str, object] = {
            "repo_id": self.args.model_id,
            "resume_download": True,
            "local_dir_use_symlinks": False,
        }
        if cache_dir is not None:
            kwargs["local_dir"] = str(cache_dir)
        if self.args.model_revision:
            kwargs["revision"] = self.args.model_revision
        if token:
            kwargs["token"] = token
        if self.args.model_allow_patterns:
            kwargs["allow_patterns"] = list(self.args.model_allow_patterns)
        if self.args.model_ignore_patterns:
            kwargs["ignore_patterns"] = list(self.args.model_ignore_patterns)

        LOGGER.info(
            "Downloading base model snapshot",
            extra={
                "model_id": self.args.model_id,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "revision": self.args.model_revision,
            },
        )
        try:
            snapshot_path = snapshot_download(**kwargs)
        except Exception as exc:  # pragma: no cover - network or auth issues
            raise AutomationError(f"Model download failed: {exc}") from exc

        resolved = Path(snapshot_path).resolve()
        self._model_snapshot_path = resolved
        return StepStatus.SUCCESS, f"Cached model to {resolved}"

    def step_finetune(self) -> tuple[StepStatus, str | None]:
        dataset_path = self.args.dataset_path
        if self.args.dataset_id is None and dataset_path is None:
            raise AutomationError(
                "Provide --dataset-id or --dataset-path for fine-tuning"
            )
        if dataset_path is not None and not dataset_path.exists():
            raise AutomationError(f"Dataset path not found: {dataset_path}")
        if (
            self.args.eval_dataset_path is not None
            and not self.args.eval_dataset_path.exists()
        ):
            raise AutomationError(
                f"Evaluation dataset path not found: {self.args.eval_dataset_path}"
            )

        output_dir = self.args.output_dir or self._default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Running fine-tuning",
            extra={
                "model_id": self.args.model_id,
                "dataset_id": self.args.dataset_id,
                "dataset_path": str(dataset_path) if dataset_path else None,
                "output_dir": str(output_dir),
            },
        )
        results = run_unsloth_finetune(
            model_id=self.args.model_id,
            output_dir=output_dir,
            dataset_id=self.args.dataset_id,
            dataset_path=dataset_path,
            max_seq_len=self.args.max_seq_len,
            vram_budget_mb=self.args.vram_budget_mb,
            activation_buffer_mb=self.args.activation_buffer_mb,
            batch_size=self.args.batch_size,
            grad_accum=self.args.grad_accum,
            learning_rate=self.args.learning_rate,
            epochs=self.args.epochs,
            max_steps=self.args.max_steps,
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            train_fraction=self.args.train_fraction,
            eval_dataset_id=self.args.eval_dataset_id,
            eval_dataset_path=self.args.eval_dataset_path,
            eval_batch_size=self.args.eval_batch_size,
            run_smoke_tests=not self.args.skip_smoke_tests,
            write_metadata=not self.args.skip_metadata,
            merge_to_fp16=not self.args.skip_merge,
        )
        self._finetune_results = {**results, "output_dir": output_dir}
        summary = (
            f"LoRA saved to {results['chat_lora_dir']}"
            if "chat_lora_dir" in results
            else "Fine-tuning completed"
        )
        return StepStatus.SUCCESS, summary

    def step_update_manifest(self) -> tuple[StepStatus, str | None]:
        if self.args.skip_manifest:
            return StepStatus.SKIPPED, "Manifest update skipped by flag"
        if self._finetune_results is None:
            raise AutomationError("Fine-tuning results missing; cannot update manifest")

        adapter_dir = Path(self._finetune_results["chat_lora_dir"])
        wrapper_dir = (
            Path(self._finetune_results["wrapper_dir"])
            if self._finetune_results.get("wrapper_dir")
            else None
        )
        weights_dir = self._finetune_results.get("merged_dir")
        weights_path = Path(weights_dir) if weights_dir else None

        metrics = {
            "dataset_size": self._finetune_results.get("dataset_size"),
            "eval_dataset_size": self._finetune_results.get("eval_dataset_size"),
            "quantized_4bit": self._finetune_results.get("quantized_4bit"),
        }
        evaluation_metrics = self._finetune_results.get("evaluation_metrics")
        if isinstance(evaluation_metrics, dict) and evaluation_metrics:
            metrics["evaluation"] = evaluation_metrics
        filtered_metrics = {k: v for k, v in metrics.items() if v is not None}

        training = {
            "model_id": self.args.model_id,
            "dataset_id": self.args.dataset_id,
            "dataset_path": (
                str(self.args.dataset_path) if self.args.dataset_path else None
            ),
            "eval_dataset_id": self.args.eval_dataset_id,
            "eval_dataset_path": (
                str(self.args.eval_dataset_path)
                if self.args.eval_dataset_path is not None
                else None
            ),
            "max_seq_len": self.args.max_seq_len,
            "batch_size": self.args.batch_size,
            "grad_accum": self.args.grad_accum,
            "learning_rate": self.args.learning_rate,
            "epochs": self.args.epochs,
            "max_steps": self.args.max_steps,
            "lora_rank": self.args.lora_rank,
            "lora_alpha": self.args.lora_alpha,
            "lora_dropout": self.args.lora_dropout,
            "train_fraction": self.args.train_fraction,
            "eval_batch_size": self.args.eval_batch_size,
        }
        filtered_training = {k: v for k, v in training.items() if v is not None}

        labels = {"pipeline": "unsloth_llm2vec"}
        quantized_flag = self._finetune_results.get("quantized_4bit")
        if quantized_flag is not None:
            labels["quantization"] = "4bit" if quantized_flag else "fp16"
        if wrapper_dir is not None:
            labels["llm2vec_export"] = "enabled"
        if self._model_snapshot_path is not None:
            labels["base_snapshot"] = str(self._model_snapshot_path)

        summary = build_adapter_summary(
            adapter_dir=adapter_dir,
            weights_path=weights_path,
            wrapper_dir=wrapper_dir,
            status="success",
            labels=labels,
            metrics=filtered_metrics,
            training=filtered_training,
        )
        manifest = update_manifest(self.registry_path, summary)
        self._manifest_path = manifest.path
        message = f"Adapter manifest updated at {manifest.path}"
        return StepStatus.SUCCESS, message

    def step_launch_containers(self) -> tuple[StepStatus, str | None]:
        compose = self._compose_invocation()
        if not compose:
            raise AutomationError("Docker Compose command not available")
        compose_files = [DEFAULT_COMPOSE_FILE]
        compose_files.extend(self.args.compose_files)
        base_args: list[str] = list(compose)
        for path in compose_files:
            resolved = self._resolve_project_path(path)
            if not resolved.exists():
                raise AutomationError(f"Missing compose file: {resolved}")
            base_args.extend(["-f", str(resolved)])
        if not self.args.skip_pull:
            self._run_command(base_args + ["pull"])
        self._run_command(base_args + ["build"])
        self._run_command(base_args + ["up", "-d"])
        return StepStatus.SUCCESS, "Docker stack is running"

    def step_wait_for_stack(self) -> tuple[StepStatus, str | None]:
        self._wait_for_url(self.args.api_health_url, "FastAPI service")
        self._wait_for_url(self.args.web_health_url, "Django webapp")
        return StepStatus.SUCCESS, f"Services healthy; web UI at {self.args.webapp_url}"

    def step_open_browser(self) -> tuple[StepStatus, str | None]:
        if self.args.no_browser:
            return StepStatus.SKIPPED, "Browser launch disabled"
        try:
            opened = webbrowser.open(self.args.webapp_url, new=2, autoraise=True)
        except webbrowser.Error as exc:
            LOGGER.warning("Failed to open browser", exc_info=True)
            return StepStatus.SKIPPED, str(exc)
        if opened:
            return StepStatus.SUCCESS, f"Opened {self.args.webapp_url}"
        LOGGER.warning(
            "webbrowser.open returned False", extra={"url": self.args.webapp_url}
        )
        return StepStatus.SKIPPED, "Browser reported failure"

    # --- Helpers -------------------------------------------------------

    def _build_steps(self) -> Iterable[Step]:
        return (
            Step(
                "validate",
                "Validate local tooling",
                FineTuneDeploymentAutomation.step_validate_tooling,
                "Ensure docker, python, and docker compose are available",
            ),
            Step(
                "env",
                "Prepare environment",
                FineTuneDeploymentAutomation.step_prepare_env,
                "Create .env and rotate secrets when necessary",
            ),
            Step(
                "download",
                "Download base model",
                FineTuneDeploymentAutomation.step_download_model,
                "Cache the Hugging Face snapshot required for training",
            ),
            Step(
                "finetune",
                "Fine-tune base model",
                FineTuneDeploymentAutomation.step_finetune,
                "Execute the Unsloth training pipeline and capture artefacts",
            ),
            Step(
                "manifest",
                "Refresh adapter manifest",
                FineTuneDeploymentAutomation.step_update_manifest,
                "Persist the trained adapter metadata for runtime services",
            ),
            Step(
                "compose",
                "Build and start containers",
                FineTuneDeploymentAutomation.step_launch_containers,
                "Run docker compose pull/build/up for the monGARS stack",
            ),
            Step(
                "health",
                "Wait for stack readiness",
                FineTuneDeploymentAutomation.step_wait_for_stack,
                "Poll API and web health endpoints until they report success",
            ),
            Step(
                "browser",
                "Open web console",
                FineTuneDeploymentAutomation.step_open_browser,
                "Launch the operator browser session",
            ),
        )

    def _render_summary(self, results: Iterable[StepResult]) -> None:
        print("\nSummary")
        print("-" * 40)
        for result in results:
            symbol = {
                StepStatus.SUCCESS: "✔",
                StepStatus.SKIPPED: "⚠",
                StepStatus.FAILED: "✖",
            }[result.status]
            message = f" ({result.message})" if result.message else ""
            print(f"{symbol} {result.title}: {result.status.value}{message}")

    @staticmethod
    def _normalise_outcome(
        outcome: tuple[StepStatus, str | None] | StepStatus,
    ) -> tuple[StepStatus, str | None]:
        if isinstance(outcome, tuple):
            return outcome
        return outcome, None

    def _compose_invocation(self) -> Sequence[str] | None:
        if self._compose_cache is not None:
            return self._compose_cache
        docker = shutil.which("docker")
        if docker is not None:
            completed = subprocess.run(
                [docker, "compose", "version"], capture_output=True, text=True
            )
            if completed.returncode == 0:
                self._compose_cache = (docker, "compose")
                return self._compose_cache
        docker_compose = shutil.which("docker-compose")
        if docker_compose is not None:
            self._compose_cache = (docker_compose,)
        return self._compose_cache

    def _run_command(self, command: Sequence[str]) -> None:
        LOGGER.info("Running command", extra={"command": " ".join(command)})
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as exc:
            raise AutomationError(f"Executable not found: {command[0]}") from exc
        except OSError as exc:  # pragma: no cover - unexpected subprocess failure
            raise AutomationError(str(exc)) from exc
        assert process.stdout is not None
        for line in process.stdout:
            print(f"    {line.rstrip()}")
        process.wait()
        if process.returncode != 0:
            raise AutomationError(
                f"Command failed with exit code {process.returncode}: {' '.join(command)}"
            )

    def _wait_for_url(self, url: str, label: str) -> None:
        deadline = time.monotonic() + self.args.poll_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
                    status = response.status
                    if 200 <= status < 400:
                        LOGGER.info(
                            "Service healthy",
                            extra={"label": label, "url": url, "status": status},
                        )
                        return
                    LOGGER.warning(
                        "Health check returned non-success",
                        extra={"label": label, "url": url, "status": status},
                    )
            except urllib.error.URLError as exc:
                LOGGER.debug(
                    "Health check failed",
                    extra={"label": label, "url": url, "error": str(exc)},
                )
            time.sleep(self.args.poll_interval)
        raise AutomationError(f"Timed out waiting for {label} at {url}")

    def _default_output_dir(self) -> Path:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = self.registry_path / "runs" / timestamp
        return base

    def _resolve_project_path(self, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (self.project_root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory",
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID, help="Base model identifier"
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Specific revision or commit hash to download from Hugging Face",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="Directory to store the downloaded base model snapshot",
    )
    parser.add_argument(
        "--model-allow-pattern",
        action="append",
        dest="model_allow_patterns",
        default=[],
        help="Glob patterns to include when downloading the model snapshot",
    )
    parser.add_argument(
        "--model-ignore-pattern",
        action="append",
        dest="model_ignore_patterns",
        default=[],
        help="Glob patterns to exclude when downloading the model snapshot",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip the explicit Hugging Face snapshot download step",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token used for authenticated model downloads",
    )
    parser.add_argument("--dataset-id", help="Optional Hugging Face dataset identifier")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Local JSONL dataset path",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=Path,
        default=DEFAULT_EVAL_DATASET_PATH,
        help="Optional evaluation dataset path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store fine-tuning artefacts (defaults to registry/runs/<timestamp>)",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Adapter registry path for manifest updates",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--vram-budget-mb", type=int, default=8192)
    parser.add_argument("--activation-buffer-mb", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--eval-dataset-id")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Disable embedding smoke tests after training",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip writing run_metadata.json",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Disable FP16 adapter merge",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not update the adapter manifest",
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Skip .env bootstrapping",
    )
    parser.add_argument(
        "--compose-file",
        action="append",
        type=Path,
        default=[],
        help="Additional docker compose file(s) to include",
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip docker compose pull before build",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=420,
        help="Seconds to wait for services to become healthy",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between health check attempts",
    )
    parser.add_argument(
        "--api-health-url",
        default=DEFAULT_API_HEALTH_URL,
        help="Health endpoint for the FastAPI service",
    )
    parser.add_argument(
        "--web-health-url",
        default=DEFAULT_WEB_HEALTH_URL,
        help="Endpoint that confirms the Django webapp is ready",
    )
    parser.add_argument(
        "--webapp-url",
        default=DEFAULT_WEBAPP_URL,
        help="URL to open in the browser after the stack is healthy",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not launch a browser after services are ready",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.resolve()
    args.registry_path = _resolve_with_root(args.registry_path, args.project_root)
    args.output_dir = (
        _resolve_with_root(args.output_dir, args.project_root)
        if args.output_dir is not None
        else None
    )
    args.dataset_path = _resolve_with_root(args.dataset_path, args.project_root)
    args.eval_dataset_path = (
        _resolve_with_root(args.eval_dataset_path, args.project_root)
        if args.eval_dataset_path is not None
        else None
    )
    if args.model_cache_dir is None and not args.skip_model_download:
        default_cache = _default_model_cache_dir(args.model_id)
        args.model_cache_dir = _resolve_with_root(default_cache, args.project_root)
    else:
        args.model_cache_dir = _resolve_with_root(
            args.model_cache_dir, args.project_root
        )
    args.compose_files = [
        _resolve_with_root(path, args.project_root) for path in args.compose_file or []
    ]


def _resolve_with_root(value: Path | None, project_root: Path) -> Path | None:
    if value is None:
        return None
    if value.is_absolute():
        return value
    return (project_root / value).resolve()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_paths(args)
    automation = FineTuneDeploymentAutomation(args)
    return automation.run()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
