"""Background scheduler orchestrating evolution training cycles."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

import psutil
from apscheduler.schedulers.background import BackgroundScheduler

try:  # pragma: no cover - optional dependency at runtime
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional in some environments
    torch = None  # type: ignore[assignment]

from modules.evolution_engine.self_training import collect_curated_data
from modules.neurons.registry import update_manifest
from modules.neurons.training.mntp_trainer import MNTPTrainer, TrainingStatus
from modules.ray_service import update_ray_deployment
from monGARS.config import get_settings
from monGARS.core.model_slot_manager import ModelSlotManager

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
DEFAULT_REGISTRY_PATH = Path("models/encoders")
DEFAULT_CONFIG_PATH = Path("configs/training/mntp_mistral_config.json")
TRAINING_SUMMARY_FILENAME = "training_summary.json"
MAX_VRAM_GB = 6.0
CPU_IDLE_THRESHOLD = 20.0
MEMORY_IDLE_THRESHOLD = 70.0
SCHEDULER_JOB_ID = "evolution-engine-training-cycle"

CuratedDataset = Sequence[dict[str, Any]] | Any


class EvolutionOrchestrator:
    """Coordinate background MNTP training cycles with resource safeguards."""

    def __init__(
        self,
        *,
        model_registry_path: str | os.PathLike[str] = DEFAULT_REGISTRY_PATH,
        training_config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
        model_id: str = DEFAULT_MODEL_ID,
        trainer_cls: type[MNTPTrainer] = MNTPTrainer,
        slot_manager_cls: type[ModelSlotManager] | None = ModelSlotManager,
        data_collector: Callable[[], CuratedDataset] = collect_curated_data,
        scheduler: BackgroundScheduler | None = None,
        autostart: bool = True,
        scheduler_job_id: str = SCHEDULER_JOB_ID,
    ) -> None:
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.training_config_path = Path(training_config_path)
        self.model_id = model_id
        self._trainer_cls = trainer_cls
        self._slot_manager_cls = slot_manager_cls
        self._data_collector = data_collector
        self._scheduler = scheduler or BackgroundScheduler()
        self._scheduler_job = self._scheduler.add_job(
            self.run_training_cycle,
            "interval",
            minutes=20,
            jitter=300,
            id=scheduler_job_id,
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        if autostart and not self._scheduler.running:
            self._scheduler.start()

    @property
    def scheduler(self) -> BackgroundScheduler:
        return self._scheduler

    def shutdown(self, *, wait: bool = False) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)

    def trigger_encoder_training_pipeline(self) -> str:
        run_dir = self.run_training_cycle(force=True)
        if run_dir is None:
            raise RuntimeError("Training cycle did not produce any artifacts")
        return str(run_dir)

    def run_training_cycle(self, *, force: bool = False) -> Path | None:
        if not force and not self._system_is_idle():
            logger.info("Skipping training cycle because the host is busy")
            return None

        dataset = self._collect_dataset()
        if self._dataset_empty(dataset):
            logger.info("Skipping training cycle because no curated data is available")
            return None

        run_dir = self._prepare_run_directory()
        trainer = self._instantiate_trainer(run_dir)

        try:
            with self._acquire_model_slot():
                summary = trainer.fit(dataset)
        except Exception:
            logger.exception("Evolution training cycle failed during MNTP fitting")
            raise

        status = str(summary.get("status") or "").lower()
        if status != TrainingStatus.SUCCESS.value:
            raise RuntimeError(
                f"MNTP trainer reported unsuccessful status: {summary.get('status')!r}"
            )

        summary.setdefault("version", summary.get("version") or uuid4().hex)
        summary.setdefault("completed_at", datetime.now(timezone.utc).isoformat())

        try:
            self._write_summary(run_dir, summary)
        except Exception:  # pragma: no cover - defensive persistence guard
            logger.exception(
                "Failed to persist training summary", extra={"run_dir": str(run_dir)}
            )

        try:
            manifest = update_manifest(self.registry_path, summary)
            logger.info(
                "Adapter manifest updated",
                extra={"manifest_path": str(manifest.path)},
            )
        except Exception:
            logger.exception("Failed to update adapter manifest")
            raise

        try:
            self.rollout_adapter(summary)
        except (
            Exception
        ):  # pragma: no cover - rollout should never crash the orchestrator
            logger.exception("Adapter rollout failed")

        return run_dir

    def rollout_adapter(self, summary: dict[str, Any]) -> None:
        if not self._ray_rollout_enabled():
            return

        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, dict):
            logger.warning(
                "Training summary missing artifacts payload; skipping rollout"
            )
            return

        adapter_path = artifacts.get("adapter")
        if not adapter_path:
            logger.warning("Training summary missing adapter path; skipping rollout")
            return

        payload = {
            "adapter_path": str(adapter_path),
            "version": str(summary.get("version") or ""),
        }
        weights_path = artifacts.get("weights")
        if weights_path:
            payload["weights_path"] = str(weights_path)

        try:
            update_ray_deployment(payload)
        except RuntimeError as exc:
            logger.warning(
                "Failed to update Ray Serve deployment",
                extra={"reason": str(exc)},
            )
        except (
            Exception
        ):  # pragma: no cover - defensive guard for unexpected Ray errors
            logger.exception("Unexpected Ray Serve deployment failure")

    def _collect_dataset(self) -> CuratedDataset:
        try:
            return self._data_collector()
        except Exception:  # pragma: no cover - defensive guard around data ingestion
            logger.exception("Failed to collect curated dataset")
            return []

    def _dataset_empty(self, dataset: CuratedDataset) -> bool:
        if dataset is None:
            return True
        try:
            length = len(dataset)  # type: ignore[arg-type]
        except Exception:
            return False
        return length == 0

    def _instantiate_trainer(self, run_dir: Path) -> MNTPTrainer:
        return self._trainer_cls(
            training_config_path=str(self.training_config_path),
            output_dir=str(run_dir),
        )

    def _prepare_run_directory(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = self.registry_path / f"cycle-{timestamp}-{uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_summary(self, run_dir: Path, summary: dict[str, Any]) -> None:
        path = run_dir / TRAINING_SUMMARY_FILENAME
        path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    @contextmanager
    def _acquire_model_slot(self) -> Any:
        if self._slot_manager_cls is None:
            yield None
            return
        try:
            manager = self._slot_manager_cls("primary", model_id=self.model_id)
        except Exception as exc:  # pragma: no cover - optional dependency failure
            logger.warning(
                "model.slot.unavailable",
                extra={"model_id": self.model_id, "error": str(exc)},
            )
            yield None
            return
        with manager:  # type: ignore[misc]
            yield manager

    def _system_is_idle(self) -> bool:
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception as exc:
            logger.warning("Failed to measure CPU utilisation", exc_info=exc)
            return False

        try:
            memory_percent = float(psutil.virtual_memory().percent)
        except Exception as exc:
            logger.warning("Failed to measure memory utilisation", exc_info=exc)
            return False

        if cpu_percent >= CPU_IDLE_THRESHOLD or memory_percent >= MEMORY_IDLE_THRESHOLD:
            logger.info(
                "System not idle",
                extra={"cpu_percent": cpu_percent, "memory_percent": memory_percent},
            )
            return False

        vram_usage = self._current_vram_usage_gb()
        if vram_usage is not None and vram_usage > MAX_VRAM_GB:
            logger.info(
                "Skipping training due to VRAM pressure",
                extra={"vram_gb": round(vram_usage, 2)},
            )
            return False

        return True

    def _current_vram_usage_gb(self) -> float | None:
        if torch is None or not hasattr(torch, "cuda"):
            return None

        cuda = torch.cuda  # type: ignore[union-attr]
        try:
            if not cuda.is_available():
                return None
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Unable to query CUDA availability", exc_info=exc)
            return None

        try:
            device_count = int(cuda.device_count())
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Unable to enumerate CUDA devices", exc_info=exc)
            return None

        if device_count <= 0:
            return None

        allocations: list[float] = []
        for index in range(device_count):
            try:
                cuda.get_device_properties(index)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Failed to access CUDA device properties",  # pragma: no cover
                    extra={"device_index": index},
                    exc_info=exc,
                )
                return None

            try:
                with cuda.device(index):
                    allocated = float(cuda.memory_allocated())
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Failed to inspect CUDA memory allocation",  # pragma: no cover
                    extra={"device_index": index},
                    exc_info=exc,
                )
                return None

            allocations.append(allocated)

        if not allocations:
            return None
        return max(allocations) / float(1024**3)

    def _ray_rollout_enabled(self) -> bool:
        try:
            settings = get_settings()
        except Exception:  # pragma: no cover - defensive guard for config access
            settings = None

        for attr in ("use_ray_serve", "USE_RAY_SERVE", "use_ray"):
            if settings is not None and hasattr(settings, attr):
                value = getattr(settings, attr)
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in {"true", "1", "yes", "on"}

        env_flag = os.getenv("USE_RAY_SERVE")
        if env_flag is None:
            return False
        return env_flag.strip().lower() in {"true", "1", "yes", "on"}


__all__ = ["EvolutionOrchestrator"]
