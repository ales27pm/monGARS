"""Workflow-managed orchestration for evolution engine training cycles."""

from __future__ import annotations

import json
import logging
import os
import random
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

import psutil

try:  # pragma: no cover - optional dependency in lightweight environments
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch may be absent in CI
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - Prefect is optional but preferred for orchestration
    from prefect import flow as prefect_flow
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import IntervalSchedule
except Exception:  # pragma: no cover - gracefully degrade when Prefect missing
    prefect_flow = None  # type: ignore[assignment]
    Deployment = None  # type: ignore[assignment]
    IntervalSchedule = None  # type: ignore[assignment]

from modules.evolution_engine.energy import EnergyTracker, EnergyUsageReport
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
ENERGY_REPORT_FILENAME = "energy_report.json"
MAX_VRAM_GB = 6.0
CPU_IDLE_THRESHOLD = 20.0
MEMORY_IDLE_THRESHOLD = 70.0
WORKFLOW_NAME = "evolution-training-flow"
WORKFLOW_DEPLOYMENT_NAME = "evolution-training-deployment"

CuratedDataset = Sequence[dict[str, Any]] | Any
EnergyTrackerFactory = Callable[[], EnergyTracker]


class WorkflowBackend(Protocol):
    """Protocol describing the minimal workflow backend surface area."""

    def build_flow(
        self, func: Callable[..., Any], *, name: str
    ) -> Callable[..., Any]:
        """Return a callable representing the orchestrated flow."""

    def ensure_schedule(
        self, flow: Callable[..., Any], *, parameters: Mapping[str, Any]
    ) -> None:
        """Register or update the recurring schedule for ``flow``."""

    def run(
        self, flow: Callable[..., Any], *, parameters: Mapping[str, Any]
    ) -> Any:
        """Execute ``flow`` with ``parameters`` and return its result."""


class InlineWorkflowBackend:
    """A lightweight backend that executes flows synchronously."""

    def __init__(self) -> None:
        self.last_schedule: dict[str, Any] | None = None

    def build_flow(
        self, func: Callable[..., Any], *, name: str
    ) -> Callable[..., Any]:  # noqa: D401 - signature enforced by protocol
        return func

    def ensure_schedule(
        self, flow: Callable[..., Any], *, parameters: Mapping[str, Any]
    ) -> None:
        self.last_schedule = {"flow": flow, "parameters": dict(parameters)}

    def run(
        self, flow: Callable[..., Any], *, parameters: Mapping[str, Any]
    ) -> Any:
        return flow(**parameters)


class PrefectWorkflowBackend:
    """Workflow backend leveraging Prefect deployments for scheduling."""

    def __init__(
        self,
        *,
        flow_name: str,
        deployment_name: str,
        interval_minutes: int,
        jitter_seconds: float,
    ) -> None:
        if (
            prefect_flow is None
            or Deployment is None
            or IntervalSchedule is None
        ):  # pragma: no cover - optional dependency branch
            raise RuntimeError("Prefect is not available in this environment")

        self._flow_decorator = prefect_flow
        self._deployment_cls = Deployment
        self._schedule_cls = IntervalSchedule
        self._flow_name = flow_name
        self._deployment_name = deployment_name
        self._interval = timedelta(minutes=max(1, int(interval_minutes)))
        self._anchor_offset = max(0.0, float(jitter_seconds))

    def build_flow(
        self, func: Callable[..., Any], *, name: str
    ) -> Callable[..., Any]:
        return self._flow_decorator(name=name)(func)

    def ensure_schedule(
        self, flow: Callable[..., Any], *, parameters: Mapping[str, Any]
    ) -> None:
        anchor = datetime.now(timezone.utc)
        if self._anchor_offset > 0:
            anchor += timedelta(seconds=random.uniform(0.0, self._anchor_offset))

        schedule = self._schedule_cls(interval=self._interval, anchor_date=anchor)
        deployment = self._deployment_cls.build_from_flow(
            flow=flow,
            name=self._deployment_name,
            schedule=schedule,
            parameters=dict(parameters),
            tags=["evolution-engine", "training"],
        )
        try:
            deployment.apply()  # type: ignore[no-untyped-call]
        except Exception as exc:  # pragma: no cover - Prefect server optional
            logger.warning(
                "prefect.deployment.apply_failed",
                extra={"deployment": self._deployment_name, "error": str(exc)},
            )

    def run(
        self, flow: Callable[..., Any], *, parameters: Mapping[str, Any]
    ) -> Any:
        return flow(**parameters)


class EvolutionOrchestrator:
    """Coordinate MNTP training cycles using a workflow orchestrator."""

    def __init__(
        self,
        *,
        model_registry_path: str | os.PathLike[str] = DEFAULT_REGISTRY_PATH,
        training_config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
        model_id: str = DEFAULT_MODEL_ID,
        trainer_cls: type[MNTPTrainer] = MNTPTrainer,
        slot_manager_cls: type[ModelSlotManager] | None = ModelSlotManager,
        data_collector: Callable[[], CuratedDataset] = collect_curated_data,
        workflow_backend: WorkflowBackend | None = None,
        schedule_interval_minutes: int = 20,
        schedule_jitter_seconds: float = 300.0,
        flow_name: str = WORKFLOW_NAME,
        deployment_name: str = WORKFLOW_DEPLOYMENT_NAME,
        energy_tracker_factory: EnergyTrackerFactory | None = None,
    ) -> None:
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.training_config_path = Path(training_config_path)
        self.model_id = model_id
        self._trainer_cls = trainer_cls
        self._slot_manager_cls = slot_manager_cls
        self._data_collector = data_collector
        self._energy_tracker_factory = energy_tracker_factory
        self._schedule_interval_minutes = int(schedule_interval_minutes)
        self._schedule_jitter_seconds = max(0.0, float(schedule_jitter_seconds))
        self._flow_name = flow_name
        self._deployment_name = deployment_name

        self._workflow_backend = (
            workflow_backend
            if workflow_backend is not None
            else self._default_backend()
        )
        self._flow = self._workflow_backend.build_flow(
            self._run_flow, name=self._flow_name
        )
        self._register_schedule()

    @property
    def workflow_backend(self) -> WorkflowBackend:
        return self._workflow_backend

    def trigger_encoder_training_pipeline(self) -> str:
        run_dir = self.run_training_cycle(force=True)
        if run_dir is None:
            raise RuntimeError("Training cycle did not produce any artifacts")
        return str(run_dir)

    def run_training_cycle(self, *, force: bool = False) -> Path | None:
        try:
            result = self._workflow_backend.run(
                self._flow, parameters={"force": force}
            )
        except Exception:
            logger.exception("Workflow execution failed")
            raise

        if result is None:
            return None

        return Path(result)

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

        payload: dict[str, Any] = {
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
        except Exception:  # pragma: no cover - unexpected Ray exceptions
            logger.exception("Unexpected Ray Serve deployment failure")

    def _default_backend(self) -> WorkflowBackend:
        try:
            return PrefectWorkflowBackend(
                flow_name=self._flow_name,
                deployment_name=self._deployment_name,
                interval_minutes=self._schedule_interval_minutes,
                jitter_seconds=self._schedule_jitter_seconds,
            )
        except RuntimeError:
            logger.info(
                "Prefect not available; using synchronous workflow backend instead."
            )
            return InlineWorkflowBackend()

    def _register_schedule(self) -> None:
        try:
            self._workflow_backend.ensure_schedule(
                self._flow, parameters={"force": False}
            )
        except Exception:
            logger.exception("Failed to register evolution workflow schedule")

    def _run_flow(self, *, force: bool = False) -> str | None:
        run_dir = self._execute_training_cycle(force=force)
        return str(run_dir) if run_dir is not None else None

    def _execute_training_cycle(self, *, force: bool) -> Path | None:
        if not force and not self._system_is_idle():
            logger.info("Skipping training cycle because the host is busy")
            return None

        dataset = self._collect_dataset()
        if self._dataset_empty(dataset):
            logger.info(
                "Skipping training cycle because no curated data is available"
            )
            return None

        run_dir = self._prepare_run_directory()
        trainer = self._instantiate_trainer(run_dir)
        tracker = self._energy_tracker_factory() if self._energy_tracker_factory else None
        energy_report: EnergyUsageReport | None = None

        try:
            with self._acquire_model_slot():
                if tracker is not None:
                    with tracker.track():
                        summary = trainer.fit(dataset)
                    energy_report = tracker.last_report
                else:
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

        self._persist_run_artifacts(run_dir, summary, energy_report)
        self._update_manifest(summary)
        try:
            self.rollout_adapter(summary)
        except Exception:  # pragma: no cover - rollout failures must not crash
            logger.exception("Adapter rollout failed")

        return run_dir

    def _persist_run_artifacts(
        self,
        run_dir: Path,
        summary: dict[str, Any],
        energy_report: EnergyUsageReport | None,
    ) -> None:
        try:
            self._write_summary(run_dir, summary)
        except Exception:  # pragma: no cover - defensive persistence guard
            logger.exception(
                "Failed to persist training summary", extra={"run_dir": str(run_dir)}
            )

        if energy_report is None:
            return

        try:
            self._write_energy_report(run_dir, energy_report)
        except Exception:  # pragma: no cover - defensive persistence guard
            logger.exception(
                "Failed to persist energy report", extra={"run_dir": str(run_dir)}
            )

    def _update_manifest(self, summary: dict[str, Any]) -> None:
        try:
            manifest = update_manifest(self.registry_path, summary)
            logger.info(
                "Adapter manifest updated",
                extra={"manifest_path": str(manifest.path)},
            )
        except Exception:
            logger.exception("Failed to update adapter manifest")
            raise

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

    def _write_energy_report(
        self, run_dir: Path, report: EnergyUsageReport
    ) -> None:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        path = run_dir / ENERGY_REPORT_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

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
                    "Failed to access CUDA device properties",
                    extra={"device_index": index},
                    exc_info=exc,
                )
                return None

            try:
                with cuda.device(index):
                    allocated = float(cuda.memory_allocated())
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Failed to inspect CUDA memory allocation",
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


__all__ = ["EvolutionOrchestrator", "InlineWorkflowBackend", "PrefectWorkflowBackend"]

