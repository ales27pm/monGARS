from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable
from uuid import uuid4

from modules.neurons.registry import update_manifest
from modules.neurons.training.mntp_trainer import MNTPTrainer, TrainingStatus

from .energy import EnergyTracker, EnergyUsageReport

logger = logging.getLogger(__name__)


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol describing the trainer expected by the orchestrator."""

    def __init__(self, training_config_path: str, output_dir: str) -> None:
        """Construct a trainer bound to the provided config and output path."""

    def train(self) -> dict[str, object]:
        """Execute the training pipeline and return a summary payload."""


class EvolutionOrchestrator:
    """Coordinate encoder refresh pipelines built around :class:`MNTPTrainer`."""

    def __init__(
        self,
        model_registry_path: str = "models/encoders/",
        config_path: str | None = None,
        *,
        trainer_cls: type[TrainerProtocol] = MNTPTrainer,
        energy_tracker_factory: Callable[[], EnergyTracker] | None = None,
    ) -> None:
        self.model_registry_path = Path(model_registry_path)
        self.config_path = (
            Path(config_path)
            if config_path
            else Path("configs/training/mntp_mistral_config.json")
        )
        self._trainer_cls: type[TrainerProtocol] = trainer_cls
        if energy_tracker_factory is None:
            self._energy_tracker_factory = lambda: EnergyTracker()
        else:
            self._energy_tracker_factory = energy_tracker_factory

    def trigger_encoder_training_pipeline(self) -> str:
        """Launch the MNTP pipeline and return the produced artifact directory."""

        logger.info("Starting training pipeline for a new encoder")
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        unique_dir = self.model_registry_path / f"temp-mistral-mntp-step-{uuid4()}"
        trainer = self._trainer_cls(
            training_config_path=str(self.config_path),
            output_dir=str(unique_dir),
        )
        tracker = (
            self._energy_tracker_factory() if self._energy_tracker_factory else None
        )
        energy_report: EnergyUsageReport | None = None
        if tracker:
            tracker.start()
        try:
            summary = trainer.train()
        except Exception as exc:  # pragma: no cover - unexpected training error
            logger.error("Training failed: %s", exc, exc_info=True)
            if tracker:
                try:
                    energy_report = tracker.stop()
                    self._persist_energy_report(unique_dir, energy_report)
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception(
                        "Failed to persist energy report after training failure",
                        extra={"output_dir": str(unique_dir)},
                    )
            raise
        finally:
            if tracker and energy_report is None:
                try:
                    energy_report = tracker.stop()
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception("Energy tracker stop failed", exc_info=True)
                    energy_report = None

        status_value = summary.get("status")
        status = str(status_value).lower() if status_value is not None else ""
        if status != TrainingStatus.SUCCESS.value:
            logger.error(
                "Trainer returned non-success status",
                extra={
                    "status": status_value,
                    "artifacts": summary.get("artifacts", {}),
                    "metrics": summary.get("metrics", {}),
                    "encoder_path": str(unique_dir),
                },
            )
            raise RuntimeError(
                f"MNTP trainer reported unsuccessful status: {status_value!r}"
            )

        artifacts = summary.get("artifacts") or {}
        adapter_path_raw = artifacts.get("adapter")
        if not adapter_path_raw:
            raise RuntimeError("Trainer did not return an adapter artifact path")

        adapter_path = Path(adapter_path_raw)
        if not adapter_path.exists():
            raise RuntimeError(f"Adapter artifact path '{adapter_path}' does not exist")

        try:
            adapter_path.resolve().relative_to(unique_dir.resolve())
        except Exception as exc:
            raise RuntimeError(
                "Trainer produced adapter artifact outside orchestrator output directory"
            ) from exc

        if weights_path_raw := artifacts.get("weights"):
            weights_path = Path(weights_path_raw)
            if not weights_path.exists():
                raise RuntimeError(
                    f"Adapter weights path '{weights_path}' does not exist"
                )
            try:
                weights_path.resolve().relative_to(unique_dir.resolve())
            except Exception as exc:
                raise RuntimeError(
                    "Trainer produced adapter weights outside orchestrator output directory"
                ) from exc

        if energy_report is not None:
            self._augment_summary_with_energy(unique_dir, summary, energy_report)

        try:
            manifest = update_manifest(self.model_registry_path, summary)
        except Exception:
            logger.error(
                "Adapter manifest update failed",
                extra={
                    "encoder_path": str(unique_dir),
                    "status": summary.get("status"),
                    "artifacts": summary.get("artifacts", {}),
                },
                exc_info=True,
            )
            raise
        logger.info(
            "Pipeline finished",
            extra={
                "encoder_path": str(unique_dir),
                "status": summary.get("status"),
                "artifacts": summary.get("artifacts", {}),
                "manifest_path": str(manifest.path),
            },
        )
        return str(unique_dir)

    def _augment_summary_with_energy(
        self,
        run_dir: Path,
        summary: dict[str, object],
        report: EnergyUsageReport,
    ) -> None:
        if not isinstance(summary.get("metrics"), dict):
            summary["metrics"] = {}

        metrics = summary["metrics"]
        metrics.update(
            {
                "energy_wh": round(report.energy_wh, 4),
                "energy_backend": report.backend,
                "cpu_seconds": round(report.cpu_seconds, 4),
                "run_duration_seconds": round(report.duration_seconds, 4),
            }
        )
        telemetry = summary.setdefault("telemetry", {})
        if isinstance(telemetry, dict):
            telemetry["energy"] = report.to_dict()
        self._persist_energy_report(run_dir, report)
        try:
            (run_dir / "training_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )
        except Exception:  # pragma: no cover - defensive persistence guard
            logger.exception(
                "Failed to update training summary with energy metrics",
                extra={"run_dir": str(run_dir)},
            )

    def _persist_energy_report(self, run_dir: Path, report: EnergyUsageReport) -> None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "energy_report.json").write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True)
            )
        except Exception:  # pragma: no cover - defensive persistence guard
            logger.exception(
                "Failed to write energy report", extra={"run_dir": str(run_dir)}
            )


if __name__ == "__main__":  # pragma: no cover - manual execution
    orchestrator = EvolutionOrchestrator()
    orchestrator.trigger_encoder_training_pipeline()
