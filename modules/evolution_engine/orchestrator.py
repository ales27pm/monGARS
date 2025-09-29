from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from modules.neurons.registry import update_manifest
from modules.neurons.training.mntp_trainer import MNTPTrainer, TrainingStatus

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
    ) -> None:
        self.model_registry_path = Path(model_registry_path)
        self.config_path = (
            Path(config_path)
            if config_path
            else Path("configs/training/mntp_mistral_config.json")
        )
        self._trainer_cls: type[TrainerProtocol] = trainer_cls

    def trigger_encoder_training_pipeline(self) -> str:
        """Launch the MNTP pipeline and return the produced artifact directory."""

        logger.info("Starting training pipeline for a new encoder")
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        unique_dir = self.model_registry_path / f"temp-mistral-mntp-step-{uuid4()}"
        trainer = self._trainer_cls(
            training_config_path=str(self.config_path),
            output_dir=str(unique_dir),
        )
        try:
            summary = trainer.train()
        except Exception as exc:  # pragma: no cover - unexpected training error
            logger.error("Training failed: %s", exc, exc_info=True)
            raise

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


if __name__ == "__main__":  # pragma: no cover - manual execution
    orchestrator = EvolutionOrchestrator()
    orchestrator.trigger_encoder_training_pipeline()
