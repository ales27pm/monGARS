from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from modules.neurons.training.mntp_trainer import MNTPTrainer

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
        logger.info(
            "Pipeline finished",
            extra={
                "encoder_path": str(unique_dir),
                "status": summary.get("status"),
                "artifacts": summary.get("artifacts", {}),
            },
        )
        return str(unique_dir)


if __name__ == "__main__":  # pragma: no cover - manual execution
    orchestrator = EvolutionOrchestrator()
    orchestrator.trigger_encoder_training_pipeline()
