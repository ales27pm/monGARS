from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from modules.neurons.training.mntp_trainer import MNTPTrainer

logger = logging.getLogger(__name__)


class EvolutionOrchestrator:
    """Trigger training pipelines for new encoders."""

    def __init__(
        self,
        model_registry_path: str = "models/encoders/",
        config_path: str | None = None,
    ) -> None:
        self.model_registry_path = Path(model_registry_path)
        self.config_path = (
            Path(config_path)
            if config_path
            else Path("configs/training/mntp_mistral_config.json")
        )

    def trigger_encoder_training_pipeline(self) -> str:
        """Launch a dummy two-step training pipeline."""
        logger.info("Starting training pipeline for a new encoder")
        unique_dir = self.model_registry_path / f"temp-mistral-mntp-step-{uuid4()}"
        trainer = MNTPTrainer(
            training_config_path=str(self.config_path), output_dir=str(unique_dir)
        )
        try:
            trainer.train()
        except Exception as exc:  # pragma: no cover - unexpected training error
            logger.error("Training failed: %s", exc)
            raise
        logger.info("Pipeline finished. New encoder at: %s", unique_dir)
        return str(unique_dir)


if __name__ == "__main__":  # pragma: no cover - manual execution
    orchestrator = EvolutionOrchestrator()
    orchestrator.trigger_encoder_training_pipeline()
