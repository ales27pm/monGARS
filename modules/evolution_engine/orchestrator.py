from __future__ import annotations

from pathlib import Path

from modules.neurons.training.mntp_trainer import MNTPTrainer


class EvolutionOrchestrator:
    """Trigger training pipelines for new encoders."""

    def __init__(self, model_registry_path: str = "models/encoders/") -> None:
        self.model_registry_path = Path(model_registry_path)

    def trigger_encoder_training_pipeline(self) -> str:
        """Launch a dummy two-step training pipeline."""
        print("Début du pipeline d'entraînement d'un nouvel encodeur")
        config_path = "configs/training/mntp_mistral_config.json"
        output_path = self.model_registry_path / "temp-mistral-mntp-step"
        trainer = MNTPTrainer(
            training_config_path=config_path, output_dir=str(output_path)
        )
        trainer.train()
        print(f"Pipeline terminé. Nouvel encodeur disponible à: {output_path}")
        return str(output_path)


if __name__ == "__main__":  # pragma: no cover - manual execution
    orchestrator = EvolutionOrchestrator()
    orchestrator.trigger_encoder_training_pipeline()
