import json
from pathlib import Path
from typing import Any, Dict


class MNTPTrainer:
    """Simplified trainer that simulates MNTP training."""

    def __init__(self, training_config_path: str, output_dir: str) -> None:
        self.config_path = Path(training_config_path)
        self.output_dir = Path(output_dir)
        self.config: Dict[str, Any] = {}

    def _load_config(self) -> None:
        with self.config_path.open() as f:
            self.config = json.load(f)

    def train(self) -> None:
        """Run a dummy training loop and save placeholder weights."""
        self._load_config()
        print("Entraînement MNTP lancé")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Save config to simulate training output
        with (self.output_dir / "training_config.json").open("w") as f:
            json.dump(self.config, f, indent=2)
        # Create a dummy model file
        (self.output_dir / "adapter_model.bin").write_text("placeholder")
        print(f"Modèle sauvegardé dans {self.output_dir}")
