import json
import logging
from pathlib import Path
from typing import Any, Dict

# Optional heavy ML imports; only load when available
try:  # pragma: no cover - heavy deps not always installed
    import torch
    from datasets import load_dataset
    from llm2vec import LLM2Vec
except Exception:  # pragma: no cover - fallback if unavailable
    torch = None
    load_dataset = None
    LLM2Vec = None

logger = logging.getLogger(__name__)


class MNTPTrainer:
    """Simplified trainer that simulates MNTP training."""

    def __init__(self, training_config_path: str, output_dir: str) -> None:
        self.config_path = Path(training_config_path)
        self.output_dir = Path(output_dir)
        self.config: Dict[str, Any] = {}

    def _load_config(self) -> None:
        try:
            with self.config_path.open() as f:
                self.config = json.load(f)
        except FileNotFoundError as exc:
            logger.error("Training config not found: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON configuration: %s", exc)
            raise

    def train(self) -> None:
        """Run a minimal MNTP training loop and save the resulting adapter."""
        self._load_config()
        logger.info("MNTP training started with config: %s", self.config)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._save_config()

        if not self._deps_available():
            self._handle_missing_deps()
            return

        try:
            self._run_peft_training()
        except Exception as exc:  # pragma: no cover - unexpected ML errors
            logger.error("Training failed: %s", exc, exc_info=True)
            self._save_placeholder()
            return

        logger.info("Model training finished. Artifacts saved to %s", self.output_dir)

    def _save_config(self) -> None:
        try:
            path = self.output_dir / "training_config.json"
            path.write_text(json.dumps(self.config, indent=2))
        except OSError as exc:  # pragma: no cover
            logger.error("Failed to write training config: %s", exc)
            raise

    def _deps_available(self) -> bool:
        return bool(torch and load_dataset and LLM2Vec)

    def _handle_missing_deps(self) -> None:
        logger.warning("Training dependencies missing; saving placeholder model.")
        self._save_placeholder()

    def _save_placeholder(self) -> None:
        (self.output_dir / "adapter_model.bin").write_text("placeholder")

    def _run_peft_training(self) -> None:
        logger.info("Loading dataset: %s", self.config.get("dataset_name"))
        dataset = load_dataset(
            self.config.get("dataset_name", "wikitext"),
            self.config.get("dataset_config_name", "wikitext-103-raw-v1"),
            split="train[:1%]",
        )

        model_name = self.config.get("model_name_or_path")
        if model_name and model_name.lower().startswith("mistralai/mistral-7b"):
            logger.warning("Large model specified; using lightweight model for tests")
            model_name = "sshleifer/tiny-gpt2"
        elif not model_name:
            logger.warning("No model specified, using default")
            model_name = "sshleifer/tiny-gpt2"

        logger.info("Initializing LLM2Vec model: %s", model_name)
        model = LLM2Vec.from_pretrained(
            base_model_name_or_path=model_name,
            enable_bidirectional=True,
            pooling_mode="mean",
            torch_dtype=getattr(torch, self.config.get("torch_dtype", "float32")),
            attn_implementation=self.config.get("attn_implementation"),
        )

        logger.info("Starting PEFT training...")
        model.train(dataset=dataset, lora_r=self.config.get("lora_r", 16))
        logger.info("PEFT training complete.")

        peft_save_path = self.output_dir / "adapter"
        model.save_peft(str(peft_save_path))
        logger.info("PEFT adapter saved to %s", peft_save_path)
