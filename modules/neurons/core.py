from __future__ import annotations

from typing import List, Optional


class NeuronManager:
    """Manage loading and switching of LLM2Vec encoders."""

    def __init__(
        self, base_model_path: str, default_encoder_path: Optional[str] = None
    ) -> None:
        self.base_model_path = base_model_path
        self.encoder_path = default_encoder_path
        self.model = None
        if self.encoder_path:
            self._load_encoder()

    def _load_encoder(self) -> None:
        """Load an encoder if available. Heavy dependencies are imported lazily."""
        print("Chargement de l'encodeur LLM2Vec.")
        print(f"  - Modèle de base: {self.base_model_path}")
        print(f"  - Poids de l'adaptateur PEFT: {self.encoder_path}")
        try:
            from llm2vec import LLM2Vec  # type: ignore

            self.model = LLM2Vec.from_pretrained(
                base_model_name_or_path=self.base_model_path,
                peft_model_name_or_path=self.encoder_path,
                enable_bidirectional=True,
                pooling_mode="mean",
                torch_dtype="bfloat16",
                device_map="cpu",
            )
            print("Encodeur chargé avec succès.")
        except (
            OSError,
            ImportError,
        ) as exc:  # pragma: no cover - expected model loading errors
            print(f"Impossible de charger l'encodeur: {exc}")
            self.model = None

    def switch_encoder(self, new_encoder_path: str) -> None:
        """Change the current encoder."""
        self.encoder_path = new_encoder_path
        self._load_encoder()

    def encode(self, texts: List[str], instruction: str = ""):
        """Encode a list of texts using the current model."""
        if not self.model:
            raise RuntimeError("Encoder not loaded")

        formatted_texts = [[instruction, text] for text in texts]
        disable_cm = getattr(self.model, "disable_gradient", None)
        if disable_cm:
            with disable_cm():
                return self.model.encode(
                    formatted_texts, batch_size=8, show_progress_bar=False
                )
        return self.model.encode(formatted_texts, batch_size=8, show_progress_bar=False)
