from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - heavy dependency is optional
    from llm2vec import LLM2Vec
except Exception:  # pragma: no cover - library not available in tests
    LLM2Vec = None  # type: ignore[assignment]


class NeuronManager:
    """Manage loading and switching of LLM2Vec encoders with graceful fallbacks."""

    def __init__(
        self,
        base_model_path: str,
        default_encoder_path: str | None = None,
        *,
        fallback_dimensions: int = 384,
        fallback_cache_size: int = 256,
        llm2vec_factory: Callable[[str, str | None], Any] | None = None,
        llm2vec_options: dict[str, Any] | None = None,
    ) -> None:
        if fallback_dimensions <= 0:
            raise ValueError("fallback_dimensions must be a positive integer")
        if fallback_cache_size <= 0:
            raise ValueError("fallback_cache_size must be a positive integer")

        self.base_model_path = base_model_path
        self.encoder_path = default_encoder_path
        self._fallback_dimensions = fallback_dimensions
        self._fallback_cache_size = fallback_cache_size
        self._fallback_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._llm2vec_factory = llm2vec_factory
        self._llm2vec_options = llm2vec_options or {}
        self.model: Any | None = None
        self._load_attempted = False

        if self.encoder_path:
            self._load_encoder()

    @property
    def is_ready(self) -> bool:
        """Return ``True`` when an encoder is loaded and ready for use."""

        return self.model is not None

    def unload(self) -> None:
        """Release the currently loaded model if it exposes a ``close`` method."""

        self._dispose_model()
        self._load_attempted = False

    def _load_encoder(self) -> None:
        """Load an encoder, handling optional dependencies gracefully."""

        logger.info(
            "Loading LLM2Vec encoder",
            extra={
                "base_model_path": self.base_model_path,
                "encoder_path": self.encoder_path,
            },
        )
        self._dispose_model()
        self._load_attempted = True
        if not self.encoder_path:
            logger.debug("No encoder path provided; skipping load")
            return

        try:
            self.model = self._build_model()
        except Exception as exc:  # pragma: no cover - unexpected loader error
            logger.warning("Failed to load LLM2Vec encoder: %s", exc, exc_info=True)
            self.model = None
        else:
            if self.model is None:
                logger.warning("LLM2Vec unavailable; using fallback embeddings")
            else:
                logger.info("LLM2Vec encoder ready")

    def _build_model(self) -> Any | None:
        if self._llm2vec_factory is not None:
            return self._llm2vec_factory(self.base_model_path, self.encoder_path)
        if LLM2Vec is None:
            return None
        options: dict[str, Any] = {
            "base_model_name_or_path": self.base_model_path,
            "enable_bidirectional": True,
            "pooling_mode": "mean",
            "torch_dtype": self._llm2vec_options.get("torch_dtype", "bfloat16"),
            "device_map": self._llm2vec_options.get("device_map", "cpu"),
        }
        if self.encoder_path:
            options["peft_model_name_or_path"] = self.encoder_path
        options.update(self._llm2vec_options)
        return LLM2Vec.from_pretrained(**options)

    def switch_encoder(self, new_encoder_path: str) -> None:
        """Change the current encoder and reload it."""

        self.encoder_path = new_encoder_path
        self._load_attempted = False
        self._load_encoder()

    def encode(self, texts: Sequence[str], instruction: str = "") -> list[list[float]]:
        """Encode a list of texts using the current model or a deterministic fallback."""

        if not texts:
            return []

        if self.encoder_path and not self._load_attempted:
            self._load_encoder()

        if not self.model:
            logger.debug("Using fallback embeddings for %d texts", len(texts))
            return [self._fallback_vector(instruction, text) for text in texts]

        formatted_texts = [[instruction, text] for text in texts]
        disable_cm = getattr(self.model, "disable_gradient", None)
        try:
            if callable(disable_cm):
                with disable_cm():
                    encoded = self.model.encode(
                        formatted_texts, batch_size=8, show_progress_bar=False
                    )
            else:
                encoded = self.model.encode(
                    formatted_texts, batch_size=8, show_progress_bar=False
                )
        except Exception as exc:  # pragma: no cover - inference failures are rare
            logger.warning(
                "LLM2Vec encoding failed; falling back to deterministic vectors: %s",
                exc,
                exc_info=True,
            )
            return [self._fallback_vector(instruction, text) for text in texts]

        try:
            return self._normalise_embeddings(encoded, expected=len(texts))
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "LLM2Vec returned invalid embeddings; using fallback: %s", exc
            )
            return [self._fallback_vector(instruction, text) for text in texts]

    def _dispose_model(self) -> None:
        if not self.model:
            return
        close = getattr(self.model, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Error while closing LLM2Vec model: %s", exc)
        self.model = None

    def _fallback_vector(self, instruction: str, text: str) -> list[float]:
        cache_key = f"{instruction}\u241f{text}"
        cached = self._fallback_cache.get(cache_key)
        if cached is not None:
            self._fallback_cache.move_to_end(cache_key)
            return list(cached)

        digest = hashlib.sha256(cache_key.encode("utf-8")).digest()
        required = self._fallback_dimensions
        repeated = (digest * ((required // len(digest)) + 1))[:required]
        vector = [(byte / 255.0) * 2 - 1 for byte in repeated]
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude > 0:
            vector = [value / magnitude for value in vector]
        else:  # pragma: no cover - impossible for SHA-256 digest
            vector = [0.0] * required

        self._fallback_cache[cache_key] = list(vector)
        if len(self._fallback_cache) > self._fallback_cache_size:
            self._fallback_cache.popitem(last=False)
        return list(vector)

    def _normalise_embeddings(
        self, encoded: Any, *, expected: int
    ) -> list[list[float]]:
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()

        if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
            raise TypeError("Model returned non-iterable embedding")

        if not encoded:
            raise ValueError("Model returned empty embeddings")

        first = encoded[0]
        if hasattr(first, "tolist"):
            first = first.tolist()

        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            vectors = [self._ensure_float_list(row) for row in encoded]
        else:
            vectors = [self._ensure_float_list(encoded)]

        if len(vectors) != expected:
            logger.debug(
                "LLM2Vec returned %d embeddings for %d inputs", len(vectors), expected
            )

        return vectors

    @staticmethod
    def _ensure_float_list(values: Any) -> list[float]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("Embedding row must be an iterable of numbers")
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise TypeError("Embedding row contains non-numeric values") from exc
