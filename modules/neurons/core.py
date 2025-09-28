from __future__ import annotations

import hashlib
import json
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


def _get_torch_module() -> Any | None:
    """Import ``torch`` lazily to avoid mandatory dependency at module load."""

    try:  # pragma: no cover - executed only when torch is installed
        import torch  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - torch missing in lightweight envs
        return None
    return torch


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
        encode_options: dict[str, Any] | None = None,
    ) -> None:
        if fallback_dimensions <= 0:
            raise ValueError("fallback_dimensions must be a positive integer")
        if fallback_cache_size <= 0:
            raise ValueError("fallback_cache_size must be a positive integer")

        self.base_model_path = base_model_path
        self.encoder_path = default_encoder_path
        self._fallback_dimensions = fallback_dimensions
        self._fallback_cache_size = fallback_cache_size
        self._fallback_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._llm2vec_factory = llm2vec_factory
        self._llm2vec_options = llm2vec_options or {}
        self._encode_options = self._normalise_encode_options(encode_options)
        self.model: Any | None = None
        self._load_attempted = False

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
        if not hasattr(LLM2Vec, "from_pretrained"):
            logger.warning(
                "LLM2Vec missing from_pretrained; falling back to placeholder"
            )
            return None

        overrides = {
            key: value
            for key, value in self._llm2vec_options.items()
            if value is not None
        }
        if "torch_dtype" in overrides:
            torch_dtype_value = overrides.pop("torch_dtype")
        else:
            torch_dtype_value = "bfloat16"

        options: dict[str, Any] = {
            "base_model_name_or_path": self.base_model_path,
            "enable_bidirectional": True,
            "pooling_mode": "mean",
            "device_map": overrides.pop("device_map", "cpu"),
        }
        resolved_dtype = self._resolve_torch_dtype(torch_dtype_value)
        if resolved_dtype is not None:
            options["torch_dtype"] = resolved_dtype
        if self.encoder_path:
            options["peft_model_name_or_path"] = self.encoder_path
        options |= overrides
        return LLM2Vec.from_pretrained(**options)

    def _normalise_encode_options(
        self, options: dict[str, Any] | None
    ) -> dict[str, Any]:
        defaults: dict[str, Any] = {"batch_size": 8, "show_progress_bar": False}
        if not options:
            return defaults

        merged = dict(defaults)
        for key, value in options.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        return self._validate_encode_options(merged)

    def _validate_encode_options(self, options: dict[str, Any]) -> dict[str, Any]:
        validated = dict(options)

        if "batch_size" in validated:
            batch_size = validated["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")

        for boolean_key in (
            "show_progress_bar",
            "convert_to_numpy",
            "convert_to_tensor",
        ):
            if boolean_key in validated and not isinstance(
                validated[boolean_key], bool
            ):
                raise TypeError(f"{boolean_key} must be a boolean when provided")

        return validated

    def _prepare_encode_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        options = dict(self._encode_options)
        for key, value in overrides.items():
            if value is None:
                options.pop(key, None)
            else:
                options[key] = value
        return self._validate_encode_options(options)

    def _prepare_prompt_pairs(
        self,
        texts: Sequence[str] | Sequence[Sequence[str]],
        instruction: str | Sequence[str] | None,
    ) -> list[tuple[str, str]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single string")

        text_list = list(texts)
        if not text_list:
            return []

        first = text_list[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            if instruction not in ("", None):
                raise ValueError(
                    "instruction must be omitted when providing pre-formatted prompts"
                )
            pairs: list[tuple[str, str]] = []
            for item in text_list:
                if (
                    not isinstance(item, Sequence)
                    or isinstance(item, (str, bytes))
                    or len(item) != 2
                ):
                    raise TypeError(
                        "pre-formatted prompts must be sequences of two strings"
                    )
                inst, text = item  # type: ignore[misc]
                if not isinstance(inst, str) or not isinstance(text, str):
                    raise TypeError("instruction and text must both be strings")
                pairs.append((inst, text))
            return pairs

        instructions: list[str]
        if instruction is None:
            instructions = [""] * len(text_list)
        elif isinstance(instruction, str):
            instructions = [instruction] * len(text_list)
        else:
            instructions = list(instruction)
            if len(instructions) != len(text_list):
                raise ValueError(
                    "instruction list must match the number of texts provided"
                )
            for inst in instructions:
                if not isinstance(inst, str):
                    raise TypeError("all instructions must be strings")

        pairs = []
        for inst, text in zip(instructions, text_list, strict=True):
            if not isinstance(text, str):
                raise TypeError("texts must be strings")
            pairs.append((inst, text))
        return pairs

    def _resolve_torch_dtype(self, dtype: Any) -> Any | None:
        """Translate ``dtype`` to a torch dtype instance when possible."""

        if dtype is None:
            return None
        if not isinstance(dtype, str):
            return dtype

        normalized = dtype.strip()
        if not normalized:
            return None
        if normalized.lower() == "auto":
            return "auto"

        torch_module = _get_torch_module()
        if torch_module is None:
            logger.warning(
                "Torch is unavailable; ignoring torch_dtype override '%s'", dtype
            )
            return None

        attribute_name = normalized.split(".")[-1]
        resolved = getattr(torch_module, attribute_name, None)
        if resolved is None:
            logger.warning(
                "Unknown torch dtype '%s'; falling back to library defaults", dtype
            )
        return resolved

    def switch_encoder(self, new_encoder_path: str) -> None:
        """Change the current encoder and reload it."""

        self.encoder_path = new_encoder_path
        self._load_attempted = False
        self._load_encoder()

    def reload(self) -> None:
        """Force a reload of the current encoder configuration."""

        self._load_encoder()

    def set_encode_options(self, **encode_options: Any) -> None:
        """Update default encode keyword arguments used for future requests."""

        if not encode_options:
            return

        merged_options = dict(self._encode_options)
        for key, value in encode_options.items():
            if value is None:
                merged_options.pop(key, None)
            else:
                merged_options[key] = value

        self._encode_options = self._validate_encode_options(merged_options)

    def encode(
        self,
        texts: Sequence[str] | Sequence[Sequence[str]],
        instruction: str | Sequence[str] | None = "",
        **encode_kwargs: Any,
    ) -> list[list[float]]:
        """Encode texts with LLM2Vec, supporting per-text instructions and overrides."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single string")

        prompts = self._prepare_prompt_pairs(texts, instruction)
        if not prompts:
            return []

        options = self._prepare_encode_kwargs(encode_kwargs)

        if not self._load_attempted:
            self._load_encoder()

        if not self.model:
            logger.debug("Using fallback embeddings for %d texts", len(prompts))
            return [self._fallback_vector(inst, text) for inst, text in prompts]

        formatted_texts = [[inst, text] for inst, text in prompts]
        disable_cm = getattr(self.model, "disable_gradient", None)
        try:
            if hasattr(self.model, "last_kwargs"):
                try:
                    setattr(self.model, "last_kwargs", dict(options))
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Unable to record encode options on model: %s", exc)
            if callable(disable_cm):
                with disable_cm():
                    encoded = self.model.encode(formatted_texts, **options)
            else:
                encoded = self.model.encode(formatted_texts, **options)
        except Exception as exc:  # pragma: no cover - inference failures are rare
            logger.warning(
                "LLM2Vec encoding failed; falling back to deterministic vectors: %s",
                exc,
                exc_info=True,
            )
            return [self._fallback_vector(inst, text) for inst, text in prompts]

        try:
            return self._normalise_embeddings(encoded, expected=len(prompts))
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "LLM2Vec returned invalid embeddings; using fallback: %s", exc
            )
            return [self._fallback_vector(inst, text) for inst, text in prompts]

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
        cache_key = (instruction, text)
        cached = self._fallback_cache.get(cache_key)
        if cached is not None:
            self._fallback_cache.move_to_end(cache_key)
            return list(cached)

        serialized_key = json.dumps(cache_key, ensure_ascii=False)
        digest = hashlib.sha256(serialized_key.encode("utf-8")).digest()
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
            logger.warning(
                "LLM2Vec returned %d embeddings for %d inputs", len(vectors), expected
            )
            raise ValueError(
                f"LLM2Vec returned {len(vectors)} embeddings for {expected} inputs"
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
