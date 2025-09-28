from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

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


@dataclass(frozen=True)
class CuratedSample:
    embedding: list[float]
    target: float
    metadata: dict[str, Any]


class CuratedDatasetBuilder:
    """Normalise curated records into a dataset suitable for adapter training."""

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self._records = records

    def build(self) -> list[CuratedSample]:
        samples: list[CuratedSample] = []
        for index, record in enumerate(self._records):
            sample = normalize_curated_record(record, index)
            if sample is not None:
                samples.append(sample)

        if not samples:
            raise ValueError("No valid curated records supplied for training")

        return samples


def normalize_curated_record(
    record: dict[str, Any], index: int
) -> CuratedSample | None:
    embedding_raw = record.get("embedding") or record.get("vector")
    if not isinstance(embedding_raw, Iterable):
        logger.debug("Skipping curated record %s without embedding", index)
        return None

    embedding: list[float] = []
    for value in embedding_raw:
        try:
            embedding.append(float(value))
        except (TypeError, ValueError):
            logger.debug(
                "Dropping non-numeric embedding value '%s' in record %s",
                value,
                index,
            )
            continue

    if not embedding:
        logger.debug("Skipping curated record %s with empty embedding", index)
        return None

    target_raw = record.get("target", record.get("score"))
    if target_raw is None:
        logger.debug("Skipping curated record %s without target", index)
        return None

    try:
        target = float(target_raw)
    except (TypeError, ValueError):
        logger.debug("Skipping curated record %s with invalid target", index)
        return None

    metadata = {
        "source_id": record.get("source_id"),
        "confidence": record.get("confidence", target),
        "text_preview": record.get("text_preview"),
        "used_fallback_embedding": record.get("used_fallback_embedding", False),
    }
    return CuratedSample(embedding=embedding, target=target, metadata=metadata)


class LinearAdapterTrainer:
    """Train a lightweight linear adapter from curated embedding samples."""

    def __init__(self, *, learning_rate: float, epochs: int, gradient_clip: float) -> None:
        if epochs < 1:
            raise ValueError(f"Number of epochs must be at least 1. Got: {epochs}")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gradient_clip = gradient_clip

    def train(
        self, dataset: Sequence[CuratedSample]
    ) -> tuple[list[float], float, dict[str, Any]]:
        feature_dim = len(dataset[0].embedding)
        weights = [0.0 for _ in range(feature_dim)]
        bias = 0.0
        losses: list[float] = []

        for _ in range(self.epochs):
            total_loss = 0.0
            for sample in dataset:
                prediction = bias + sum(
                    weight * feature for weight, feature in zip(weights, sample.embedding)
                )
                error = prediction - sample.target
                total_loss += error * error

                clipped_error = max(
                    -self.gradient_clip, min(self.gradient_clip, error)
                )

                for i, feature in enumerate(sample.embedding):
                    weights[i] -= self.learning_rate * clipped_error * feature

                bias -= self.learning_rate * clipped_error

            losses.append(total_loss / len(dataset))

        metrics = {
            "loss": losses[-1],
            "initial_loss": losses[0],
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }
        return weights, bias, metrics


class TrainingStatus(str, Enum):
    """Possible outcomes for a training run."""

    SUCCESS = "success"
    FALLBACK = "fallback"


class MNTPTrainer:
    """Execute the MNTP encoder fine-tuning pipeline with graceful fallbacks."""

    def __init__(self, training_config_path: str, output_dir: str) -> None:
        self.config_path = Path(training_config_path)
        self.output_dir = Path(output_dir)
        self.config: dict[str, Any] = {}

    def _load_config(self) -> None:
        try:
            with self.config_path.open() as file:
                self.config = json.load(file)
        except FileNotFoundError as exc:
            logger.error("Training config not found: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON configuration: %s", exc)
            raise

        self._validate_and_apply_defaults()

    def _validate_and_apply_defaults(self) -> None:
        if not isinstance(self.config, dict):  # pragma: no cover - defensive guard
            raise TypeError("Training configuration must be a JSON object")

        defaults: dict[str, Any] = {
            "dataset_name": "wikitext",
            "dataset_config_name": "wikitext-103-raw-v1",
            "dataset_split": "train[:1%]",
            "model_name_or_path": "sshleifer/tiny-gpt2",
            "lora_r": 16,
            "torch_dtype": "float32",
            "attn_implementation": None,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 512,
            "max_steps": 32,
        }
        merged = {**defaults, **self.config}

        required_keys = ("dataset_name", "model_name_or_path")
        for key in required_keys:
            value = merged.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Configuration key '{key}' must be a non-empty string"
                )

        try:
            merged["lora_r"] = int(merged["lora_r"])
            merged["per_device_train_batch_size"] = int(
                merged["per_device_train_batch_size"]
            )
            merged["gradient_accumulation_steps"] = int(
                merged["gradient_accumulation_steps"]
            )
            merged["max_seq_length"] = int(merged["max_seq_length"])
            merged["max_steps"] = int(merged["max_steps"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Numeric configuration options must be castable to int"
            ) from exc

        self.config = merged

    def train(
        self, curated_records: Sequence[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Run the MNTP pipeline and return a structured training summary.

        When ``curated_records`` are provided, a lightweight masked-next-token
        style adapter is trained directly from the supplied embeddings. This
        keeps the trainer usable in environments without the optional heavy
        dependencies while still providing a deterministic update path for the
        self-training engine.
        """

        self._load_config()
        logger.info(
            "MNTP training started", extra={"config_path": str(self.config_path)}
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        if curated_records:
            summary = self._run_curated_training(curated_records)
        elif not self._deps_available():
            logger.warning(
                "Training dependencies missing; falling back to deterministic adapter"
            )
            summary = self._materialise_fallback_adapter(reason="missing_dependencies")
        else:
            try:
                summary = self._run_peft_training()
            except Exception as exc:  # pragma: no cover - unexpected ML errors
                logger.error("Training failed: %s", exc, exc_info=True)
                summary = self._materialise_fallback_adapter(
                    reason="training_failed", details=str(exc)
                )

        self._write_summary(summary)
        return summary

    def _run_curated_training(
        self, curated_records: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        dataset = CuratedDatasetBuilder(curated_records).build()
        feature_dim = max(len(sample.embedding) for sample in dataset)

        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        trainer = LinearAdapterTrainer(
            learning_rate=float(self.config.get("curated_learning_rate", 0.05)),
            epochs=int(self.config.get("curated_epochs", 15)),
            gradient_clip=float(self.config.get("curated_gradient_clip", 1.0)),
        )
        weights, bias, metrics = trainer.train(dataset)
        artifact_path = adapter_dir / "curated_linear_adapter.json"
        payload = {
            "schema_version": 1,
            "weights": weights,
            "bias": bias,
            "feature_dimension": feature_dim,
            "metrics": metrics,
            "records": [sample.metadata for sample in dataset],
        }
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        logger.info(
            "Curated MNTP training complete",
            extra={
                "records": len(dataset),
                "artifact_path": str(artifact_path),
                "loss": metrics.get("loss"),
            },
        )

        return {
            "status": TrainingStatus.SUCCESS.value,
            "mode": "curated_linear_adapter",
            "artifacts": {"adapter": str(adapter_dir), "weights": str(artifact_path)},
            "metrics": metrics
            | {
                "training_examples": len(dataset),
                "feature_dimension": feature_dim,
            },
        }

    def _save_config(self) -> None:
        try:
            path = self.output_dir / "training_config.json"
            path.write_text(json.dumps(self.config, indent=2, sort_keys=True))
        except OSError as exc:  # pragma: no cover
            logger.error("Failed to write training config: %s", exc)
            raise

    def _deps_available(self) -> bool:
        return bool(
            torch and load_dataset and LLM2Vec and hasattr(LLM2Vec, "from_pretrained")
        )

    def _materialise_fallback_adapter(
        self, *, reason: str, details: str | None = None
    ) -> dict[str, Any]:
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        weights_payload = self._derive_fallback_weights()
        weights_path = adapter_dir / "fallback_adapter.json"
        weights_path.write_text(json.dumps(weights_payload, indent=2, sort_keys=True))

        logger.info(
            "Fallback adapter generated",
            extra={"reason": reason, "weights_path": str(weights_path)},
        )

        return {
            "status": TrainingStatus.FALLBACK.value,
            "reason": reason,
            "details": details,
            "artifacts": {"adapter": str(adapter_dir), "weights": str(weights_path)},
            "metrics": {
                "training_examples": 0,
                "per_device_train_batch_size": self.config[
                    "per_device_train_batch_size"
                ],
                "gradient_accumulation_steps": self.config[
                    "gradient_accumulation_steps"
                ],
            },
        }

    def _write_summary(self, summary: dict[str, Any]) -> None:
        summary_path = self.output_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    def _derive_fallback_weights(self) -> dict[str, Any]:
        fingerprint: dict[str, Any] = {
            "keys": sorted(self.config.keys()),
            "string_lengths": {},
            "numeric": {},
            "boolean": {},
        }

        for key, value in self.config.items():
            if isinstance(value, bool):
                fingerprint["boolean"][key] = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                fingerprint["numeric"][key] = float(value)
            elif isinstance(value, str):
                fingerprint["string_lengths"][key] = len(value)

        serialized = json.dumps(
            fingerprint, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(serialized).digest()
        rows = max(4, min(64, int(self.config.get("lora_r", 16))))
        cols = max(8, min(128, int(self.config.get("max_seq_length", 512)) // 4))

        matrix: list[list[float]] = []
        idx = 0
        for _ in range(rows):
            row: list[float] = []
            for _ in range(cols):
                byte = digest[idx % len(digest)]
                value = round(((byte / 255.0) * 2) - 1, 6)
                row.append(value)
                idx += 1
            matrix.append(row)

        checksum = hashlib.sha256(
            json.dumps(matrix, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return {
            "rows": rows,
            "cols": cols,
            "checksum": checksum,
            "schema_version": 1,
            "matrix": matrix,
        }

    def _run_peft_training(self) -> dict[str, Any]:
        dataset = self._load_dataset()
        model_name = self._resolve_model_name()
        model = self._initialise_model(model_name)
        metrics = self._execute_training(model, dataset)
        artifact_dir = self._persist_model(model)

        logger.info(
            "Model training finished",
            extra={"artifact_dir": str(artifact_dir), "model_name": model_name},
        )
        return {
            "status": TrainingStatus.SUCCESS.value,
            "model_name": model_name,
            "dataset_name": self.config["dataset_name"],
            "artifacts": {"adapter": str(artifact_dir)},
            "metrics": metrics,
        }

    def _load_dataset(self) -> Any:
        if load_dataset is None:
            raise RuntimeError("datasets library unavailable")

        try:
            dataset = load_dataset(
                self.config["dataset_name"],
                self.config.get("dataset_config_name"),
                split=self.config.get("dataset_split", "train[:1%]"),
            )
        except Exception as exc:  # pragma: no cover - network/IO errors
            raise RuntimeError("Failed to load dataset") from exc
        return dataset

    def _resolve_model_name(self) -> str:
        model_name = self.config["model_name_or_path"].strip()
        if model_name.lower().startswith("mistralai/mistral-7b"):
            logger.warning(
                "Large model specified; defaulting to lightweight reference model",
                extra={"requested_model": model_name},
            )
            return "sshleifer/tiny-gpt2"
        return model_name or "sshleifer/tiny-gpt2"

    def _initialise_model(self, model_name: str) -> Any:
        if LLM2Vec is None:
            raise RuntimeError("LLM2Vec is unavailable")

        from transformers import AutoTokenizer  # local import to avoid hard dependency

        if not getattr(AutoTokenizer, "_mon_gars_cleanup_patch", False):
            original_from_pretrained = AutoTokenizer.from_pretrained

            def _from_pretrained_with_cleanup(*args: Any, **kwargs: Any):
                kwargs.setdefault("clean_up_tokenization_spaces", True)
                return original_from_pretrained(*args, **kwargs)

            AutoTokenizer.from_pretrained = _from_pretrained_with_cleanup  # type: ignore[assignment]
            setattr(AutoTokenizer, "_mon_gars_cleanup_patch", True)

        dtype_name = str(self.config.get("torch_dtype", "float32"))
        torch_dtype = getattr(torch, dtype_name, None) if torch else None
        if torch_dtype is None and torch is not None:
            logger.warning(
                "Unsupported torch dtype '%s'; defaulting to float32", dtype_name
            )
            torch_dtype = torch.float32

        try:
            model = LLM2Vec.from_pretrained(
                base_model_name_or_path=model_name,
                enable_bidirectional=True,
                pooling_mode="mean",
                torch_dtype=torch_dtype,
                attn_implementation=self.config.get("attn_implementation"),
            )
        except Exception as exc:  # pragma: no cover - depends on external library
            raise RuntimeError("Failed to initialise LLM2Vec") from exc
        return model

    def _execute_training(self, model: Any, dataset: Any) -> dict[str, Any]:
        if not hasattr(model, "train"):
            raise RuntimeError("LLM2Vec model does not expose a train method")

        lora_rank = int(self.config.get("lora_r", 16))
        max_steps = int(self.config.get("max_steps", 32))

        try:
            result = model.train(
                dataset=dataset,
                lora_r=lora_rank,
                max_steps=max_steps,
                per_device_train_batch_size=self.config.get(
                    "per_device_train_batch_size"
                ),
                gradient_accumulation_steps=self.config.get(
                    "gradient_accumulation_steps"
                ),
            )
        except TypeError:
            result = model.train(dataset=dataset, lora_r=lora_rank)

        metrics = self._extract_metrics(result, dataset)
        metrics["max_steps"] = max_steps
        metrics["lora_r"] = lora_rank
        return metrics

    def _extract_metrics(self, result: Any, dataset: Any) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "training_examples": self._estimate_dataset_size(dataset),
            "per_device_train_batch_size": self.config.get(
                "per_device_train_batch_size"
            ),
            "gradient_accumulation_steps": self.config.get(
                "gradient_accumulation_steps"
            ),
        }

        if isinstance(result, dict):
            metrics |= {
                key: value
                for key, value in result.items()
                if isinstance(value, (int, float))
            }

        return metrics

    def _estimate_dataset_size(self, dataset: Any) -> int:
        try:
            return len(dataset)  # type: ignore[arg-type]
        except (TypeError, ValueError):  # pragma: no cover - iterable datasets
            batch = int(self.config.get("per_device_train_batch_size", 1))
            steps = int(self.config.get("max_steps", 1))
            return (
                batch * steps * int(self.config.get("gradient_accumulation_steps", 1))
            )

    def _persist_model(self, model: Any) -> Path:
        if not hasattr(model, "save_peft"):
            raise RuntimeError("LLM2Vec model does not expose save_peft")

        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        try:
            model.save_peft(str(adapter_dir))
        except TypeError:
            model.save_peft(adapter_dir)
        except Exception as exc:  # pragma: no cover - filesystem issues
            raise RuntimeError("Failed to persist PEFT adapter") from exc

        return adapter_dir
