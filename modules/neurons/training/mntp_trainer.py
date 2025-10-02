from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

# Optional heavy ML imports; only load when available
try:  # pragma: no cover - heavy deps not always installed
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup
except Exception:  # pragma: no cover - fallback if unavailable
    torch = None
    load_dataset = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    clip_grad_norm_ = None
    DataLoader = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    get_linear_schedule_with_warmup = None

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

    def __init__(
        self, *, learning_rate: float, epochs: int, gradient_clip: float
    ) -> None:
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
                    weight * feature
                    for weight, feature in zip(weights, sample.embedding)
                )
                error = prediction - sample.target
                total_loss += error * error

                clipped_error = max(-self.gradient_clip, min(self.gradient_clip, error))

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


class MaskedNextTokenDataset:
    """Generate masked-next-token training examples from raw text records."""

    def __init__(
        self,
        dataset: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        max_seq_length: int,
        mask_token_id: int,
        mlm_probability: float,
        max_masks_per_sample: int,
        seed: int,
        text_field: str | None = None,
    ) -> None:
        self._examples: list[dict[str, Any]] = []
        self._max_seq_length = max_seq_length
        self._mask_token_id = mask_token_id

        rng = random.Random(seed)
        for index, record in enumerate(dataset):
            text = self._extract_text(record, text_field)
            if not text:
                continue

            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=True,
            )
            input_ids: list[int] = list(tokenized.get("input_ids", []))
            if len(input_ids) < 2:
                continue

            positions = self._select_positions(
                input_ids, rng, mlm_probability, max_masks_per_sample
            )
            for position in positions:
                context = input_ids[:position]
                if not context:
                    continue
                label = input_ids[position]
                trimmed_context = self._trim_context(context)
                input_ids_with_mask = trimmed_context + [mask_token_id]
                attention_mask = [1] * len(input_ids_with_mask)
                self._examples.append(
                    {
                        "input_ids": input_ids_with_mask,
                        "attention_mask": attention_mask,
                        "label": label,
                        "source_index": index,
                    }
                )

        if not self._examples:
            raise ValueError(
                "Unable to construct masked-next-token dataset from provided records"
            )

    @staticmethod
    def _extract_text(record: dict[str, Any], field: str | None) -> str | None:
        if field:
            value = record.get(field)
            if isinstance(value, str):
                return value
        # Fall back to first string value
        for value in record.values():
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _select_positions(
        input_ids: Sequence[int],
        rng: random.Random,
        probability: float,
        max_masks_per_sample: int,
    ) -> list[int]:
        limit = max(1, max_masks_per_sample)
        positions: list[int] = []
        for idx in range(1, len(input_ids)):
            if rng.random() <= probability:
                positions.append(idx)
                if len(positions) >= limit:
                    break
        if not positions:
            # Always include at least one deterministic position for stability
            fallback_position = min(len(input_ids) - 1, max(1, len(input_ids) // 2))
            positions.append(fallback_position)
        return positions

    def _trim_context(self, context: Sequence[int]) -> list[int]:
        max_context = max(1, self._max_seq_length - 1)
        if len(context) <= max_context:
            return list(context)
        return list(context[-max_context:])

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._examples[index]


class MaskedNextTokenCollator:
    """Pad variable-length MNTP samples into uniform mini-batches."""

    def __init__(self, pad_token_id: int) -> None:
        self._pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for MNTP collation")

        batch_size = len(batch)
        max_length = max(len(item["input_ids"]) for item in batch)

        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=self._pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)

        for row, item in enumerate(batch):
            sequence = torch.tensor(item["input_ids"], dtype=torch.long)
            mask = torch.tensor(item["attention_mask"], dtype=torch.long)
            length = sequence.size(0)
            input_ids[row, :length] = sequence
            attention_mask[row, :length] = mask
            labels[row] = int(item["label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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
        checksum = self._compute_file_checksum(artifact_path)

        logger.info(
            "Curated MNTP training complete",
            extra={
                "records": len(dataset),
                "artifact_path": str(artifact_path),
                "loss": metrics.get("loss"),
                "checksum": checksum,
            },
        )

        return {
            "status": TrainingStatus.SUCCESS.value,
            "mode": "curated_linear_adapter",
            "version": checksum,
            "artifacts": {
                "adapter": str(adapter_dir),
                "weights": str(artifact_path),
                "weights_checksum": checksum,
            },
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
        return not self._missing_training_dependencies()

    def _materialise_fallback_adapter(
        self, *, reason: str, details: str | None = None
    ) -> dict[str, Any]:
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        weights_payload = self._derive_fallback_weights()
        weights_path = adapter_dir / "fallback_adapter.json"
        weights_path.write_text(json.dumps(weights_payload, indent=2, sort_keys=True))
        checksum = self._compute_file_checksum(weights_path)

        logger.info(
            "Fallback adapter generated",
            extra={
                "reason": reason,
                "weights_path": str(weights_path),
                "checksum": checksum,
            },
        )

        return {
            "status": TrainingStatus.FALLBACK.value,
            "reason": reason,
            "details": details,
            "version": checksum,
            "artifacts": {
                "adapter": str(adapter_dir),
                "weights": str(weights_path),
                "weights_checksum": checksum,
            },
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

    def _compute_file_checksum(self, path: Path) -> str:
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error(
                "Failed to compute checksum for artifact",
                extra={"path": str(path)},
                exc_info=exc,
            )
            raise RuntimeError("Failed to compute artifact checksum") from exc
        return hashlib.sha256(data).hexdigest()

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
        if missing := self._missing_training_dependencies():
            joined = ", ".join(sorted(missing))
            raise RuntimeError(f"Optional training dependencies unavailable: {joined}")

        dataset = self._load_dataset()
        model_name = self._resolve_model_name()
        tokenizer = self._prepare_tokenizer(model_name)
        training_dataset = self._prepare_mntp_dataset(dataset, tokenizer)
        if len(training_dataset) == 0:
            raise RuntimeError("MNTP dataset preprocessing yielded no examples")
        device = self._resolve_device()
        model = self._initialise_model(model_name, tokenizer, device=device)
        pad_token_id = self._resolve_pad_token_id(tokenizer)
        metrics = self._execute_training(
            model, training_dataset, pad_token_id, device=device
        )
        average_loss = metrics.get("average_loss")
        if not isinstance(average_loss, (int, float)) or not math.isfinite(
            average_loss
        ):
            raise RuntimeError("Training produced an invalid average loss metric")
        if average_loss < 0:
            raise RuntimeError("Training produced a negative average loss metric")
        try:
            source_records = len(dataset)
        except Exception:  # pragma: no cover - streaming datasets
            source_records = None
        if source_records is not None:
            metrics.setdefault("source_records", source_records)
        artifact_dir, weights_path = self._persist_model(model)
        checksum = self._compute_file_checksum(weights_path)

        logger.info(
            "Model training finished",
            extra={
                "artifact_dir": str(artifact_dir),
                "model_name": model_name,
                "examples": metrics.get("training_examples"),
                "checksum": checksum,
            },
        )
        artifacts: dict[str, str] = {
            "adapter": str(artifact_dir),
            "weights": str(weights_path),
            "weights_checksum": checksum,
        }
        return {
            "status": TrainingStatus.SUCCESS.value,
            "model_name": model_name,
            "dataset_name": self.config["dataset_name"],
            "version": checksum,
            "artifacts": artifacts,
            "metrics": metrics,
        }

    def _load_dataset(self) -> Sequence[dict[str, Any]]:
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

    def _prepare_tokenizer(self, model_name: str) -> Any:
        if AutoTokenizer is None:
            raise RuntimeError("transformers AutoTokenizer unavailable")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            pad_token = tokenizer.eos_token or tokenizer.unk_token
            if pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
            else:
                tokenizer.pad_token = pad_token
        tokenizer.padding_side = "left"
        if hasattr(tokenizer, "clean_up_tokenization_spaces"):
            tokenizer.clean_up_tokenization_spaces = True
        return tokenizer

    def _prepare_mntp_dataset(
        self, dataset: Sequence[dict[str, Any]], tokenizer: Any
    ) -> MaskedNextTokenDataset:
        probability = float(self.config.get("mlm_probability", 0.2))
        max_seq_length = int(self.config.get("max_seq_length", 512))
        max_masks = int(self.config.get("max_masks_per_sample", 4))
        seed = int(self.config.get("seed", 17))
        mask_token_id = self._resolve_mask_token_id(tokenizer)
        text_field = self.config.get("dataset_text_field")

        mntp_dataset = MaskedNextTokenDataset(
            dataset,
            tokenizer,
            max_seq_length=max_seq_length,
            mask_token_id=mask_token_id,
            mlm_probability=max(probability, 0.0),
            max_masks_per_sample=max_masks,
            seed=seed,
            text_field=str(text_field) if isinstance(text_field, str) else None,
        )
        return mntp_dataset

    def _resolve_mask_token_id(self, tokenizer: Any) -> int:
        mask_type = str(self.config.get("mask_token_type", "mask")).lower()
        if mask_type in {"mask", "mask_token"} and tokenizer.mask_token_id is not None:
            return int(tokenizer.mask_token_id)
        if mask_type == "pad" and tokenizer.pad_token_id is not None:
            return int(tokenizer.pad_token_id)
        if mask_type == "eos" and tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
        if mask_type == "blank":
            # blank means no explicit token; reuse pad/eos fallback
            if tokenizer.pad_token_id is not None:
                return int(tokenizer.pad_token_id)
            if tokenizer.eos_token_id is not None:
                return int(tokenizer.eos_token_id)
        # Generic fallback to ensure training can proceed
        if tokenizer.mask_token_id is not None:
            return int(tokenizer.mask_token_id)
        if tokenizer.pad_token_id is not None:
            return int(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
        return 0

    def _resolve_pad_token_id(self, tokenizer: Any) -> int:
        if tokenizer.pad_token_id is not None:
            return int(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
        if tokenizer.mask_token_id is not None:
            return int(tokenizer.mask_token_id)
        return 0

    def _initialise_model(
        self, model_name: str, tokenizer: Any, *, device: "torch.device"
    ) -> Any:
        if AutoModelForCausalLM is None or get_peft_model is None:
            raise RuntimeError("transformers/peft unavailable for training")

        dtype_name = str(self.config.get("torch_dtype", "float32"))
        torch_dtype = getattr(torch, dtype_name, None) if torch else None
        if torch_dtype is None and torch is not None:
            logger.warning(
                "Unsupported torch dtype '%s'; defaulting to float32", dtype_name
            )
            torch_dtype = torch.float32
        if (
            torch is not None
            and torch_dtype
            in {getattr(torch, "bfloat16", None), getattr(torch, "float16", None)}
            and device.type == "cpu"
        ):
            logger.info(
                "Requested torch dtype %s is not supported on CPU; using float32",
                dtype_name,
            )
            torch_dtype = torch.float32

        load_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        attn_impl = self.config.get("attn_implementation")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        try:
            base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:  # pragma: no cover - external library errors
            raise RuntimeError("Failed to load base language model") from exc

        if (
            tokenizer.pad_token_id is not None
            and base_model.config.pad_token_id is None
        ):
            base_model.config.pad_token_id = tokenizer.pad_token_id

        if hasattr(base_model, "resize_token_embeddings"):
            base_model.resize_token_embeddings(len(tokenizer))

        lora_config = self._build_lora_config()
        try:
            peft_model = get_peft_model(base_model, lora_config)
        except Exception as exc:  # pragma: no cover - PEFT misconfiguration
            raise RuntimeError("Failed to configure LoRA adapters") from exc

        return peft_model.to(device)

    def _build_lora_config(self) -> Any:
        if LoraConfig is None or TaskType is None:
            raise RuntimeError("peft library unavailable")

        target_modules = self.config.get("lora_target_modules")
        if isinstance(target_modules, str):
            modules = [
                item.strip() for item in target_modules.split(",") if item.strip()
            ]
        elif isinstance(target_modules, Sequence):
            modules = [str(item) for item in target_modules if str(item).strip()]
        else:
            modules = []
        if not modules:
            modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "c_attn",
                "c_proj",
            ]

        return LoraConfig(
            r=int(self.config.get("lora_r", 16)),
            lora_alpha=int(self.config.get("lora_alpha", 32)),
            lora_dropout=float(self.config.get("lora_dropout", 0.05)),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=modules,
        )

    def _execute_training(
        self,
        model: Any,
        dataset: MaskedNextTokenDataset,
        pad_token_id: int,
        *,
        device: "torch.device",
    ) -> dict[str, Any]:
        if torch is None or DataLoader is None or clip_grad_norm_ is None:
            raise RuntimeError("torch dependencies unavailable for training")

        if len(dataset) == 0:
            raise RuntimeError("MNTP training dataset is empty")

        batch_size = int(self.config.get("per_device_train_batch_size", 8))
        grad_accum = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        max_steps = max(1, int(self.config.get("max_steps", 32)))
        learning_rate = float(self.config.get("learning_rate", 5e-5))
        weight_decay = float(self.config.get("weight_decay", 0.0))
        max_grad_norm = float(self.config.get("max_grad_norm", 1.0))

        collator = MaskedNextTokenCollator(pad_token_id)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
        )

        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        optimizer.zero_grad()

        total_updates = max_steps
        warmup_steps = int(self.config.get("warmup_steps", max(1, total_updates // 10)))
        if get_linear_schedule_with_warmup is not None:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_updates,
            )
        else:  # pragma: no cover - fallback when scheduler unavailable
            scheduler = None

        total_loss = 0.0
        last_loss = 0.0
        completed_steps = 0
        examples_processed = 0
        correct_predictions = 0

        loss_fn = torch.nn.CrossEntropyLoss()
        micro_step = 0

        data_iterator = iter(data_loader)
        while completed_steps < total_updates:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                batch = next(data_iterator)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            examples_processed += labels.size(0)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            last_positions = attention_mask.sum(dim=1) - 1
            selected_logits = logits[
                torch.arange(logits.size(0), device=device), last_positions
            ]
            loss = loss_fn(selected_logits, labels)
            loss = loss / grad_accum
            loss.backward()

            micro_step += 1
            last_loss = loss.item() * grad_accum
            total_loss += last_loss

            batch_predictions = selected_logits.detach().argmax(dim=-1)
            correct_predictions += (batch_predictions == labels).sum().item()

            if micro_step % grad_accum == 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

        average_loss = total_loss / max(1, completed_steps)
        if completed_steps < total_updates:
            raise RuntimeError(
                "Training stopped before reaching the configured number of steps"
            )

        if examples_processed == 0:
            raise RuntimeError("No MNTP examples were processed during training")

        accuracy = correct_predictions / examples_processed

        metrics = {
            "training_examples": len(dataset),
            "processed_examples": examples_processed,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss": last_loss,
            "average_loss": average_loss,
            "accuracy": accuracy,
        }
        if scheduler is not None:
            metrics["warmup_steps"] = warmup_steps
        return metrics

    def _resolve_device(self) -> "torch.device":
        if torch is None:
            raise RuntimeError("torch unavailable")

        requested = self.config.get("device")
        if isinstance(requested, str) and requested:
            try:
                return torch.device(requested)
            except Exception:  # pragma: no cover - invalid user configuration
                logger.warning(
                    "Invalid device override '%s'; falling back to auto", requested
                )
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _persist_model(self, model: Any) -> tuple[Path, Path | None]:
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(str(adapter_dir))
            except Exception as exc:  # pragma: no cover - filesystem issues
                raise RuntimeError("Failed to persist PEFT adapter") from exc
        else:  # pragma: no cover - incompatible model
            raise RuntimeError("Model does not support save_pretrained")

        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            raise RuntimeError("Adapter configuration missing after save_pretrained")

        weight_candidates = [
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "adapter_model.bin",
        ]
        weights_path = next((path for path in weight_candidates if path.exists()), None)
        if weights_path is None:
            logging.error(
                "Adapter weights missing after save_pretrained. Checked candidate paths",
                extra={"candidates": [str(path) for path in weight_candidates]},
            )
            raise RuntimeError("Adapter weights missing after save_pretrained")

        return adapter_dir, weights_path

    def _missing_training_dependencies(self) -> list[str]:
        dependencies: dict[str, Any] = {
            "torch": torch,
            "datasets.load_dataset": load_dataset,
            "transformers.AutoTokenizer": AutoTokenizer,
            "transformers.AutoModelForCausalLM": AutoModelForCausalLM,
            "peft.get_peft_model": get_peft_model,
            "torch.utils.data.DataLoader": DataLoader,
            "torch.nn.utils.clip_grad_norm_": clip_grad_norm_,
        }
        return [name for name, value in dependencies.items() if value is None]
