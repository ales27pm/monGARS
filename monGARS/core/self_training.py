from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
from uuid import uuid4

from monGARS.core.neurones import EmbeddingSystem

logger = logging.getLogger(__name__)


class SelfTrainingEngine:
    """Batch curated records and trigger incremental training updates."""

    DEFAULT_BATCH_LIMIT = 100

    def __init__(
        self,
        training_threshold: float = 0.8,
        retrain_interval: int = 3600,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
        *,
        trainer_cls: type | None = None,
        training_config_path: str | None = None,
        dataset_root: str | None = None,
        model_registry_path: str | None = None,
        curated_feature_limit: int = 128,
    ) -> None:
        self.training_threshold = training_threshold
        self.retrain_interval = retrain_interval
        self.batch_limit = batch_limit
        self.training_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self.model_versions: Dict[str, Dict[str, Any]] = {}
        self.last_retrain_time: float = 0.0
        self._embedding_model = EmbeddingSystem()
        self.lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        if trainer_cls is None:
            from modules.neurons.training.mntp_trainer import MNTPTrainer

            self._trainer_cls = MNTPTrainer
        else:
            self._trainer_cls = trainer_cls
        self.training_config_path = Path(
            training_config_path or "configs/training/mntp_mistral_config.json"
        )
        self.dataset_root = Path(dataset_root or "models/datasets/curated")
        self.model_registry_path = Path(
            model_registry_path or "models/encoders/self_training"
        )
        self.curated_feature_limit = max(1, curated_feature_limit)
        logger.info("SelfTrainingEngine initialized.")

    async def auto_improve(self) -> None:
        """Periodically trigger training cycles until shutdown."""

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self.retrain_interval
                )
            except asyncio.TimeoutError:
                await self._run_training_cycle()

    async def _run_training_cycle(self) -> None:
        batch: list[tuple[Dict[str, Any], float]] = []
        discarded = 0
        while not self.training_queue.empty() and len(batch) < self.batch_limit:
            record = await self.training_queue.get()
            confidence, accepted = self._assess_record_confidence(record)
            if accepted:
                batch.append((record, confidence))
            else:
                discarded += 1

        logger.info(
            "training_cycle_evaluated",
            extra={
                "accepted_items": len(batch),
                "discarded_items": discarded,
                "threshold": self.training_threshold,
            },
        )

        if not batch:
            return

        async with self.lock:
            curated_batch, fallback_count = await self._prepare_curated_batch(batch)
            if not curated_batch:
                logger.info("No curated samples available; skipping training run")
                return

            dataset_metadata = await asyncio.to_thread(
                self._persist_curated_dataset, curated_batch
            )

            try:
                summary = await asyncio.to_thread(
                    self._launch_trainer, curated_batch, dataset_metadata
                )
            except Exception as exc:  # pragma: no cover - unexpected training error
                logger.error("Self-training run failed: %s", exc, exc_info=True)
                return

            new_version = len(self.model_versions) + 1
            loop = asyncio.get_running_loop()
            version_key = f"v{new_version}"
            self.model_versions[version_key] = {
                "trained_at": loop.time(),
                "data_count": len(curated_batch),
                "dataset": dataset_metadata,
                "summary": summary,
                "fallback_embeddings": fallback_count,
            }
            logger.info("Training complete. New model version: %s", version_key)
            self.last_retrain_time = loop.time()

    def shutdown(self) -> None:
        """Signal the auto improvement loop to stop."""

        self._shutdown_event.set()

    def _assess_record_confidence(self, record: Dict[str, Any]) -> tuple[float, bool]:
        """Return the parsed confidence value and whether it meets the threshold."""

        confidence_raw = record.get("confidence")
        try:
            confidence_value = (
                float(confidence_raw) if confidence_raw is not None else 0.0
            )
        except (TypeError, ValueError):
            confidence_value = 0.0
        return confidence_value, confidence_value >= self.training_threshold

    async def _prepare_curated_batch(
        self, batch: Sequence[tuple[Dict[str, Any], float]]
    ) -> tuple[list[dict[str, Any]], int]:
        curated: list[dict[str, Any]] = []
        fallback_count = 0

        for record, confidence in batch:
            text = self._extract_training_text(record)
            if not text:
                logger.debug("Skipping record without trainable text")
                continue

            try:
                embedding, used_fallback = await self._embedding_model.encode(text)
            except Exception as exc:  # pragma: no cover - unexpected embedding error
                logger.warning("Embedding failed for curated record: %s", exc)
                continue

            fallback_count += 1 if used_fallback else 0
            trimmed_embedding = self._trim_embedding(embedding)
            if not trimmed_embedding:
                logger.debug("Trimmed embedding empty; skipping record")
                continue

            curated.append(
                {
                    "embedding": trimmed_embedding,
                    "target": confidence,
                    "confidence": confidence,
                    "source_id": record.get("id") or record.get("message_id"),
                    "text_preview": text[:200],
                    "used_fallback_embedding": used_fallback,
                }
            )

        return curated, fallback_count

    def _extract_training_text(self, record: Dict[str, Any]) -> str | None:
        candidates: Iterable[str] = (
            record.get("text"),
            record.get("response"),
            record.get("prompt"),
            record.get("content"),
            record.get("data"),
        )
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _trim_embedding(self, embedding: Sequence[Any]) -> list[float]:
        trimmed: list[float] = []
        for index, value in enumerate(embedding):
            if index >= self.curated_feature_limit:
                break
            try:
                trimmed.append(float(value))
            except (TypeError, ValueError):
                break
        return trimmed

    def _persist_curated_dataset(
        self, curated_batch: Sequence[dict[str, Any]]
    ) -> Dict[str, Any]:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        run_id = f"self-training-{timestamp}-{uuid4().hex[:6]}"
        dataset_dir = self.dataset_root / run_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_dir / "curated_batch.jsonl"

        with dataset_file.open("w", encoding="utf-8") as handle:
            for record in curated_batch:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")

        return {
            "run_id": run_id,
            "dataset_dir": str(dataset_dir),
            "dataset_file": str(dataset_file),
            "records": len(curated_batch),
        }

    def _launch_trainer(
        self,
        curated_batch: Sequence[dict[str, Any]],
        dataset_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_dir = self.model_registry_path / dataset_metadata["run_id"]
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = self._trainer_cls(
            training_config_path=str(self.training_config_path),
            output_dir=str(output_dir),
        )
        summary = trainer.train(curated_records=curated_batch)
        summary.setdefault("dataset", dataset_metadata)
        summary.setdefault("artifacts", {})
        summary["artifacts"].setdefault("training_output", str(output_dir))
        return summary
