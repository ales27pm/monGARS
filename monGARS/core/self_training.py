from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from monGARS.core.neurones import EmbeddingSystem

logger = logging.getLogger(__name__)


class SelfTrainingEngine:
    """Simplified self-training engine that batches data and records model versions."""

    DEFAULT_BATCH_LIMIT = 100

    def __init__(
        self,
        training_threshold: float = 0.8,
        retrain_interval: int = 3600,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
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
        batch: list[Dict[str, Any]] = []
        discarded = 0
        while not self.training_queue.empty() and len(batch) < self.batch_limit:
            record = await self.training_queue.get()
            _, accepted = self._assess_record_confidence(record)
            if accepted:
                batch.append(record)
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
            new_version = len(self.model_versions) + 1
            loop = asyncio.get_running_loop()
            self.model_versions[f"v{new_version}"] = {
                "trained_at": loop.time(),
                "data_count": len(batch),
            }
            logger.info("Training complete. New model version: v%s", new_version)
            self.last_retrain_time = loop.time()

    def shutdown(self) -> None:
        """Signal the auto improvement loop to stop."""
        self._shutdown_event.set()

    def _assess_record_confidence(self, record: Dict[str, Any]) -> tuple[float, bool]:
        """Return the parsed confidence value and whether it meets the threshold.

        Missing or malformed confidence values are treated as ``0.0`` to ensure
        they are excluded unless the threshold is configured to accept such
        entries explicitly.
        """

        confidence_raw = record.get("confidence")
        try:
            confidence_value = (
                float(confidence_raw) if confidence_raw is not None else 0.0
            )
        except (TypeError, ValueError):
            confidence_value = 0.0
        return confidence_value, confidence_value >= self.training_threshold
