from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from datetime import datetime

try:
    from datetime import UTC  # Python 3.11+
except ImportError:  # Python 3.10 fallback
    from datetime import timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
from uuid import uuid4

from models.datasets import (
    DatasetCatalog,
    DatasetGovernance,
    sanitize_record,
    scrub_text,
)
from monGARS.core.llm_integration import LLMIntegration

logger = logging.getLogger(__name__)


class SelfTrainingEngine:
    """Batch curated records and trigger incremental training updates."""

    DEFAULT_BATCH_LIMIT = 100
    SYSTEM_PROMPT = (
        "You are the monGARS reasoning engine. Respond with explicit step-by-step "
        "analysis enclosed in <reasoning>...</reasoning> tags and finish with a "
        "concise conclusion inside <answer>...</answer>."
    )
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    _REASONING_PATTERN = re.compile(
        r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL
    )
    _GSM_PATTERN = re.compile(r"####\s*(.+)")

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
        self._llm: LLMIntegration | None
        try:
            self._llm = LLMIntegration.instance()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning(
                "self_training.embedding_unavailable", extra={"error": repr(exc)}
            )
            self._llm = None
        self.lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        if trainer_cls is None:
            from modules.neurons.training.mntp_trainer import MNTPTrainer

            self._trainer_cls = MNTPTrainer
        else:
            self._trainer_cls = trainer_cls
        self.training_config_path = Path(
            training_config_path or "configs/training/mntp_dolphin_config.json"
        )
        self.dataset_root = Path(dataset_root or "models/datasets/curated")
        self.model_registry_path = Path(
            model_registry_path or "models/encoders/self_training"
        )
        self.curated_feature_limit = max(1, curated_feature_limit)
        self._dataset_catalog = DatasetCatalog(self.dataset_root)
        self._dataset_governance = DatasetGovernance()
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

            sanitized_batch = [sanitize_record(record) for record in curated_batch]

            dataset_metadata = await asyncio.to_thread(
                self._persist_curated_dataset, sanitized_batch
            )

            try:
                summary = await asyncio.to_thread(
                    self._launch_trainer, sanitized_batch, dataset_metadata
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

        if self._llm is None:
            logger.warning("self_training.embedding_runtime_missing")
            return curated, fallback_count

        for record, confidence in batch:
            text = self._extract_training_text(record)
            if not text:
                logger.debug("Skipping record without trainable text")
                continue

            sanitized_text = scrub_text(text)

            vectors = await self._embed_texts([sanitized_text])
            if not vectors:
                logger.warning(
                    "Embedding failed for curated record: %s", sanitized_text[:40]
                )
                continue

            trimmed_embedding = self._trim_embedding(vectors[0])
            if not trimmed_embedding:
                logger.debug("Trimmed embedding empty; skipping record")
                continue

            source_id = record.get("id") or record.get("message_id")
            if isinstance(source_id, str):
                source_id = scrub_text(source_id)

            curated.append(
                {
                    "embedding": trimmed_embedding,
                    "target": confidence,
                    "confidence": confidence,
                    "source_id": source_id,
                    "text_preview": sanitized_text[:200],
                    "used_fallback_embedding": False,
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
        return next(
            (
                value.strip()
                for value in candidates
                if isinstance(value, str) and value.strip()
            ),
            None,
        )

    def _trim_embedding(self, embedding: Sequence[Any]) -> list[float]:
        trimmed: list[float] = []
        for index, value in enumerate(embedding):
            if index >= self.curated_feature_limit:
                break
            try:
                trimmed.append(float(value))
            except (TypeError, ValueError):
                continue
        return trimmed

    async def _embed_texts(self, texts: Sequence[str]) -> list[list[float]] | None:
        if not texts:
            return None
        if self._llm is None:
            return None
        try:
            return await asyncio.to_thread(self._llm.embed_batch, list(texts))
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning(
                "self_training.embedding_runtime_error", extra={"error": repr(exc)}
            )
            return None

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

        created_at = datetime.now(UTC)
        evaluation = self._dataset_governance.evaluate_dataset(
            dataset_file,
            run_id=run_id,
            record_count=len(curated_batch),
            created_at=created_at,
        )

        version = self._dataset_catalog.register(
            run_id=run_id,
            dataset_dir=dataset_dir,
            dataset_file=dataset_file,
            record_count=len(curated_batch),
            created_at=created_at,
            governance=evaluation.metadata,
            compliance=evaluation.as_dict(),
            quarantined=evaluation.status != "approved",
        )

        return {
            "run_id": run_id,
            "dataset_dir": str(dataset_dir),
            "dataset_file": str(dataset_file),
            "records": len(curated_batch),
            "version": version.version,
            "governance": evaluation.metadata,
            "compliance": evaluation.as_dict(),
            "quarantined": evaluation.status != "approved",
            "catalog_entry": version.as_dict(),
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

    # ------------------------------------------------------------------
    # Reasoning dataset curation
    # ------------------------------------------------------------------
    def curate_reasoning_dataset(
        self, num_samples: int = 200, internal_ratio: float = 0.5
    ) -> tuple[Any, Any]:
        """Curate a mixed dataset of internal and external reasoning prompts."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        bounded_ratio = max(0.0, min(1.0, float(internal_ratio)))
        internal_limit = int(num_samples * bounded_ratio)
        external_limit = max(0, num_samples - internal_limit)

        try:  # pragma: no cover - heavy dependency gated at runtime
            from datasets import Dataset, concatenate_datasets
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "The 'datasets' package is required for reasoning dataset curation"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError("Failed to import the datasets library") from exc

        datasets: list[Dataset] = []

        internal_records = self._collect_internal_reasoning_records(internal_limit)
        if internal_records:
            datasets.append(Dataset.from_list(internal_records))

        if external_limit:
            gsm_dataset = self._load_gsm8k_reasoning_dataset(external_limit)
            if gsm_dataset is not None and len(gsm_dataset) > 0:
                datasets.append(gsm_dataset)

        if not datasets:
            raise RuntimeError(
                "Unable to curate reasoning dataset; no qualifying samples available"
            )

        combined: Dataset
        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = concatenate_datasets(datasets)

        if len(combined) <= 1:
            return combined, combined

        if len(combined) < 10:
            test_size = max(1, len(combined) // 5)
        else:
            test_size = 0.2

        split = combined.train_test_split(test_size=test_size, seed=3407)
        return split["train"], split["test"]

    def _collect_internal_reasoning_records(self, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        try:
            from monGARS.core.hippocampus import Hippocampus
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning(
                "self_training.reasoning.internal_unavailable",
                extra={"error": str(exc)},
            )
            return []

        hippocampus = Hippocampus(enable_scheduler=False)
        history_limit = min(limit * 3, getattr(hippocampus, "MAX_HISTORY", 100))
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        else:  # pragma: no cover - depends on runtime environment
            running_loop = True

        try:
            if running_loop:
                records = self._run_history_with_dedicated_loop(
                    hippocampus, history_limit
                )
            else:
                records = asyncio.run(
                    hippocampus.history("global", limit=history_limit)
                )
        except RuntimeError as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "self_training.reasoning.loop_unavailable",
                extra={"error": str(exc)},
            )
            return []
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "self_training.reasoning.history_failed",
                extra={"error": str(exc)},
            )
            return []

        curated: list[dict[str, Any]] = []
        for item in records:
            query = getattr(item, "query", None)
            response = getattr(item, "response", None)
            if not isinstance(query, str) or not isinstance(response, str):
                continue
            if not self.is_reasoning_query(query):
                continue
            answer = self.extract_final_answer(response)
            if not answer:
                continue
            curated.append(
                {
                    "prompt": [
                        {"role": "system", "content": self.SYSTEM_PROMPT.strip()},
                        {"role": "user", "content": query.strip()},
                    ],
                    "answer": answer,
                    "metadata": {"source": "hippocampus"},
                }
            )
            if len(curated) >= limit:
                break
        return curated

    def _run_history_with_dedicated_loop(
        self, hippocampus: Any, history_limit: int
    ) -> list[Any]:
        """Execute ``Hippocampus.history`` when an event loop is already running."""

        result_holder: list[list[Any]] = []
        error_holder: list[BaseException] = []

        def runner() -> None:
            try:
                coro = hippocampus.history("global", limit=history_limit)
                result = asyncio.run(coro)
                result_holder.append(list(result))
            except BaseException as err:  # pragma: no cover - defensive guard
                error_holder.append(err)

        thread = threading.Thread(
            target=runner,
            name="hippocampus-history",
            daemon=True,
        )
        thread.start()
        thread.join()

        if error_holder:
            raise error_holder[0]

        return result_holder[0] if result_holder else []

    def _load_gsm8k_reasoning_dataset(self, limit: int) -> Any:
        if limit <= 0:
            return None
        try:  # pragma: no cover - heavy dependency gated at runtime
            from datasets import load_dataset
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            logger.warning("self_training.reasoning.gsm8k_missing")
            return None

        slice_expr = f"train[:{limit}]" if limit < 7474 else "train"
        try:
            dataset = load_dataset("openai/gsm8k", "main", split=slice_expr)
        except Exception as exc:  # pragma: no cover - network/offline guard
            logger.warning(
                "self_training.reasoning.gsm8k_unavailable",
                extra={"error": str(exc)},
            )
            return None

        def _format_record(example: dict[str, Any]) -> dict[str, Any]:
            question = (example.get("question") or "").strip()
            answer = self.extract_gsm8k_answer(example.get("answer") or "")
            return {
                "prompt": [
                    {"role": "system", "content": self.SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": question},
                ],
                "answer": answer,
                "metadata": {"source": "gsm8k"},
            }

        formatted = dataset.map(_format_record, remove_columns=dataset.column_names)
        filtered = formatted.filter(lambda entry: bool(entry["answer"]))
        return filtered.select(range(min(len(filtered), limit)))

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_reasoning_query(query: str) -> bool:
        lowered = query.lower()
        keyword_hits = (
            "calculate",
            "solve",
            "derive",
            "proof",
            "reason",
            "step",
            "explain",
        )
        if any(token in lowered for token in keyword_hits):
            return True
        return bool(re.search(r"\d+\s*[+\-*/^]", lowered))

    @staticmethod
    def extract_final_answer(text: str | None) -> str:
        if not text:
            return ""
        match = SelfTrainingEngine._ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def extract_reasoning_chain(text: str | None) -> str:
        if not text:
            return ""
        match = SelfTrainingEngine._REASONING_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def extract_gsm8k_answer(text: str | None) -> str:
        if not text:
            return ""
        match = SelfTrainingEngine._GSM_PATTERN.search(text)
        return match.group(1).strip() if match else ""
