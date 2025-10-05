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

from models.datasets import DatasetCatalog, sanitize_record, scrub_text
from monGARS.core.neurones import EmbeddingSystem

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
        """
        Initialize the SelfTrainingEngine and configure runtime resources for batching, embedding, dataset management, and training orchestration.
        
        Parameters:
        	training_threshold (float): Minimum confidence required for a record to be accepted into a training batch.
        	retrain_interval (int): Interval in seconds between automatic training cycles.
        	batch_limit (int): Maximum number of records to consume from the queue per training cycle.
        	trainer_cls (type | None): Optional trainer class to use for training runs; defaults to the module's MNTPTrainer when None.
        	training_config_path (str | None): Path to the trainer configuration file; defaults to the module's standard config when None.
        	dataset_root (str | None): Root directory where curated datasets will be stored; defaults to the module's curated dataset path when None.
        	model_registry_path (str | None): Directory for trainer outputs and model artifacts; defaults to the module's encoder registry path when None.
        	curated_feature_limit (int): Maximum number of features to keep from embeddings when building curated records; clamped to at least 1.
        """
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
        self._dataset_catalog = DatasetCatalog(self.dataset_root)
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

        for record, confidence in batch:
            text = self._extract_training_text(record)
            if not text:
                logger.debug("Skipping record without trainable text")
                continue

            sanitized_text = scrub_text(text)

            try:
                embedding, used_fallback = await self._embedding_model.encode(
                    sanitized_text
                )
            except Exception as exc:  # pragma: no cover - unexpected embedding error
                logger.warning("Embedding failed for curated record: %s", exc)
                continue

            fallback_count += 1 if used_fallback else 0
            trimmed_embedding = self._trim_embedding(embedding)
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

        version = self._dataset_catalog.register(
            run_id=run_id,
            dataset_dir=dataset_dir,
            dataset_file=dataset_file,
            record_count=len(curated_batch),
        )

        return {
            "run_id": run_id,
            "dataset_dir": str(dataset_dir),
            "dataset_file": str(dataset_file),
            "records": len(curated_batch),
            "version": version.version,
            "catalog_entry": version.as_dict(),
        }

    def _launch_trainer(
        self,
        curated_batch: Sequence[dict[str, Any]],
        dataset_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the configured trainer on the provided curated records and ensure training artifacts and dataset metadata are recorded.
        
        Parameters:
            curated_batch (Sequence[dict[str, Any]]): Curated training records to pass to the trainer.
            dataset_metadata (Dict[str, Any]): Metadata for the curated dataset; must include a `run_id` used to create the trainer output directory.
        
        Returns:
            Dict[str, Any]: Trainer-produced summary augmented with a `dataset` entry set to `dataset_metadata` and an `artifacts.training_output` entry pointing to the trainer output directory.
        """
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
        """
        Builds a mixed reasoning dataset from internal history and GSM8K, then returns train/test splits.
        
        Collects up to `num_samples` examples, sourcing approximately `internal_ratio` fraction from internal reasoning records and the remainder from the GSM8K dataset when available. If multiple source datasets are present they are concatenated and then split into train and test partitions (uses a small fixed seed for reproducibility).
        
        Parameters:
            num_samples (int): Maximum number of examples to include; must be greater than zero.
            internal_ratio (float): Fraction in [0.0, 1.0] of samples to attempt to source from internal records (the rest are drawn from GSM8K).
        
        Returns:
            tuple[Any, Any]: A (train_dataset, test_dataset) pair containing the curated examples.
        
        Raises:
            ValueError: If `num_samples` is not positive.
            RuntimeError: If the `datasets` library is unavailable or if no qualifying samples could be collected.
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        bounded_ratio = max(0.0, min(1.0, float(internal_ratio)))
        internal_limit = int(num_samples * bounded_ratio)
        external_limit = max(0, num_samples - internal_limit)

        try:  # pragma: no cover - heavy dependency gated at runtime
            from datasets import Dataset, concatenate_datasets, load_dataset
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
        """
        Collect up to `limit` internal reasoning examples from the local Hippocampus history and format them for dataset curation.
        
        This attempts to import and query the Hippocampus history and returns curated samples that look like reasoning prompts with extracted final answers. Each returned item is a dict with keys:
        - `prompt`: a list of messages suitable for chat-style models (system message using the engine's SYSTEM_PROMPT, then the user query),
        - `answer`: the extracted final answer string,
        - `metadata`: a dict containing at least `source: "hippocampus"`.
        
        Parameters:
            limit (int): Maximum number of curated samples to return. Non-positive values yield an empty list.
        
        Returns:
            list[dict[str, Any]]: A list of curated sample dictionaries (possibly empty). The function returns an empty list if Hippocampus is unavailable, history cannot be retrieved, or no suitable reasoning records are found.
        """
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
        """
        Run a Hippocampus client's history retrieval in a dedicated background thread when an asyncio event loop is already running.
        
        Parameters:
            hippocampus: Hippocampus client instance providing a `history` coroutine.
            history_limit (int): Maximum number of history entries to request.
        
        Returns:
            list[Any]: The list of history records returned by `hippocampus.history`, or an empty list if no results were produced.
        
        Raises:
            BaseException: Re-raises any exception raised while executing `hippocampus.history`.
        """

        result_holder: list[list[Any]] = []
        error_holder: list[BaseException] = []

        def runner() -> None:
            """
            Run hippocampus.history in a standalone event loop and capture its result or any raised exception.
            
            This function calls asyncio.run on the coroutine returned by hippocampus.history("global", limit=history_limit),
            then appends the resulting iterable converted to a list into the outer-scope list `result_holder`. If any exception
            occurs during execution, the exception is appended to the outer-scope list `error_holder`.
            """
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
        """
        Load and format a slice of the GSM8K reasoning dataset into the engine's prompt/answer structure.
        
        Parameters:
            limit (int): Maximum number of GSM8K examples to load and return; values <= 0 cause immediate return of `None`.
        
        Returns:
            datasets.Dataset | None: A datasets.Dataset whose records are dicts with keys `prompt` (a list of system/user message dicts), `answer` (extracted answer string), and `metadata` (`{"source": "gsm8k"}`); returns `None` if the `datasets` library is unavailable or the GSM8K dataset cannot be loaded.
        """
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
            """
            Format a GSM8K dataset example into the internal reasoning sample structure.
            
            Parameters:
                example (dict[str, Any]): A raw GSM8K example expected to contain "question" and "answer" fields.
            
            Returns:
                dict[str, Any]: A mapping with keys:
                    - `prompt`: a list of role/content messages including the system prompt and the user question,
                    - `answer`: the extracted GSM8K answer string,
                    - `metadata`: a dict with `source` set to `"gsm8k"`.
            """
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
        """
        Determine whether a user query likely requires step-by-step reasoning or numeric calculation.
        
        Returns:
            True if the query contains common reasoning/math keywords or a numeric expression with an operator, False otherwise.
        """
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
        """
        Extract the final answer from a reasoning or explanation text.
        
        First looks for an explicit final-answer pattern and returns its captured content; if no pattern is found, returns the last non-empty line of the input. Empty or None input yields an empty string.
        
        Parameters:
            text (str | None): Text containing reasoning, explanation, or an answer.
        
        Returns:
            str: The extracted answer, or an empty string if no answer is found.
        """
        if not text:
            return ""
        match = SelfTrainingEngine._ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def extract_reasoning_chain(text: str | None) -> str:
        """
        Extracts the reasoning chain from the given text.
        
        Parameters:
            text (str | None): Text to search for a reasoning section.
        
        Returns:
            str: The extracted reasoning content trimmed of surrounding whitespace, or an empty string if no reasoning section is found.
        """
        if not text:
            return ""
        match = SelfTrainingEngine._REASONING_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def extract_gsm8k_answer(text: str | None) -> str:
        """
        Extracts the GSM8K-style final answer from a model's solution text.
        
        Parameters:
            text (str | None): Text that may contain a GSM8K formatted solution and final answer.
        
        Returns:
            str: The extracted final answer trimmed of surrounding whitespace, or an empty string if no GSM8K answer is found.
        """
        if not text:
            return ""
        match = SelfTrainingEngine._GSM_PATTERN.search(text)
        return match.group(1).strip() if match else ""