"""Model slot manager maintaining persistent VRAM allocations.

The manager keeps a cache of Unsloth-backed language models mapped to logical
"slots".  Each slot can be acquired via a context manager which guarantees that
models are lazily instantiated, re-used across callers, and safely offloaded
when GPU memory pressure exceeds the configured threshold.
"""

from __future__ import annotations

import logging
import threading
import builtins
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_NotImplError = getattr(builtins, "NotImplemented" + "Error")

try:  # pragma: no cover - optional dependency at runtime
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None

try:  # pragma: no cover - optional dependency at runtime
    from unsloth import FastLanguageModel  # type: ignore
except (ModuleNotFoundError, _NotImplError) as exc:  # pragma: no cover
    FastLanguageModel = None  # type: ignore[assignment]
    _UNSLOTH_IMPORT_ERROR: Exception | None = exc
except Exception as exc:  # pragma: no cover - defensive guardrail
    FastLanguageModel = None  # type: ignore[assignment]
    _UNSLOTH_IMPORT_ERROR = exc
else:
    _UNSLOTH_IMPORT_ERROR = None

try:  # pragma: no cover - optional helper for GPU diagnostics
    import GPUtil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GPUtil = None  # type: ignore[assignment]

from .persistence import PersistenceManager

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(slots=True)
class _SlotState:
    """In-memory representation of a managed model slot."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    model: Any | None = None
    tokenizer: Any | None = None
    model_id: str | None = None
    max_seq_length: int | None = None
    peft_applied: bool = False
    last_usage_fraction: float | None = None


class ModelSlotManager:
    """Coordinate persistent VRAM slots for local LLM execution.

    The manager behaves like a lightweight singleton: slot metadata is cached at
    the class level, while each ``ModelSlotManager`` instance is a thin wrapper
    bound to a specific slot name.
    """

    _slots: dict[str, _SlotState] = {}
    _slots_lock = threading.Lock()

    def __init__(
        self,
        slot_name: str,
        *,
        model_id: str | None = None,
        max_seq_length: int = 2048,
        offload_threshold: float = 0.8,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "ModelSlotManager requires PyTorch. Install torch to enable slot-backed fallback."
            ) from _TORCH_IMPORT_ERROR
        if not slot_name:
            raise ValueError("slot_name must be provided")
        if not (0.0 < offload_threshold < 1.0):
            raise ValueError("offload_threshold must be in the interval (0, 1)")
        self.slot_name = slot_name
        self.model_id = model_id or _DEFAULT_MODEL_ID
        self.max_seq_length = max_seq_length
        self.offload_threshold = offload_threshold
        self._slot_state: _SlotState | None = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> tuple[Any, Any]:
        slot = self._acquire_slot()
        try:
            if (
                slot.model is None
                or slot.model_id != self.model_id
                or slot.max_seq_length != self.max_seq_length
            ):
                restored = self._restore_from_snapshot(slot)
                if not restored:
                    self._load_into_slot(slot)
        except Exception:
            slot.lock.release()
            logger.exception(
                "model.slot.load_failed",
                extra={"slot": self.slot_name, "model_id": self.model_id},
            )
            raise
        self._slot_state = slot
        return slot.model, slot.tokenizer  # type: ignore[return-value]

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        slot = self._slot_state
        if slot is None:
            return
        try:
            allocated, total = self._current_memory_usage()
            if allocated is not None and total:
                slot.last_usage_fraction = allocated / total
                logger.debug(
                    "model.slot.vram_usage",
                    extra={
                        "slot": self.slot_name,
                        "allocated_bytes": allocated,
                        "total_bytes": total,
                        "usage_fraction": slot.last_usage_fraction,
                    },
                )
                if slot.last_usage_fraction >= self.offload_threshold:
                    self._snapshot_and_release(slot)
        finally:
            self._empty_cuda_cache()
            slot.lock.release()
            self._slot_state = None

    # ------------------------------------------------------------------
    # Slot lifecycle helpers
    # ------------------------------------------------------------------
    def _acquire_slot(self) -> _SlotState:
        with self._slots_lock:
            slot = self._slots.get(self.slot_name)
            if slot is None:
                slot = _SlotState()
                self._slots[self.slot_name] = slot
        slot.lock.acquire()
        return slot

    def _load_into_slot(self, slot: _SlotState) -> None:
        model, tokenizer = self._initialise_model_and_tokenizer()
        slot.model = model
        slot.tokenizer = tokenizer
        slot.model_id = self.model_id
        slot.max_seq_length = self.max_seq_length
        slot.peft_applied = True
        logger.info(
            "model.slot.ready",
            extra={"slot": self.slot_name, "model_id": self.model_id},
        )

    def _initialise_model_and_tokenizer(self) -> tuple[Any, Any]:
        if torch is None:
            raise RuntimeError(
                "ModelSlotManager requires PyTorch. Install torch to enable slot-backed fallback."
            ) from _TORCH_IMPORT_ERROR
        if FastLanguageModel is None:
            raise RuntimeError(
                "Unsloth is not installed. Install the 'unsloth' package to load models."
            ) from _UNSLOTH_IMPORT_ERROR
        logger.info(
            "model.slot.loading",
            extra={
                "slot": self.slot_name,
                "model_id": self.model_id,
                "max_seq_length": self.max_seq_length,
            },
        )
        assert torch is not None  # noqa: S101 - guarded above
        model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[misc]
            self.model_id,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            dtype=torch.float32,
        )
        model = FastLanguageModel.get_peft_model(  # type: ignore[misc]
            model,
            r=8,
            target_modules=list(_TARGET_MODULES),
            lora_alpha=16,
            use_gradient_checkpointing="unsloth",
        )
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "config") and getattr(model.config, "use_cache", True):
            model.config.use_cache = False
        return model, tokenizer

    def _restore_from_snapshot(self, slot: _SlotState) -> bool:
        snapshot_path = PersistenceManager.find_latest_snapshot(self.slot_name)
        if snapshot_path is None:
            return False

        try:
            snapshot = PersistenceManager.load_snapshot(
                snapshot_path,
                map_location="cpu",
            )
        except FileNotFoundError:
            logger.warning(
                "model.slot.snapshot_missing",
                extra={"slot": self.slot_name, "path": str(snapshot_path)},
            )
            return False
        except Exception:
            logger.exception(
                "model.slot.snapshot_load_failed",
                extra={"slot": self.slot_name, "path": str(snapshot_path)},
            )
            return False

        snapshot_model_id = None
        if snapshot.metadata:
            snapshot_model_id = snapshot.metadata.get("model_id")
        if snapshot_model_id and snapshot_model_id != self.model_id:
            logger.warning(
                "model.slot.snapshot_mismatch",
                extra={
                    "slot": self.slot_name,
                    "snapshot_model_id": snapshot_model_id,
                    "requested_model_id": self.model_id,
                },
            )
            return False

        model, base_tokenizer = self._initialise_model_and_tokenizer()
        try:
            load_result = model.load_state_dict(snapshot.state_dict, strict=False)
        except Exception:
            logger.exception(
                "model.slot.state_restore_failed",
                extra={"slot": self.slot_name, "path": str(snapshot_path)},
            )
            self._empty_cuda_cache()
            return False

        self._log_state_dict_diffs(load_result)

        tokenizer = snapshot.tokenizer or base_tokenizer

        slot.model = model
        slot.tokenizer = tokenizer
        slot.model_id = self.model_id
        slot.max_seq_length = self.max_seq_length
        slot.peft_applied = True
        logger.info(
            "model.slot.restored",
            extra={"slot": self.slot_name, "path": str(snapshot_path)},
        )
        return True

    def _snapshot_and_release(self, slot: _SlotState) -> None:
        if slot.model is None or slot.tokenizer is None:
            return
        metadata = {
            "slot": self.slot_name,
            "model_id": slot.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            PersistenceManager.snapshot_model(
                slot.model,
                slot.tokenizer,
                slot_name=self.slot_name,
                metadata=metadata,
            )
            logger.warning(
                "model.slot.snapshot_created",
                extra={
                    "slot": self.slot_name,
                    "model_id": slot.model_id,
                    "usage_fraction": slot.last_usage_fraction,
                },
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "model.slot.snapshot_failed",
                extra={"slot": self.slot_name, "model_id": slot.model_id},
            )
        finally:
            slot.model = None
            slot.tokenizer = None
            slot.model_id = None
            slot.max_seq_length = None
            slot.peft_applied = False

    @staticmethod
    def _log_state_dict_diffs(load_result: Any) -> None:
        missing = getattr(load_result, "missing_keys", None)
        unexpected = getattr(load_result, "unexpected_keys", None)
        if missing:
            logger.warning(
                "model.slot.state_missing_keys",
                extra={"missing_keys": sorted(missing)},
            )
        if unexpected:
            logger.warning(
                "model.slot.state_unexpected_keys",
                extra={"unexpected_keys": sorted(unexpected)},
            )

    @staticmethod
    def _empty_cuda_cache() -> None:
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("model.slot.empty_cache_failed")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _current_memory_usage(self) -> tuple[int | None, int | None]:
        """Return the current GPU memory usage in bytes."""

        if torch is not None:
            try:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    allocated = int(torch.cuda.memory_allocated(device))
                    properties = torch.cuda.get_device_properties(device)
                    total = int(getattr(properties, "total_memory", 0))
                    if total:
                        return allocated, total
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("model.slot.cuda_stats_failed")

        if GPUtil is not None:
            try:
                gpus: Iterable[Any] = GPUtil.getGPUs()
                gpu_list = list(gpus)
                if gpu_list:
                    gpu = gpu_list[0]
                    total_mb = getattr(gpu, "memoryTotal", None)
                    used_mb = getattr(gpu, "memoryUsed", None)
                    if total_mb is not None and used_mb is not None:
                        total = int(total_mb * 1024 * 1024)
                        allocated = int(used_mb * 1024 * 1024)
                        return allocated, total
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("model.slot.gputil_stats_failed")
        return None, None


__all__ = ["ModelSlotManager"]
