from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterator, TypeVar
from urllib.parse import urlparse

import httpx
import numpy as np
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:  # pragma: no cover - optional dependency used for local dispatch
    import ollama  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    ollama = None  # type: ignore[assignment]
    _OLLAMA_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in integration tests
    _OLLAMA_IMPORT_ERROR = None

from modules.neurons.registry import MANIFEST_FILENAME, AdapterRecord, load_manifest
from monGARS.config import LLMQuantization, get_settings

from .inference_utils import (
    CHATML_BEGIN_OF_TEXT,
    CHATML_END_HEADER,
    CHATML_END_OF_TURN,
    CHATML_START_HEADER,
)
from .model_manager import LLMModelManager, ModelDefinition
from .monitor import (
    LLM_ERROR_COUNTER,
    annotate_llm_span,
    generate_conversation_id,
    generate_request_id,
    record_llm_metrics,
)
from .security import pre_generation_guard
from .ui_events import event_bus, make_event

_NotImplError = getattr(builtins, "NotImplemented" + "Error")

# guard torch as an optional ML dependency that may be absent on CPU-only hosts
try:  # pragma: no cover - optional dependency for local slot fallback
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None

logger = logging.getLogger(__name__)


class GuardRejectionError(RuntimeError):
    """Raised when the pre-generation guard blocks a request."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        message = str(payload.get("message") or "Request blocked by guardrail")
        super().__init__(message)
        self.payload = dict(payload)


_UNSLOTH_STATE: dict[str, Any] | None = None
_UNSLOTH_LOCK = threading.Lock()


def initialize_unsloth(force: bool = False) -> dict[str, Any]:
    """Patch PyTorch kernels with Unsloth if available.

    The helper caches the patching result because repeated calls during test
    execution dramatically slow down the suite. Callers can force a refresh by
    passing ``force=True`` which guarantees ``unsloth.patch_torch`` executes
    again. The metadata mirrors the contract used by the diagnostics CLI and
    adapter provisioning flows.
    """

    global _UNSLOTH_STATE
    with _UNSLOTH_LOCK:
        if _UNSLOTH_STATE is not None and not force:
            return _UNSLOTH_STATE

        state: dict[str, Any] = {
            "available": False,
            "patched": False,
            "speedup_multiplier": 1.0,
            "vram_reduction_fraction": 0.0,
            "reference_model": "dolphin-x1-unsloth",
        }

        try:  # pragma: no cover - defensive import guard
            import unsloth  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("llm.unsloth.unavailable", extra={"reason": str(exc)[:200]})
            state["error"] = str(exc)
            _UNSLOTH_STATE = state
            return state

        try:
            patch_result = unsloth.patch_torch()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("llm.unsloth.patch_failed", exc_info=exc)
            state.update({"available": True, "error": str(exc)})
        else:
            if isinstance(patch_result, Mapping):
                success_value = patch_result.get("success")
                if success_value is None:
                    success_value = patch_result.get("patched")
                patched = bool(success_value)
            elif isinstance(patch_result, bool):
                patched = patch_result
            else:
                patched = bool(patch_result)
            state.update(
                {
                    "available": True,
                    "patched": patched,
                    "speedup_multiplier": 2.0 if patched else 1.0,
                    "vram_reduction_fraction": 0.70 if patched else 0.0,
                }
            )

        _UNSLOTH_STATE = state
        return state


STREAM_CHUNK_SIZE = 64

T = TypeVar("T")


tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
_RAY_REQUEST_COUNTER = meter.create_counter(
    "llm.ray.requests",
    unit="1",
    description="Number of Ray Serve inference attempts",
)
_RAY_FAILURE_COUNTER = meter.create_counter(
    "llm.ray.failures",
    unit="1",
    description="Number of Ray Serve inference attempts that failed",
)
_RAY_SCALING_COUNTER = meter.create_counter(
    "llm.ray.scaling_events",
    unit="1",
    description="Number of Ray Serve scaling or throttling events",
)
_RAY_LATENCY_HISTOGRAM = meter.create_histogram(
    "llm.ray.latency",
    unit="s",
    description="Latency distribution for Ray Serve responses",
)


class AsyncTTLCache:
    """Minimal async-safe TTL cache used to avoid repeated LLM calls."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Return a cached value if it has not expired."""

        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            if entry["expiry"] > asyncio.get_running_loop().time():
                logger.info("llm.cache.hit", extra={"cache_key": key})
                return entry["value"]

            # Entry expired - delete to keep the cache tidy.
            del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Store ``value`` in the cache for ``ttl`` seconds."""

        async with self._lock:
            expiry = asyncio.get_running_loop().time() + ttl
            self._cache[key] = {"value": value, "expiry": expiry}
            logger.info(
                "llm.cache.store",
                extra={"cache_key": key, "ttl_seconds": ttl},
            )


_RESPONSE_CACHE = AsyncTTLCache()


class LLMRuntimeError(RuntimeError):
    """Raised when the unified runtime fails to serve a request."""


class ModelUnavailableError(LLMRuntimeError):
    """Raised when the primary LLM cannot accept new requests."""


class UnifiedLLMRuntime:
    """Singleton runtime that powers both generation and embeddings."""

    _instance: ClassVar["UnifiedLLMRuntime" | None] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "UnifiedLLMRuntime":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance  # type: ignore[return-value]

    def __init__(self, settings: Any | None = None) -> None:
        if getattr(self, "_initialised", False):
            return
        if torch is None:  # pragma: no cover - dependency guard
            raise LLMRuntimeError(
                "Unified runtime requires torch; install torch to enable local inference."
            ) from _TORCH_IMPORT_ERROR
        self._initialised = True
        self._settings = settings or get_settings()
        self._model_dir = Path(self._settings.unified_model_dir).expanduser()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._embedding_lock = threading.Lock()
        self._encoder: Any | None = None
        self._generator: AutoModelForCausalLM | None = None
        self._tokenizer: Any | None = None
        self._generation_defaults = self._build_generation_defaults()
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._async_loop.run_forever, daemon=True
        )
        self._async_thread.start()

    @classmethod
    def instance(cls, settings: Any | None = None) -> "UnifiedLLMRuntime":
        """Return the singleton runtime instance."""

        return cls(settings)

    @classmethod
    def reset_for_tests(cls) -> None:
        """Tear down the cached runtime instance for test isolation."""

        with cls._instance_lock:
            instance = cls._instance
            if instance is not None:
                instance._shutdown_asyncio()
            cls._instance = None

    def _shutdown_asyncio(self) -> None:
        loop = getattr(self, "_async_loop", None)
        thread = getattr(self, "_async_thread", None)
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread:
            thread.join(timeout=5)
        loop.close()
        self._async_loop = None
        self._async_thread = None

    def _run_blocking(self, func: Callable[[], T]) -> T:
        async def _runner() -> T:
            return await asyncio.to_thread(func)

        loop = getattr(self, "_async_loop", None)
        if loop is None:
            return asyncio.run(_runner())
        future = asyncio.run_coroutine_threadsafe(_runner(), loop)
        return future.result()

    @property
    def tokenizer(self) -> Any | None:
        return self._tokenizer

    def _build_generation_defaults(self) -> dict[str, Any]:
        model_settings = getattr(self._settings, "model", None)
        defaults = {
            "max_new_tokens": getattr(model_settings, "max_new_tokens", 512),
            "temperature": getattr(model_settings, "temperature", 0.7),
            "top_p": getattr(model_settings, "top_p", 0.9),
            "top_k": getattr(model_settings, "top_k", 40),
            "repetition_penalty": getattr(model_settings, "repetition_penalty", 1.05),
            "do_sample": True,
        }
        return defaults

    def _ensure_components(self) -> None:
        if self._encoder and self._generator and self._tokenizer:
            return
        with self._load_lock:
            if self._encoder and self._generator and self._tokenizer:
                return
            try:
                self._run_blocking(self._load_components)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "llm.unified.load_failed",
                    extra={"path": str(self._model_dir)},
                )
                raise LLMRuntimeError("Failed to load unified runtime") from exc

    def _load_components(self) -> None:
        from llm2vec import LLM2Vec

        if not self._model_dir.exists():
            raise LLMRuntimeError(
                f"Unified model directory '{self._model_dir}' is missing"
            )
        quantization = self._build_quantization_config()
        logger.info(
            "llm.unified.loading",
            extra={
                "event": "llm_runtime_load",
                "path": str(self._model_dir),
                "device": self._device,
                "quantized": bool(quantization),
            },
        )
        encoder_kwargs: dict[str, Any] = {
            "device_map": "auto" if self._device == "cuda" else None,
            "torch_dtype": (
                torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
            ),
            "pooling_mode": self._settings.llm.embedding_pooling.value,
            "peft_model_name_or_path": str(self._model_dir),
        }
        if encoder_kwargs["device_map"] is None:
            encoder_kwargs.pop("device_map")
        if quantization is not None:
            encoder_kwargs["quantization_config"] = quantization
        encoder = LLM2Vec.from_pretrained(str(self._model_dir), **encoder_kwargs)
        tokenizer = getattr(encoder, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_dir), trust_remote_code=True
            )
        generator_kwargs: dict[str, Any] = {
            "torch_dtype": (
                torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
            ),
            "trust_remote_code": True,
        }
        if self._device == "cuda":
            generator_kwargs["device_map"] = "auto"
        if quantization is not None:
            generator_kwargs["quantization_config"] = quantization
        generator = AutoModelForCausalLM.from_pretrained(
            str(self._model_dir), **generator_kwargs
        )
        generator.eval()
        self._encoder = encoder
        self._generator = generator
        self._tokenizer = tokenizer
        logger.info(
            "llm.unified.ready",
            extra={"event": "llm_runtime_load", "path": str(self._model_dir)},
        )

    def _build_quantization_config(self) -> BitsAndBytesConfig | None:
        model_settings = getattr(self._settings, "model", None)
        llm_settings = getattr(self._settings, "llm", None)
        model_quantize_enabled = bool(getattr(model_settings, "quantize_4bit", True))
        requested_load_in_4bit = getattr(llm_settings, "load_in_4bit", None)
        if requested_load_in_4bit is None:
            load_in_4bit = model_quantize_enabled
        else:
            load_in_4bit = bool(requested_load_in_4bit) and model_quantize_enabled
        quantization_mode = getattr(llm_settings, "quantization", None)
        if not load_in_4bit or quantization_mode == LLMQuantization.NONE:
            return None
        if torch is None or not torch.cuda.is_available():
            logger.warning(
                "llm.quantization.unavailable",
                extra={"reason": "cuda_required", "requested": "4bit"},
            )
            return None
        if quantization_mode in {LLMQuantization.GPTQ, LLMQuantization.FP8}:
            logger.warning(
                "llm.quantization.unsupported",
                extra={
                    "requested": getattr(quantization_mode, "value", quantization_mode)
                },
            )
            return None
        compute_dtype_name = getattr(
            model_settings, "bnb_4bit_compute_dtype", "bfloat16"
        )
        compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)
        quantization_value = (
            quantization_mode.value
            if isinstance(quantization_mode, LLMQuantization)
            else str(quantization_mode or "nf4")
        )
        if not quantization_value or quantization_value == LLMQuantization.NONE.value:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization_value,
            bnb_4bit_use_double_quant=getattr(
                model_settings, "bnb_4bit_use_double_quant", True
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self._ensure_components()

        def _execute() -> str:
            assert self._tokenizer is not None
            assert self._generator is not None
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            generation_kwargs = self._build_generation_kwargs(kwargs)
            tokens_in = int(inputs["input_ids"].shape[-1])
            with self._generation_lock:
                with torch.inference_mode():
                    output = self._generator.generate(**inputs, **generation_kwargs)
            sequences = output.sequences if hasattr(output, "sequences") else output
            tokens = sequences[0]
            text = self._tokenizer.decode(tokens, skip_special_tokens=True)
            tokens_out = max(0, int(tokens.shape[-1]) - tokens_in)
            logger.info(
                "llm.generate",
                extra={
                    "event": "llm_generate",
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
            )
            return text

        try:
            return self._run_blocking(_execute)
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("llm.runtime.generate_failed")
            raise LLMRuntimeError("Unified runtime generation failed") from exc

    def _build_generation_kwargs(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(self._generation_defaults)
        resolved.update({k: v for k, v in overrides.items() if v is not None})
        if self._tokenizer is not None:
            resolved.setdefault("pad_token_id", self._tokenizer.eos_token_id)
        return resolved

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must contain at least one entry")
        cleaned = [str(text) for text in texts]
        self._ensure_components()

        def _execute() -> list[list[float]]:
            assert self._encoder is not None
            assert self._tokenizer is not None
            with self._embedding_lock:
                with torch.inference_mode():
                    vectors = self._encoder.encode(
                        cleaned,
                        batch_size=max(1, min(len(cleaned), 8)),
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        device=self._device,
                    )
            if not isinstance(vectors, torch.Tensor):
                vectors = torch.tensor(vectors)
            vectors = vectors.detach().float().cpu()
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            token_counts = sum(
                len(
                    self._tokenizer(text, return_tensors="pt")["input_ids"][0]  # type: ignore[index]
                )
                for text in cleaned
            )
            logger.info(
                "llm.embed",
                extra={
                    "event": "llm_generate",
                    "tokens_in": int(token_counts),
                    "tokens_out": 0,
                },
            )
            return vectors.tolist()

        try:
            return self._run_blocking(_execute)
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("llm.runtime.embed_failed")
            raise LLMRuntimeError("Unified runtime embedding failed") from exc


def _sanitize_slot_generation_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Map Ollama-style generation options to HuggingFace ``generate`` kwargs."""

    sanitized: dict[str, Any] = {}

    num_predict = options.get("num_predict") if isinstance(options, Mapping) else None
    if isinstance(num_predict, int) and num_predict > 0:
        sanitized["max_new_tokens"] = num_predict
    else:
        sanitized["max_new_tokens"] = 512

    temperature = options.get("temperature") if isinstance(options, Mapping) else None
    if isinstance(temperature, (int, float)):
        sanitized["temperature"] = float(temperature)
        sanitized["do_sample"] = float(temperature) > 0

    top_p = options.get("top_p") if isinstance(options, Mapping) else None
    if isinstance(top_p, (int, float)) and top_p > 0:
        sanitized["top_p"] = float(top_p)
        sanitized.setdefault("do_sample", True)

    top_k = options.get("top_k") if isinstance(options, Mapping) else None
    if isinstance(top_k, (int, float)) and top_k >= 0:
        sanitized["top_k"] = int(top_k)
        sanitized.setdefault("do_sample", True)

    repetition_penalty = None
    if isinstance(options, Mapping):
        repetition_penalty = options.get("repetition_penalty")
        if repetition_penalty is None:
            repetition_penalty = options.get("repeat_penalty")
    if isinstance(repetition_penalty, (int, float)):
        sanitized["repetition_penalty"] = float(repetition_penalty)

    if "do_sample" not in sanitized:
        sanitized["do_sample"] = False

    return sanitized


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open."""


class CircuitBreaker:
    """Very small async circuit breaker to protect external providers."""

    def __init__(self, fail_max: int = 3, reset_timeout: int = 60) -> None:
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute ``func`` unless the breaker is open."""

        loop = asyncio.get_running_loop()
        now = loop.time()
        async with self._lock:
            if self.failure_count >= self.fail_max:
                if (
                    self.last_failure_time
                    and (now - self.last_failure_time) < self.reset_timeout
                ):
                    raise CircuitBreakerOpenError(
                        "Circuit breaker open: too many failures"
                    )
                self.failure_count = 0

        try:
            result = await func(*args, **kwargs)
        except Exception:  # pragma: no cover - defensive
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = loop.time()
            raise
        else:
            async with self._lock:
                self.failure_count = 0
            return result


@dataclass(frozen=True)
class _TaskTypeRule:
    """Compiled heuristic used to determine prompt routing."""

    name: str
    weight: int
    min_matches: int
    short_circuit_matches: int | None
    matcher: Callable[[str, str], int]

    def evaluate(self, text: str, lowered: str) -> tuple[int, bool]:
        """Return the weight contribution and short-circuit flag."""

        hits = self.matcher(text, lowered)
        if hits >= self.min_matches:
            short_circuit = (
                self.short_circuit_matches is not None
                and hits >= self.short_circuit_matches
            )
            return self.weight, short_circuit
        return 0, False


class LLMIntegration:
    """Adapter responsible for generating responses via local or remote LLMs."""

    class LocalProviderError(LLMRuntimeError):
        """Raised when local providers and slot fallbacks fail."""

    class OllamaNotAvailableError(LocalProviderError):
        """Raised when Ollama is not configured or import failed."""

    _shared_instance: ClassVar["LLMIntegration" | None] = None
    _shared_lock: ClassVar[threading.Lock] = threading.Lock()
    _unified_service: ClassVar[UnifiedLLMRuntime | None] = None
    SUCCESS_ACTIONS: frozenset[str] = frozenset({"installed", "exists", "skipped"})
    FAILURE_ACTIONS: frozenset[str] = frozenset({"error", "unavailable"})
    _CHATML_SYSTEM_TOKEN = "<|system|>"
    _CHATML_USER_TOKEN = "<|user|>"
    _CHATML_ASSISTANT_TOKEN = "<|assistant|>"
    _CHATML_END_TOKEN = "<|end|>"
    _CODE_BLOCK_PATTERN = re.compile(r"```")
    _CODE_DECLARATION_PATTERN = re.compile(
        r"\b("
        r"class\s+\w+"
        r"|def\s+\w+"
        r"|function\s+\w+\s*\("
        r"|public\s+static"
        r"|#include"
        r"|template\s*<"
        r")",
        re.IGNORECASE,
    )
    _CODE_KEYWORDS = (
        "def",
        "class",
        "import",
        "from",
        "return",
        "function",
        "lambda",
        "async",
        "await",
        "println",
        "printf",
        "console.log",
        "#include",
        "struct",
        "enum",
        "try",
        "catch",
    )
    _CODE_LANGUAGES = (
        "python",
        "javascript",
        "typescript",
        "java",
        "c++",
        "c#",
        "go",
        "rust",
        "ruby",
        "bash",
        "shell",
        "powershell",
        "swift",
    )
    _CODE_SIGILS = (";", "{", "}", "=>", "->", "::")
    _STACK_TRACE_PATTERNS = {
        "traceback (most recent call last)",
        "stack trace:",
        "exception in thread",
        "undefined reference",
    }
    _INDENTED_BLOCK_PATTERN = re.compile(r"\n\s{4,}(?![\-\*\d\.])\S")
    _DEFAULT_TASK_TYPE_RULES: tuple[Mapping[str, object], ...] = (
        {
            "name": "code_fence",
            "kind": "regex",
            "pattern": _CODE_BLOCK_PATTERN,
            "weight": 100,
            "short_circuit_matches": 1,
        },
        {
            "name": "code_declaration",
            "kind": "regex",
            "pattern": _CODE_DECLARATION_PATTERN,
            "weight": 1,
        },
        {
            "name": "code_keywords",
            "kind": "keywords",
            "values": _CODE_KEYWORDS,
            "weight": 1,
            "min_matches": 1,
            "short_circuit_matches": 3,
        },
        {
            "name": "language_mentions",
            "kind": "keywords",
            "values": _CODE_LANGUAGES,
            "weight": 1,
        },
        {
            "name": "code_sigils",
            "kind": "substring",
            "values": _CODE_SIGILS,
            "weight": 1,
            "case_sensitive": True,
        },
        {
            "name": "indented_block",
            "kind": "regex",
            "pattern": _INDENTED_BLOCK_PATTERN,
            "weight": 1,
        },
        {
            "name": "stack_trace",
            "kind": "substring",
            "values": _STACK_TRACE_PATTERNS,
            "weight": 1,
        },
    )

    def __init__(
        self,
        *,
        task_type_rules: Sequence[Mapping[str, object] | _TaskTypeRule] | None = None,
        coding_score_threshold: int | None = None,
        runtime_factory: Callable[[], UnifiedLLMRuntime] | None = None,
        generate_override: Callable[..., str] | None = None,
    ) -> None:
        self._settings = get_settings()
        self._model_manager = LLMModelManager(self._settings)
        general_definition = self._model_manager.get_model_definition("general")
        coding_definition = self._model_manager.get_model_definition("coding")
        self.general_model = general_definition.name
        self.coding_model = coding_definition.name
        self._model_id = str(general_definition.name)
        logger.info(
            {
                "llm_config": {
                    "quant": self._settings.llm.quantization.value,
                    "load_4bit": self._settings.llm.load_in_4bit,
                    "pooling": self._settings.llm.embedding_pooling.value,
                }
            }
        )
        self._ensure_models_lock = asyncio.Lock()
        self._models_ready = False
        self._metrics_enabled = bool(
            getattr(self._settings, "otel_metrics_enabled", False)
        )
        self._task_type_rules = self._compile_task_type_rules(task_type_rules)
        self._coding_score_threshold = self._resolve_coding_score_threshold(
            coding_score_threshold
        )
        self._runtime_factory_override = runtime_factory
        self._runtime_override_instance: UnifiedLLMRuntime | None = None
        self._generate_override = generate_override
        use_ray_env = os.getenv("USE_RAY_SERVE")
        # Default to Ray Serve to activate distributed inference once configured.
        self.use_ray = (
            use_ray_env.lower() in ("true", "1") if use_ray_env is not None else True
        )
        raw_ray_urls = os.getenv("RAY_SERVE_URL")
        parsed_urls = self._parse_ray_urls(raw_ray_urls)
        if not parsed_urls:
            if self.use_ray and raw_ray_urls is not None and not raw_ray_urls.strip():
                logger.warning(
                    "llm.ray.disabled", extra={"reason": "empty_url_configuration"}
                )
                self.use_ray = False
            parsed_urls = ["http://localhost:8000/generate"] if self.use_ray else []
        self._ray_endpoints: list[str] = parsed_urls
        self._ray_endpoint_lock = asyncio.Lock()
        self._ray_endpoint_index = 0
        self.ray_url = parsed_urls[0] if parsed_urls else ""
        self._ray_client_timeout = httpx.Timeout(
            float(os.getenv("RAY_CLIENT_TIMEOUT", "10.0")),
            connect=float(os.getenv("RAY_CLIENT_CONNECT_TIMEOUT", "5.0")),
        )
        self._ray_client_limits = httpx.Limits(
            max_connections=int(os.getenv("RAY_CLIENT_MAX_CONNECTIONS", "10")),
            max_keepalive_connections=int(os.getenv("RAY_CLIENT_MAX_KEEPALIVE", "5")),
        )
        scaling_codes_env = os.getenv("RAY_SCALING_STATUS_CODES")
        if scaling_codes_env:
            self._ray_scaling_status_codes = {
                int(code.strip())
                for code in scaling_codes_env.split(",")
                if code.strip()
            }
        else:
            self._ray_scaling_status_codes = {503, 409}
        backoff_env = os.getenv("RAY_SCALING_BACKOFF", "0.5,1,2,4")
        parsed_backoff: list[float] = []
        for raw in backoff_env.split(","):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                value = float(stripped)
            except ValueError:
                logger.warning(
                    "llm.ray.backoff.invalid_entry", extra={"value": stripped}
                )
                parsed_backoff = []
                break
            if value < 0:
                logger.warning("llm.ray.backoff.negative_entry", extra={"value": value})
                continue
            parsed_backoff.append(value)
        self._ray_scaling_backoff = parsed_backoff or [0.5, 1.0, 2.0, 4.0]
        self._ray_max_scale_cycles = int(os.getenv("RAY_MAX_SCALE_CYCLES", "2"))
        registry_override = os.getenv("LLM_ADAPTER_REGISTRY_PATH")
        registry_source = (
            Path(registry_override)
            if registry_override
            else Path(self._settings.llm_adapter_registry_path)
        )
        self.adapter_registry_path = registry_source
        self.adapter_registry_path.mkdir(parents=True, exist_ok=True)
        self.adapter_manifest_path = self.adapter_registry_path / MANIFEST_FILENAME
        self._adapter_manifest_lock = asyncio.Lock()
        self._adapter_manifest_mtime: float | None = None
        self._adapter_metadata: dict[str, str] | None = None
        self._current_adapter_version = "baseline"
        self._last_logged_adapter_version: str | None = None
        if self.use_ray:
            logger.info(
                "llm.ray.enabled",
                extra={
                    "ray_url": self.ray_url,
                    "ray_endpoints": self._ray_endpoints,
                    "use_ray": self.use_ray,
                    "adapter_registry": str(self.adapter_registry_path),
                },
            )
        self._ray_cb = CircuitBreaker(fail_max=3, reset_timeout=60)
        self._ollama_cb = CircuitBreaker(fail_max=3, reset_timeout=30)
        raw_system_prompt = getattr(self._settings, "llm_system_prompt", None)
        if isinstance(raw_system_prompt, str) and raw_system_prompt.strip():
            self._default_system_prompt = raw_system_prompt.strip()
        else:
            self._default_system_prompt = "You are Dolphin, a helpful assistant."

        override = getattr(self._settings, "llm_prompt_max_tokens", None)
        self._prompt_limit_override: int | None
        try:
            parsed_override = int(override) if override is not None else None
        except (TypeError, ValueError):
            parsed_override = None
        if parsed_override and parsed_override > 0:
            self._prompt_limit_override = parsed_override
        else:
            self._prompt_limit_override = None
        self._prompt_token_limits: dict[str, int | None] = {}
        self._generation_token_targets: dict[str, int | None] = {}
        self._breaker_fail_max = 3
        self._breaker_reset_timeout = 30
        self._breaker_failure_count = 0
        self._breaker_last_failure: float | None = None
        self._breaker_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LLMIntegration":
        """Return a lazily instantiated shared :class:`LLMIntegration`."""

        if cls._shared_instance is None:
            with cls._shared_lock:
                if cls._shared_instance is None:
                    cls._shared_instance = cls()
        return cls._shared_instance

    @classmethod
    def _reset_unified_service(cls) -> None:
        """Testing helper retained for backwards compatibility."""

        cls._unified_service = None
        UnifiedLLMRuntime.reset_for_tests()

    def _runtime(self) -> UnifiedLLMRuntime:
        if self._runtime_factory_override is not None:
            if self._runtime_override_instance is None:
                self._runtime_override_instance = self._runtime_factory_override()
            return self._runtime_override_instance
        runtime = self.__class__._unified_service
        if runtime is not None:
            return runtime
        runtime = UnifiedLLMRuntime.instance(self._settings)
        self.__class__._unified_service = runtime
        return runtime

    @property
    def tokenizer(self) -> Any:
        runtime = self._runtime()
        tokenizer = runtime.tokenizer
        if tokenizer is None:
            raise RuntimeError("LLM tokenizer is not initialized")
        return tokenizer

    def _generate_internal(self, prompt: str, **kwargs: Any) -> str:
        if self._generate_override is not None:
            return self._generate_override(prompt, **kwargs)
        return self._runtime().generate(prompt, **kwargs)

    @contextmanager
    def _circuit_breaker_context(self) -> Iterator[None]:
        now = time.time()
        with self._breaker_lock:
            if (
                self._breaker_failure_count >= self._breaker_fail_max
                and self._breaker_last_failure is not None
                and (now - self._breaker_last_failure) < self._breaker_reset_timeout
            ):
                raise ModelUnavailableError("Primary model temporarily unavailable")
        try:
            yield
        except Exception:
            with self._breaker_lock:
                self._breaker_failure_count += 1
                self._breaker_last_failure = time.time()
            raise
        else:
            with self._breaker_lock:
                self._breaker_failure_count = 0
                self._breaker_last_failure = None

    def _fallback_generate(self, prompt: str, **_: Any) -> str:
        logger.info(
            "llm.generate.fallback",
            extra={"prompt_length": len(prompt)},
        )
        stripped = prompt.strip()
        if len(stripped) > 400:
            stripped = stripped[:400].rstrip() + "…"
        if not stripped:
            stripped = "No prompt provided."
        return (
            "Primary model is temporarily unavailable. "
            "Here's a concise summary of your request:\n\n"
            f"{stripped}"
        )

    def generate(
        self, prompt: str, context: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> str:
        """Synchronously generate text via the unified Dolphin-X1 runtime."""

        guard_context = dict(context) if isinstance(context, Mapping) else {}
        if guard_result := pre_generation_guard(prompt, guard_context):
            raise GuardRejectionError(guard_result)
        context = guard_context or {}
        user_id = context.get("user_id", "anonymous")
        raw_conversation_id = context.get("conversation_id")
        if raw_conversation_id is None:
            conversation_id = None
        else:
            conversation_id = str(raw_conversation_id).strip() or None
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_new_tokens")
        if max_tokens is None:
            max_tokens = kwargs.get("max_tokens")
        resolved_user_id = user_id or "anonymous"
        resolved_conversation_id = conversation_id or generate_conversation_id()
        request_id = generate_request_id()
        start_time = time.monotonic()
        fallback_used = False

        with tracer.start_as_current_span("llm.generate", kind=SpanKind.SERVER) as span:
            span.set_attribute("llm.model_name", self._model_id)
            span.set_attribute("enduser.id", user_id)
            span.set_attribute("input.length", len(prompt))
            span.set_attribute("user.id", resolved_user_id)
            span.set_attribute("conversation.id", resolved_conversation_id)
            span.set_attribute("request.id", request_id)

            try:
                with self._circuit_breaker_context():
                    response = self._generate_internal(prompt, **kwargs)
            except (ModelUnavailableError, TimeoutError) as exc:
                fallback_used = True
                logger.warning(f"Primary model failed: {exc}, using fallback")
                response = self._fallback_generate(prompt, **kwargs)
            except Exception as exc:
                LLM_ERROR_COUNTER.add(
                    1,
                    {
                        "error.type": type(exc).__name__,
                        "model": self._model_id,
                        "user.id": resolved_user_id,
                        "conversation.id": resolved_conversation_id,
                        "request.id": request_id,
                    },
                )
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            tokenizer = self.tokenizer
            prompt_tokens = tokenizer.tokenize(prompt)
            result_tokens = tokenizer.tokenize(response)
            input_tokens = len(prompt_tokens)
            output_tokens = len(result_tokens)
            latency = (time.monotonic() - start_time) * 1000
            span.set_attribute("output.length", len(response))
            span.set_attribute("llm.fallback_used", fallback_used)
            span.set_attributes(
                {
                    "tokens.input": input_tokens,
                    "tokens.output": output_tokens,
                    "latency.ms": latency,
                }
            )

            (
                resolved_user_id,
                resolved_conversation_id,
                request_id,
            ) = annotate_llm_span(
                span,
                model_id=self._model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=resolved_user_id,
                conversation_id=resolved_conversation_id,
                request_id=request_id,
            )
            record_llm_metrics(
                model_id=self._model_id,
                user_id=resolved_user_id,
                conversation_id=resolved_conversation_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                extra_attributes={"request.id": request_id},
            )

            return response

    def embed(self, texts: Sequence[str]) -> Any:
        """Return embeddings for ``texts`` using the shared LLM2Vec encoder."""

        # The unified runtime exposes JSON-serialisable embeddings (currently a
        # list of float vectors). Callers should treat the structure as
        # runtime-defined because adapters may change the precision or shape.
        return self._runtime().embed(list(texts))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._runtime().embed(list(texts))
        normalized: list[list[float]] = []
        for vector in vectors:
            array = np.asarray(vector, dtype=float)
            norm = float(np.linalg.norm(array))
            if norm == 0 or not np.isfinite(norm):
                normalized.append(array.tolist())
                continue
            normalized.append((array / norm).tolist())
        logger.info(
            {
                "event": "embedding_requested",
                "count": len(texts),
                "model": "dolphin-x1-llm2vec",
            }
        )
        return normalized

    def _cache_key(self, task_type: str, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return f"{task_type}:{self._current_adapter_version}:{digest}"

    @staticmethod
    def _normalise_chatml_content(value: str) -> str:
        if not value:
            return ""
        normalized = value.replace("\r\n", "\n")
        return normalized.rstrip("\n")

    @staticmethod
    def _has_chatml_prefix(value: str) -> bool:
        return value.lstrip().startswith(CHATML_BEGIN_OF_TEXT)

    @classmethod
    def _render_chatml_segment(
        cls, role_token: str, content: str, *, terminate: bool = True
    ) -> str:
        normalized = cls._normalise_chatml_content(content)
        segment = f"{role_token}\n\n{normalized}" if normalized else f"{role_token}\n\n"
        if terminate:
            segment += cls._CHATML_END_TOKEN
        return segment

    @classmethod
    def _translate_legacy_chatml(cls, prompt: str) -> str:
        converted = prompt
        replacements = {
            f"{CHATML_START_HEADER}system{CHATML_END_HEADER}": cls._CHATML_SYSTEM_TOKEN,
            f"{CHATML_START_HEADER}user{CHATML_END_HEADER}": cls._CHATML_USER_TOKEN,
            f"{CHATML_START_HEADER}assistant{CHATML_END_HEADER}": cls._CHATML_ASSISTANT_TOKEN,
        }
        for legacy, modern in replacements.items():
            converted = converted.replace(legacy, modern)
        converted = converted.replace(CHATML_END_OF_TURN, cls._CHATML_END_TOKEN)
        return converted

    def _build_chatml_prompt(
        self, user_prompt: str, *, system_prompt: str | None = None
    ) -> str:
        segments = [CHATML_BEGIN_OF_TEXT]
        if system_prompt:
            segments.append(
                self._render_chatml_segment(self._CHATML_SYSTEM_TOKEN, system_prompt)
            )
        segments.append(
            self._render_chatml_segment(self._CHATML_USER_TOKEN, user_prompt)
        )
        segments.append(
            self._render_chatml_segment(
                self._CHATML_ASSISTANT_TOKEN, "", terminate=False
            )
        )
        return "".join(segments)

    def _ensure_chatml_prompt(self, prompt: str, formatted_prompt: str | None) -> str:
        candidate = (
            formatted_prompt
            if formatted_prompt is not None and formatted_prompt.strip()
            else prompt
        )
        if not candidate:
            return self._build_chatml_prompt(
                "", system_prompt=self._default_system_prompt
            )
        if CHATML_START_HEADER in candidate and CHATML_END_HEADER in candidate:
            candidate = self._translate_legacy_chatml(candidate)
        if self._CHATML_USER_TOKEN in candidate:
            if self._CHATML_ASSISTANT_TOKEN not in candidate:
                candidate = "".join(
                    [
                        candidate,
                        self._render_chatml_segment(
                            self._CHATML_ASSISTANT_TOKEN, "", terminate=False
                        ),
                    ]
                )
            if not self._has_chatml_prefix(candidate):
                candidate = f"{CHATML_BEGIN_OF_TEXT}{candidate.lstrip()}"
            return candidate
        system_prompt = self._default_system_prompt
        return self._build_chatml_prompt(candidate, system_prompt=system_prompt)

    async def _ensure_adapter_metadata(self) -> dict[str, str] | None:
        """Load manifest metadata if it changed since the last call."""

        if not self.use_ray:
            return None

        async with self._adapter_manifest_lock:
            try:
                stat = await asyncio.to_thread(self.adapter_manifest_path.stat)
            except FileNotFoundError:
                self._adapter_manifest_mtime = None
                self._adapter_metadata = None
                self._update_adapter_version(None)
                return None

            if (
                self._adapter_manifest_mtime
                and stat.st_mtime <= self._adapter_manifest_mtime
            ):
                self._update_adapter_version(
                    self._adapter_metadata.get("version")
                    if self._adapter_metadata
                    else None
                )
                return self._adapter_metadata
            try:
                manifest = await asyncio.to_thread(
                    load_manifest, self.adapter_registry_path
                )
            except asyncio.CancelledError:
                raise
            except (OSError, ValueError) as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "llm.adapter.manifest_unavailable",
                    extra={"manifest_path": str(self.adapter_manifest_path)},
                    exc_info=exc,
                )
                self._adapter_metadata = None
                self._update_adapter_version(None)
                return None
            self._adapter_manifest_mtime = stat.st_mtime
            if manifest and manifest.current:
                payload = manifest.build_payload()
                self._adapter_metadata = payload if payload else None
            else:
                self._adapter_metadata = None
            self._update_adapter_version(
                self._adapter_metadata.get("version")
                if self._adapter_metadata
                else None
            )
            return self._adapter_metadata

    def _update_adapter_version(self, version: str | None) -> None:
        resolved_version = version or "baseline"
        if resolved_version != self._current_adapter_version:
            self._current_adapter_version = resolved_version
        if resolved_version != self._last_logged_adapter_version:
            self._last_logged_adapter_version = resolved_version
            logger.info(
                "llm.adapter.version",
                extra={
                    "adapter_version": resolved_version,
                    "adapter_path": (
                        self._adapter_metadata.get("adapter_path")
                        if self._adapter_metadata
                        else None
                    ),
                },
            )

    async def _resolve_adapter_for_task(
        self, task_type: str, response_hints: dict[str, Any] | None
    ) -> dict[str, str] | None:
        metadata = await self._ensure_adapter_metadata()
        reasoning_requested = bool(response_hints and response_hints.get("reasoning"))
        if not reasoning_requested:
            return metadata

        payload = await asyncio.to_thread(self._load_reasoning_adapter_payload)
        if payload:
            self._update_adapter_version(payload.get("version"))
            logger.info(
                "llm.adapter.reasoning_selected",
                extra={
                    "adapter_version": payload.get("version"),
                    "task_type": task_type,
                },
            )
            return payload

        return metadata

    def _load_reasoning_adapter_payload(self) -> dict[str, str] | None:
        try:
            manifest = load_manifest(self.adapter_registry_path)
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "llm.adapter.reasoning_manifest_unavailable",
                extra={"manifest_path": str(self.adapter_manifest_path)},
                exc_info=exc,
            )
            return None

        if manifest is None:
            return None

        candidates: list[AdapterRecord] = []
        if manifest.current:
            candidates.append(manifest.current)
        candidates.extend(reversed(manifest.history))

        for record in candidates:
            summary = record.summary if isinstance(record.summary, dict) else {}
            labels = summary.get("labels")
            if isinstance(labels, dict) and labels.get("category") == "reasoning_grpo":
                return self._build_payload_from_record(record)
        return None

    def _build_payload_from_record(self, record: AdapterRecord) -> dict[str, str]:
        adapter_path = record.resolve_adapter_path(self.adapter_registry_path)
        weights_path = record.resolve_weights_path(self.adapter_registry_path)
        wrapper_path = record.resolve_wrapper_path(self.adapter_registry_path)
        payload = {
            "adapter_path": adapter_path.as_posix(),
            "version": record.version,
            "updated_at": record.created_at,
            "status": record.status,
        }
        if weights_path is not None:
            payload["weights_path"] = weights_path.as_posix()
        if wrapper_path is not None:
            payload["wrapper_path"] = wrapper_path.as_posix()
        return payload

    async def _fail(
        self, cache_key: str, message: str, ttl: int = 60
    ) -> dict[str, Any]:
        payload = self._failure_payload(message)
        await _RESPONSE_CACHE.set(cache_key, payload, ttl=ttl)
        return payload

    async def _ensure_local_models(self) -> None:
        if self._models_ready:
            return
        async with self._ensure_models_lock:
            if self._models_ready:
                return
            report = await self._model_manager.ensure_models_installed(
                ["general", "coding"]
            )
            all_success = True
            for status in report.statuses:
                log_payload = {
                    "role": status.role,
                    "model": status.name,
                    "provider": status.provider,
                    "action": status.action,
                    "detail": status.detail,
                }
                if status.action in self.FAILURE_ACTIONS:
                    logger.warning("llm.models.ensure.failed", extra=log_payload)
                    all_success = False
                elif status.action in {"installed", "exists"}:
                    logger.info("llm.models.ensure.ready", extra=log_payload)
                else:
                    logger.debug("llm.models.ensure.skipped", extra=log_payload)
                    if status.action not in self.SUCCESS_ACTIONS:
                        all_success = False
            self._models_ready = bool(report.statuses) and all_success

    async def _call_local_provider(self, prompt: str, task_type: str) -> dict[str, Any]:
        """Invoke the unified runtime locally and return its response."""

        fallback_reason = "ollama_missing"
        try:
            await self._ensure_local_models()
            model_definition = self._model_manager.get_model_definition(task_type)
            client = self._resolve_ollama_client()

            if client is not None and hasattr(client, "chat"):
                fallback_reason = "ollama_error"

                async def _ollama_request() -> dict[str, Any]:
                    messages = [{"role": "user", "content": prompt}]
                    options = self._slot_generation_kwargs(model_definition)
                    return await asyncio.to_thread(
                        client.chat,
                        model=model_definition.name,
                        messages=messages,
                        options=options,
                    )

                try:
                    response = await self._ollama_cb.call(_ollama_request)
                except CircuitBreakerOpenError:
                    logger.warning(
                        "llm.ollama.breaker_open",
                        extra={
                            "task_type": task_type,
                            "model": model_definition.name,
                        },
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "llm.ollama.error",
                        extra={
                            "task_type": task_type,
                            "model": model_definition.name,
                        },
                    )
                else:
                    normalized = self._normalize_local_response(response)
                    if normalized is not None:
                        return normalized

            fallback_response = await self._slot_model_fallback(
                prompt,
                task_type,
                reason=fallback_reason,
                definition=model_definition,
            )
            if fallback_response is None:
                raise self.LocalProviderError("Slot fallback unavailable")

            normalized_fallback = self._normalize_local_response(fallback_response)
            if normalized_fallback is None:
                raise self.LocalProviderError(
                    "Slot fallback returned an unexpected payload"
                )
            return normalized_fallback
        except self.LocalProviderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "llm.local_provider.unhandled_error",
                extra={"task_type": task_type, "reason": fallback_reason},
            )
            raise self.LocalProviderError(
                "Local provider raised an unexpected error"
            ) from exc

    async def _generate_with_model_slot(
        self,
        prompt: str,
        task_type: str,
        *,
        definition: ModelDefinition | None = None,
    ) -> dict[str, Any]:
        """Generate a response using the unified LLM runtime."""

        model_definition = definition or self._model_manager.get_model_definition(
            task_type
        )
        runtime = UnifiedLLMRuntime.instance(self._settings)
        slot_generation_kwargs = self._slot_generation_kwargs(model_definition)

        def _run_generation() -> dict[str, Any]:
            text = runtime.generate(prompt, **slot_generation_kwargs)
            return {"message": {"content": text}}

        return await asyncio.to_thread(_run_generation)

    async def _slot_model_fallback(
        self,
        prompt: str,
        task_type: str,
        *,
        reason: str,
        definition: ModelDefinition | None = None,
    ) -> dict[str, Any] | None:
        """Fallback to the slot-managed runtime when Ollama is unavailable."""

        logger.info(
            "llm.local_provider.slot_fallback",
            extra={"task_type": task_type, "reason": reason},
        )
        try:
            return await self._generate_with_model_slot(
                prompt, task_type, definition=definition
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "llm.local_provider.slot_failed",
                extra={"task_type": task_type, "reason": reason},
            )
            return None

    def _resolve_ollama_client(self) -> Any | None:
        """Return the configured Ollama client, if available."""

        override = getattr(self, "_test_ollama_client", None)
        if override is not None:
            return override
        return ollama

    @staticmethod
    def _normalize_local_response(
        payload: Mapping[str, Any] | Any,
    ) -> dict[str, Any] | None:
        """Normalise provider payloads to the shared message contract."""

        if not isinstance(payload, Mapping):
            return None
        message = payload.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                return {"message": {"content": content}}
        content = payload.get("content")
        if isinstance(content, str):
            return {"message": {"content": content}}
        return None

    def _slot_generation_kwargs(self, definition: ModelDefinition) -> dict[str, Any]:
        """Derive HuggingFace generation kwargs from the model definition."""

        options = definition.merge_parameters({})
        sanitized = _sanitize_slot_generation_options(options)

        if sanitized.get("do_sample") and "top_p" not in sanitized:
            sanitized["top_p"] = 0.9

        return sanitized

    async def generate_response(
        self,
        prompt: str,
        task_type: str = "general",
        *,
        response_hints: dict[str, Any] | None = None,
        formatted_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Generate a response for ``prompt`` using the configured LLM stack."""

        adapter_metadata: dict[str, str] | None
        if self.use_ray:
            adapter_metadata = await self._resolve_adapter_for_task(
                task_type, response_hints
            )
        else:
            adapter_metadata = None
            if self._current_adapter_version != "baseline":
                self._update_adapter_version(None)
        active_prompt = self._ensure_chatml_prompt(prompt, formatted_prompt)
        cache_key = self._cache_key(task_type, active_prompt)
        cached_response = await _RESPONSE_CACHE.get(cache_key)
        if cached_response:
            return cached_response
        response_source = "ray" if self.use_ray else "local"
        try:
            if self.use_ray:
                logger.info(
                    "llm.ray.dispatch",
                    extra={
                        "task_type": task_type,
                        "ray_url": self.ray_url,
                        "adapter_version": self._current_adapter_version,
                    },
                )
                try:
                    response = await self._ray_call(
                        active_prompt, task_type, adapter_metadata
                    )
                except Exception:
                    logger.exception(
                        "llm.ray.error",
                        extra={"task_type": task_type, "cache_key": cache_key},
                    )
                    response = await self._call_local_provider(active_prompt, task_type)
                    response_source = "local"
                    logger.info(
                        "llm.ray.fallback_local",
                        extra={
                            "task_type": task_type,
                            "adapter_version": self._current_adapter_version,
                            "reason": "ray_exception",
                        },
                    )
                else:
                    error_message = self._ray_response_error(response)
                    if error_message:
                        logger.warning(
                            "llm.ray.error_response",
                            extra={
                                "task_type": task_type,
                                "cache_key": cache_key,
                                "error": error_message,
                            },
                        )
                        response = await self._call_local_provider(
                            active_prompt, task_type
                        )
                        response_source = "local"
                        logger.info(
                            "llm.ray.fallback_local",
                            extra={
                                "task_type": task_type,
                                "adapter_version": self._current_adapter_version,
                                "reason": "error_response",
                            },
                        )
            else:
                response = await self._call_local_provider(active_prompt, task_type)
        except LLMRuntimeError as exc:
            return await self._fail(cache_key, str(exc))
        generated_text = self._extract_text(response)
        if not generated_text and response_source == "ray":
            logger.warning(
                "llm.ray.empty_response",
                extra={
                    "task_type": task_type,
                    "adapter_version": self._current_adapter_version,
                },
            )
            try:
                response = await self._call_local_provider(active_prompt, task_type)
            except LLMRuntimeError as exc:
                return await self._fail(cache_key, str(exc))
            generated_text = self._extract_text(response)
            response_source = "local"
            logger.info(
                "llm.ray.fallback_local",
                extra={
                    "task_type": task_type,
                    "adapter_version": self._current_adapter_version,
                    "reason": "empty_response",
                },
            )
        confidence = self._calculate_confidence(generated_text)
        tokens_used = len(generated_text.split())
        result = {
            "text": generated_text,
            "confidence": confidence,
            "tokens_used": tokens_used,
            "source": response_source,
            "adapter_version": self._current_adapter_version,
        }
        await _RESPONSE_CACHE.set(cache_key, result, ttl=300)
        return result

    def _failure_payload(self, message: str) -> dict[str, Any]:
        """Create a standardised failure payload for telemetry."""

        return {
            "text": message,
            "confidence": 0.0,
            "tokens_used": 0,
            "source": "error",
            "adapter_version": self._current_adapter_version,
        }

    def infer_task_type(self, prompt: str, default: str = "general") -> str:
        """Infer the most suitable model role for ``prompt``.

        Heuristics err on the side of caution: switching to the coding pathway
        requires multiple independent signals such as fenced code blocks,
        language keywords, structural markers, or stack traces. Rules are
        compiled up-front so operators can tune routing behaviour without
        editing this method directly.
        """

        if not prompt:
            return default

        stripped = prompt.strip()
        if not stripped or not self._task_type_rules:
            return default

        lowered = stripped.lower()
        score = 0

        for rule in self._task_type_rules:
            weight, short_circuit = rule.evaluate(stripped, lowered)
            if weight:
                score += weight
                if short_circuit:
                    return "coding"

        return "coding" if score >= self._coding_score_threshold else default

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _resolve_prompt_token_limit(self, task_type: str) -> int | None:
        parameters = self._model_manager.get_model_parameters(task_type)
        for key in (
            "prompt_max_tokens",
            "max_prompt_tokens",
            "context_window",
            "max_tokens",
        ):
            if key not in parameters:
                continue
            candidate = self._coerce_positive_int(parameters.get(key))
            if candidate is not None:
                return candidate
        fallback = self._coerce_positive_int(parameters.get("num_predict"))
        return fallback

    def prompt_token_limit(self, task_type: str = "general") -> int | None:
        """Return the configured prompt token limit for ``task_type``."""

        if self._prompt_limit_override is not None:
            return self._prompt_limit_override

        normalized = (task_type or "general").lower()
        if normalized in self._prompt_token_limits:
            return self._prompt_token_limits[normalized]

        resolved = self._resolve_prompt_token_limit(normalized)
        self._prompt_token_limits[normalized] = resolved
        return resolved

    def _resolve_generation_token_target(self, task_type: str) -> int | None:
        parameters = self._model_manager.get_model_parameters(task_type)
        for key in (
            "generation_tokens",
            "prompt_generation_tokens",
            "max_new_tokens",
            "num_predict",
        ):
            if key not in parameters:
                continue
            candidate = self._coerce_positive_int(parameters.get(key))
            if candidate is not None:
                return candidate
        return None

    def generation_token_target(self, task_type: str = "general") -> int | None:
        """Return the configured generation token target for ``task_type``."""

        normalized = (task_type or "general").lower()
        if normalized in self._generation_token_targets:
            return self._generation_token_targets[normalized]

        resolved = self._resolve_generation_token_target(normalized)
        self._generation_token_targets[normalized] = resolved
        return resolved

    def _compile_task_type_rules(
        self,
        overrides: Sequence[Mapping[str, object] | _TaskTypeRule] | None,
    ) -> tuple[_TaskTypeRule, ...]:
        """Compile heuristic specifications into executable rules."""

        candidate_rules: Sequence[Mapping[str, object] | _TaskTypeRule] | None = (
            overrides
        )
        if candidate_rules is None:
            configured = getattr(self._settings, "llm_task_type_rules", None)
            if isinstance(configured, Sequence) and not isinstance(
                configured, (str, bytes)
            ):
                candidate_rules = configured  # type: ignore[assignment]
        if candidate_rules is None:
            candidate_rules = self._DEFAULT_TASK_TYPE_RULES

        compiled: list[_TaskTypeRule] = []
        for raw_rule in candidate_rules:
            if isinstance(raw_rule, _TaskTypeRule):
                compiled.append(raw_rule)
                continue
            if not isinstance(raw_rule, Mapping):
                logger.warning(
                    "llm.task_type_rule.invalid_type",
                    extra={"received_type": type(raw_rule).__name__},
                )
                continue
            try:
                compiled.append(self._build_task_type_rule(raw_rule))
            except Exception:  # pragma: no cover - misconfiguration surfaced via logs
                logger.exception(
                    "llm.task_type_rule.compile_failed",
                    extra={"rule_name": str(raw_rule.get("name", "unknown"))},
                )

        return tuple(compiled)

    def _build_task_type_rule(self, spec: Mapping[str, object]) -> _TaskTypeRule:
        name = str(spec.get("name") or "rule")
        kind = str(spec.get("kind", "regex")).lower()
        weight = int(spec.get("weight", 1))
        min_matches = max(1, int(spec.get("min_matches", 1)))
        short_circuit_matches_raw = spec.get("short_circuit_matches")
        short_circuit_matches: int | None
        if short_circuit_matches_raw is None:
            short_circuit_matches = None
        else:
            short_circuit_matches = max(1, int(short_circuit_matches_raw))
        case_sensitive = bool(spec.get("case_sensitive", False))

        matcher: Callable[[str, str], int]
        if kind == "regex":
            pattern_value = spec.get("pattern")
            if isinstance(pattern_value, re.Pattern):
                pattern = pattern_value
            elif isinstance(pattern_value, str):
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(pattern_value, flags)
            else:  # pragma: no cover - configuration guardrail
                raise ValueError("regex rule requires a 'pattern' entry")

            def matcher(text: str, _lowered: str, pattern=pattern) -> int:
                return len(pattern.findall(text))

        elif kind == "keywords":
            values = spec.get("values")
            if values is None:
                raise ValueError("keyword rule requires 'values'")
            compiled_patterns = tuple(
                self._compile_keyword_pattern(str(value), case_sensitive)
                for value in values
                if str(value).strip()
            )
            if not compiled_patterns:
                raise ValueError("keyword rule produced no patterns")

            def matcher(text: str, _lowered: str, patterns=compiled_patterns) -> int:
                return sum(1 for pattern in patterns if pattern.search(text))

        elif kind == "substring":
            values = spec.get("values")
            if values is None:
                raise ValueError("substring rule requires 'values'")
            tokens = tuple(str(value) for value in values if str(value))
            if not tokens:
                raise ValueError("substring rule produced no tokens")
            if case_sensitive:

                def matcher(text: str, _lowered: str, parts=tokens) -> int:
                    return sum(1 for part in parts if part in text)

            else:
                lowered_tokens = tuple(part.lower() for part in tokens)

                def matcher(_text: str, lowered: str, parts=lowered_tokens) -> int:
                    return sum(1 for part in parts if part in lowered)

        else:  # pragma: no cover - configuration guardrail
            raise ValueError(f"unsupported task type rule kind: {kind}")

        return _TaskTypeRule(
            name=name,
            weight=weight,
            min_matches=min_matches,
            short_circuit_matches=short_circuit_matches,
            matcher=matcher,
        )

    @staticmethod
    def _compile_keyword_pattern(keyword: str, case_sensitive: bool) -> re.Pattern[str]:
        token = keyword.strip()
        if not token:
            raise ValueError("keyword cannot be empty")
        prefix = r"\b" if (token[0].isalnum() or token[0] == "_") else r"(?<!\w)"
        suffix = r"\b" if (token[-1].isalnum() or token[-1] == "_") else r"(?!\w)"
        pattern = f"{prefix}{re.escape(token)}{suffix}"
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.compile(pattern, flags)

    def _resolve_coding_score_threshold(self, override: int | None) -> int:
        if isinstance(override, int) and override > 0:
            return override
        settings_value = getattr(self._settings, "coding_task_score_threshold", None)
        if isinstance(settings_value, int) and settings_value > 0:
            return settings_value
        if isinstance(settings_value, float) and settings_value > 0:
            return int(settings_value)
        return 2

    def _extract_text(self, raw_response: dict[str, Any]) -> str:
        """Normalise the text field across Ollama and Ray responses."""

        if not isinstance(raw_response, dict):
            return ""

        message = raw_response.get("message")
        if isinstance(message, dict):
            content: object | None = message.get("content")
        else:
            content = None

        if not isinstance(content, str):
            fallback = raw_response.get("content") or raw_response.get("response")
            content = fallback if isinstance(fallback, str) else ""

        return content

    def _calculate_confidence(self, text: str) -> float:
        token_count = len(text.split())
        return min(1.0, token_count / 512)

    async def _ray_call(
        self, prompt: str, task_type: str, adapter: dict[str, str] | None
    ) -> dict[str, Any]:
        """Call the Ray Serve endpoint with retries and structured errors."""

        async def call_api() -> dict[str, Any]:
            payload: dict[str, Any] = {"prompt": prompt, "task_type": task_type}
            if adapter:
                payload["adapter"] = adapter
            endpoints = await self._prepare_ray_endpoints()
            if not endpoints:
                self._record_ray_failure("configuration", endpoint=None)
                raise RuntimeError("No Ray Serve endpoints configured")
            max_attempts = max(
                len(endpoints) * self._ray_max_scale_cycles, len(endpoints)
            )
            last_exception: Exception | None = None
            last_endpoint: str | None = None
            async with httpx.AsyncClient(
                timeout=self._ray_client_timeout,
                limits=self._ray_client_limits,
            ) as client:
                for attempt in range(max_attempts):
                    endpoint = endpoints[attempt % len(endpoints)]
                    last_endpoint = endpoint
                    attributes = self._ray_metric_attributes(
                        endpoint, status="attempt", attempt=attempt + 1
                    )
                    if attributes:
                        _RAY_REQUEST_COUNTER.add(1, attributes)
                    start_time = time.perf_counter()
                    try:
                        resp = await client.post(endpoint, json=payload)
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        status_code = exc.response.status_code
                        if status_code in self._ray_scaling_status_codes:
                            delay = self._scale_delay(attempt, exc.response)
                            logger.info(
                                "llm.ray.scaling",
                                extra={
                                    "endpoint": endpoint,
                                    "status_code": status_code,
                                    "delay_seconds": delay,
                                },
                            )
                            self._record_ray_scaling_event(
                                endpoint, status_code=status_code
                            )
                            await asyncio.sleep(delay)
                            last_exception = exc
                            continue
                        if 500 <= status_code < 600:
                            logger.warning(
                                "llm.ray.server_error",
                                extra={
                                    "endpoint": endpoint,
                                    "status_code": status_code,
                                },
                            )
                            self._record_ray_failure(
                                "server_error",
                                endpoint=endpoint,
                                status_code=status_code,
                            )
                            await asyncio.sleep(min(2**attempt, 5))
                            last_exception = exc
                            continue
                        self._record_ray_failure(
                            "client_error",
                            endpoint=endpoint,
                            status_code=status_code,
                        )
                        raise
                    except httpx.RequestError as exc:
                        logger.warning(
                            "llm.ray.transport_error",
                            extra={"endpoint": endpoint, "error": str(exc)},
                        )
                        self._record_ray_failure("transport_error", endpoint=endpoint)
                        await asyncio.sleep(min(2**attempt, 5))
                        last_exception = exc
                        continue

                    try:
                        data = resp.json()
                    except ValueError as exc:  # pragma: no cover - defensive
                        logger.exception(
                            "llm.ray.invalid_json",
                            extra={"status_code": resp.status_code},
                        )
                        self._record_ray_failure(
                            "invalid_json",
                            endpoint=endpoint,
                            status_code=getattr(resp, "status_code", None),
                        )
                        raise RuntimeError("Ray Serve returned invalid JSON") from exc

                    await self._record_ray_success(endpoint)
                    self._record_ray_latency(endpoint, time.perf_counter() - start_time)
                    return data
            self._record_ray_failure("exhausted", endpoint=last_endpoint)
            raise RuntimeError("Ray Serve request failed") from last_exception

        return await self._ray_cb.call(call_api)

    async def _prepare_ray_endpoints(self) -> list[str]:
        async with self._ray_endpoint_lock:
            if not self._ray_endpoints:
                return []
            start = self._ray_endpoint_index
            endpoints = list(self._ray_endpoints)
            self._ray_endpoint_index = (self._ray_endpoint_index + 1) % len(endpoints)
        return endpoints[start:] + endpoints[:start]

    async def _record_ray_success(self, endpoint: str) -> None:
        self.ray_url = endpoint
        async with self._ray_endpoint_lock:
            try:
                index = self._ray_endpoints.index(endpoint)
            except ValueError:
                return
            self._ray_endpoint_index = (index + 1) % len(self._ray_endpoints)

    def _ray_metric_attributes(
        self,
        endpoint: str | None,
        **extra: str | int | float | bool | None,
    ) -> dict[str, str | int | float | bool] | None:
        if not self._metrics_enabled:
            return None
        host = "unknown"
        path = ""
        if endpoint:
            parsed = urlparse(endpoint)
            host = parsed.netloc or "unknown"
            path = parsed.path or ""
        attributes: dict[str, str | int | float | bool] = {"endpoint": host}
        if path and path != "/":
            attributes["path"] = path
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                attributes[key] = value
            else:  # pragma: no cover - defensive conversion
                attributes[key] = str(value)
        return attributes

    def _record_ray_failure(
        self,
        reason: str,
        *,
        endpoint: str | None = None,
        status_code: int | None = None,
    ) -> None:
        attributes = self._ray_metric_attributes(
            endpoint,
            status="failure",
            reason=reason,
            status_code=status_code,
        )
        if attributes:
            _RAY_FAILURE_COUNTER.add(1, attributes)

    def _record_ray_latency(self, endpoint: str, duration: float) -> None:
        attributes = self._ray_metric_attributes(endpoint, status="success")
        if attributes:
            _RAY_LATENCY_HISTOGRAM.record(duration, attributes)

    def _record_ray_scaling_event(
        self, endpoint: str, *, status_code: int | None = None
    ) -> None:
        attributes = self._ray_metric_attributes(
            endpoint,
            status="scaling",
            reason="status_code",
            status_code=status_code,
        )
        if attributes:
            _RAY_SCALING_COUNTER.add(1, attributes)

    def _scale_delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    value = float(retry_after)
                except ValueError:
                    value = None
                else:
                    if value >= 0:
                        return value
        idx = min(attempt, len(self._ray_scaling_backoff) - 1)
        return self._ray_scaling_backoff[idx]

    def _parse_ray_urls(self, value: str | None) -> list[str]:
        if not value:
            return []
        return [entry.strip() for entry in value.split(",") if entry.strip()]

    def _ray_response_error(self, response: dict[str, Any] | None) -> str | None:
        if not isinstance(response, dict):
            return None
        error = response.get("error")
        if not error:
            return None
        detail = response.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
        if isinstance(error, str):
            return error
        return str(error)

    async def chat(self, user_id: str, prompt: str, stream: bool = True) -> str | None:
        """Generate chat responses and publish UI streaming events."""

        logger.info(
            "llm.chat.start",
            extra={"user_id": user_id, "stream": stream},
        )
        try:
            if stream:
                async for piece in self._stream_inference(prompt):
                    if not piece:
                        continue
                    await event_bus().publish(
                        make_event(
                            "ai_model.response_chunk",
                            user_id,
                            {"delta": piece},
                        )
                    )
                await event_bus().publish(
                    make_event(
                        "ai_model.response_complete",
                        user_id,
                        {"ok": True},
                    )
                )
                return None

            text = await self._inference(prompt)
            await event_bus().publish(
                make_event(
                    "ai_model.response_complete",
                    user_id,
                    {"ok": True, "text": text},
                )
            )
            return text
        except asyncio.CancelledError:
            logger.info(
                "llm.chat.cancelled",
                extra={"user_id": user_id, "stream": stream},
            )
            await event_bus().publish(
                make_event(
                    "ai_model.response_complete",
                    user_id,
                    {"ok": False, "error": "cancelled"},
                )
            )
            raise
        except Exception as exc:
            logger.exception(
                "llm.chat.error",
                extra={"user_id": user_id, "stream": stream},
            )
            await event_bus().publish(
                make_event(
                    "ai_model.response_complete",
                    user_id,
                    {"ok": False, "error": str(exc)},
                )
            )
            raise

    async def _stream_inference(
        self, prompt: str, task_type: str = "general"
    ) -> AsyncIterator[str]:
        """Yield response fragments for the given prompt."""

        response = await self.generate_response(prompt, task_type=task_type)
        text = response.get("text", "") if isinstance(response, dict) else ""
        if text:
            for start in range(0, len(text), STREAM_CHUNK_SIZE):
                yield text[slice(start, start + STREAM_CHUNK_SIZE)]

    async def _inference(self, prompt: str, task_type: str = "general") -> str:
        """Return the full response text for the prompt."""

        response = await self.generate_response(prompt, task_type=task_type)
        return "" if not isinstance(response, dict) else str(response.get("text", ""))
