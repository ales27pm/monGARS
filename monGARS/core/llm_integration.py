from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import httpx

try:  # pragma: no cover - optional dependency during tests
    import ollama
except ImportError:  # pragma: no cover - allow tests without ollama
    ollama = None
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from modules.neurons.registry import MANIFEST_FILENAME, load_manifest
from monGARS.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar("T")


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


class OllamaNotAvailableError(RuntimeError):
    """Raised when the optional Ollama client is unavailable."""


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


class LLMIntegration:
    """Adapter responsible for generating responses via local or remote LLMs."""

    def __init__(self) -> None:
        self.general_model = "dolphin-mistral:7b-v2.8-q4_K_M"
        self.coding_model = "qwen2.5-coder:7b-instruct-q6_K"
        self.use_ray = os.getenv("USE_RAY_SERVE", "False").lower() in ("true", "1")
        self.ray_url = os.getenv("RAY_SERVE_URL", "http://localhost:8000/generate")
        registry_override = os.getenv("LLM_ADAPTER_REGISTRY_PATH")
        registry_source = registry_override or settings.llm_adapter_registry_path
        self.adapter_registry_path = Path(registry_source)
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
                    "use_ray": self.use_ray,
                    "adapter_registry": str(self.adapter_registry_path),
                },
            )
        self._ollama_cb = CircuitBreaker(fail_max=3, reset_timeout=60)
        self._ray_cb = CircuitBreaker(fail_max=3, reset_timeout=60)

    def _cache_key(self, task_type: str, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return f"{task_type}:{self._current_adapter_version}:{digest}"

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
                return self._adapter_metadata

            try:
                manifest = await asyncio.to_thread(
                    load_manifest, self.adapter_registry_path
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "llm.adapter.manifest_unavailable",
                    extra={"manifest_path": str(self.adapter_manifest_path)},
                    exc_info=exc,
                )
                self._adapter_manifest_mtime = stat.st_mtime
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

    async def _fail(
        self, cache_key: str, message: str, ttl: int = 60
    ) -> dict[str, Any]:
        payload = self._failure_payload(message)
        await _RESPONSE_CACHE.set(cache_key, payload, ttl=ttl)
        return payload

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(OllamaNotAvailableError)
            & retry_if_not_exception_type(CircuitBreakerOpenError)
        ),
    )
    async def _ollama_call(self, model: str, prompt: str) -> dict[str, Any]:
        """Invoke an Ollama model with retries and circuit breaking."""

        async def call_api() -> dict[str, Any]:
            if not ollama:
                raise OllamaNotAvailableError("Ollama client is not available")
            temperature = getattr(settings, "ai_model_temperature", None)
            if temperature is None:
                temperature = getattr(settings, "AI_MODEL_TEMPERATURE", 0.7)
            return await asyncio.to_thread(
                ollama.chat,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": float(temperature),
                    "top_p": 0.9,
                    "num_predict": 512,
                    "stream": False,
                },
            )

        return await self._ollama_cb.call(call_api)

    async def generate_response(
        self, prompt: str, task_type: str = "general"
    ) -> dict[str, Any]:
        """Generate a response for ``prompt`` using the configured LLM stack."""

        adapter_metadata: dict[str, str] | None
        if self.use_ray:
            adapter_metadata = await self._ensure_adapter_metadata()
        else:
            adapter_metadata = None
            if self._current_adapter_version != "baseline":
                self._update_adapter_version(None)
        cache_key = self._cache_key(task_type, prompt)
        cached_response = await _RESPONSE_CACHE.get(cache_key)
        if cached_response:
            return cached_response
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
                response = await self._ray_call(prompt, task_type, adapter_metadata)
            except Exception:
                logger.exception(
                    "llm.ray.error",
                    extra={"task_type": task_type, "cache_key": cache_key},
                )
                return await self._fail(cache_key, "Ray Serve unavailable.")
        else:
            model_name = (
                self.general_model
                if task_type.lower() == "general"
                else self.coding_model
            )
            logger.info(
                "llm.ollama.dispatch",
                extra={"model_name": model_name, "task_type": task_type},
            )
            if not ollama:
                logger.error("llm.ollama.unavailable")
                return await self._fail(cache_key, "Ollama client is not available.")
            try:
                response = await self._ollama_call(model_name, prompt)
            except OllamaNotAvailableError:
                logger.exception("llm.ollama.unavailable")
                return await self._fail(cache_key, "Ollama client is not available.")
            except RetryError:
                logger.exception(
                    "llm.ollama.retry_exhausted",
                    extra={"model_name": model_name, "task_type": task_type},
                )
                return await self._fail(
                    cache_key, "Unable to generate response at this time."
                )
            except CircuitBreakerOpenError:
                logger.error(
                    "llm.ollama.circuit_open",
                    extra={"model_name": model_name, "task_type": task_type},
                )
                return await self._fail(
                    cache_key, "Ollama circuit is temporarily open."
                )
            except Exception:
                logger.exception(
                    "llm.ollama.error",
                    extra={"model_name": model_name, "task_type": task_type},
                )
                return await self._fail(
                    cache_key, "An error occurred while generating the response."
                )
        generated_text = self._extract_text(response)
        confidence = self._calculate_confidence(generated_text)
        tokens_used = len(generated_text.split())
        result = {
            "text": generated_text,
            "confidence": confidence,
            "tokens_used": tokens_used,
        }
        await _RESPONSE_CACHE.set(cache_key, result, ttl=300)
        return result

    def _failure_payload(self, message: str) -> dict[str, Any]:
        """Create a standardised failure payload for telemetry."""

        return {"text": message, "confidence": 0.0, "tokens_used": 0}

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
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.ray_url,
                    json=payload,
                    timeout=10,
                )
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "llm.ray.invalid_json",
                        extra={"status_code": resp.status_code},
                    )
                    raise RuntimeError("Ray Serve returned invalid JSON") from exc

        return await self._ray_cb.call(call_api)
