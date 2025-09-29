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
        self._settings = get_settings()
        self.general_model = "dolphin-mistral:7b-v2.8-q4_K_M"
        self.coding_model = "qwen2.5-coder:7b-instruct-q6_K"
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
        self._ray_scaling_backoff = [
            float(value.strip()) for value in backoff_env.split(",") if value.strip()
        ] or [0.5, 1.0, 2.0, 4.0]
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
            temperature = float(self._settings.AI_MODEL_TEMPERATURE)
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
                if isinstance(response, dict):
                    ray_error = response.get("error")
                else:
                    ray_error = None
                if ray_error:
                    detail = (
                        response.get("detail") if isinstance(response, dict) else None
                    )
                    message = (
                        detail
                        if isinstance(detail, str) and detail
                        else f"Ray Serve reported error: {ray_error}"
                    )
                    logger.warning(
                        "llm.ray.error_response",
                        extra={
                            "task_type": task_type,
                            "cache_key": cache_key,
                            "error": ray_error,
                        },
                    )
                    return await self._fail(cache_key, message)
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
            endpoints = await self._prepare_ray_endpoints()
            if not endpoints:
                raise RuntimeError("No Ray Serve endpoints configured")
            max_attempts = max(
                len(endpoints) * self._ray_max_scale_cycles, len(endpoints)
            )
            last_exception: Exception | None = None
            async with httpx.AsyncClient(
                timeout=self._ray_client_timeout,
                limits=self._ray_client_limits,
            ) as client:
                for attempt in range(max_attempts):
                    endpoint = endpoints[attempt % len(endpoints)]
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
                            await asyncio.sleep(min(2**attempt, 5))
                            last_exception = exc
                            continue
                        raise
                    except httpx.RequestError as exc:
                        logger.warning(
                            "llm.ray.transport_error",
                            extra={"endpoint": endpoint, "error": str(exc)},
                        )
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
                        raise RuntimeError("Ray Serve returned invalid JSON") from exc

                    await self._record_ray_success(endpoint)
                    return data
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
        urls = [entry.strip() for entry in value.split(",") if entry.strip()]
        return urls
