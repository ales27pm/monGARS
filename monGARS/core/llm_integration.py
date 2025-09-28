from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
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

            # Entry expired – delete to keep the cache tidy.
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

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute ``func`` unless the breaker is open."""

        current_time = asyncio.get_running_loop().time()
        if self.failure_count >= self.fail_max:
            if (
                self.last_failure_time
                and (current_time - self.last_failure_time) < self.reset_timeout
            ):
                raise CircuitBreakerOpenError("Circuit breaker open: too many failures")
            self.failure_count = 0

        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as exc:  # pragma: no cover - defensive
            self.failure_count += 1
            self.last_failure_time = current_time
            raise exc


cb = CircuitBreaker(fail_max=3, reset_timeout=60)


class LLMIntegration:
    """Adapter responsible for generating responses via local or remote LLMs."""

    def __init__(self) -> None:
        self.general_model = "dolphin-mistral:7b-v2.8-q4_K_M"
        self.coding_model = "qwen2.5-coder:7b-instruct-q6_K"
        self.use_ray = os.getenv("USE_RAY_SERVE", "False").lower() in ("true", "1")
        self.ray_url = os.getenv("RAY_SERVE_URL", "http://localhost:8000/generate")
        if self.use_ray:
            logger.info(
                "llm.ray.enabled",
                extra={"ray_url": self.ray_url, "use_ray": self.use_ray},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(OllamaNotAvailableError)
        ),
    )
    async def _ollama_call(self, model: str, prompt: str) -> dict[str, Any]:
        """Invoke an Ollama model with retries and circuit breaking."""

        async def call_api() -> dict[str, Any]:
            if not ollama:
                raise OllamaNotAvailableError("Ollama client is not available")
            return await asyncio.to_thread(
                ollama.chat,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": settings.ai_model_temperature,
                    "top_p": 0.9,
                    "num_predict": 512,
                    "stream": False,
                },
            )

        return await cb.call(call_api)

    async def generate_response(
        self, prompt: str, task_type: str = "general"
    ) -> dict[str, Any]:
        """Generate a response for ``prompt`` using the configured LLM stack."""

        cache_key = f"{task_type}:{prompt}"
        cached_response = await _RESPONSE_CACHE.get(cache_key)
        if cached_response:
            return cached_response
        if self.use_ray:
            logger.info("Using Ray Serve for inference")
            try:
                response = await self._ray_call(prompt, task_type)
            except Exception as e:
                logger.error("Ray Serve request failed: %s", e, exc_info=True)
                fallback = self._failure_payload("Ray Serve unavailable.")
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
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
                fallback = self._failure_payload("Ollama client is not available.")
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
            try:
                response = await self._ollama_call(model_name, prompt)
            except OllamaNotAvailableError:
                logger.error("llm.ollama.unavailable")
                fallback = self._failure_payload("Ollama client is not available.")
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
            except RetryError:
                logger.error(
                    "llm.ollama.retry_exhausted",
                    extra={"model_name": model_name, "task_type": task_type},
                )
                fallback = self._failure_payload(
                    "Unable to generate response at this time."
                )
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
            except Exception as e:
                logger.error(
                    "llm.ollama.error",
                    exc_info=True,
                    extra={"model_name": model_name, "task_type": task_type},
                )
                fallback = self._failure_payload(
                    "An error occurred while generating the response."
                )
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
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
            content = message.get("content")
            if isinstance(content, str):
                return content

        content = raw_response.get("content")
        if isinstance(content, str):
            return content

        response_text = raw_response.get("response")
        if isinstance(response_text, str):
            return response_text

        return ""

    def _calculate_confidence(self, text: str) -> float:
        token_count = len(text.split())
        return min(1.0, token_count / 512)

    async def _ray_call(self, prompt: str, task_type: str) -> dict[str, Any]:
        """Call the Ray Serve endpoint with retries and structured errors."""

        async def call_api() -> dict[str, Any]:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.ray_url,
                    json={"prompt": prompt, "task_type": task_type},
                    timeout=10,
                )
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError as exc:  # pragma: no cover - defensive
                    logger.error(
                        "llm.ray.invalid_json",
                        extra={"status_code": resp.status_code},
                    )
                    raise RuntimeError("Ray Serve returned invalid JSON") from exc

        return await cb.call(call_api)
