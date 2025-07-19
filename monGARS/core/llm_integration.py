import asyncio
import logging
import os
from typing import Dict

import httpx
import ollama
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from monGARS.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AsyncTTLCache:
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str):
        async with self._lock:
            entry = self._cache.get(key)
            if entry and entry["expiry"] > asyncio.get_event_loop().time():
                logger.info(f"Cache hit for key: {key}")
                return entry["value"]
            elif entry:
                del self._cache[key]
            return None

    async def set(self, key: str, value: Dict, ttl: int = 300):
        async with self._lock:
            expiry = asyncio.get_event_loop().time() + ttl
            self._cache[key] = {"value": value, "expiry": expiry}
            logger.info(f"Cached response for key: {key} for {ttl} seconds")


_RESPONSE_CACHE = AsyncTTLCache()


class CircuitBreaker:
    def __init__(self, fail_max: int = 3, reset_timeout: int = 60):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        current_time = asyncio.get_event_loop().time()
        if self.failure_count >= self.fail_max:
            if (
                self.last_failure_time
                and (current_time - self.last_failure_time) < self.reset_timeout
            ):
                raise Exception("Circuit breaker open: too many failures")
            else:
                self.failure_count = 0
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            raise e


cb = CircuitBreaker(fail_max=3, reset_timeout=60)


class LLMIntegration:
    def __init__(self):
        self.general_model = "dolphin-mistral:7b-v2.8-q4_K_M"
        self.coding_model = "qwen2.5-coder:7b-instruct-q6_K"
        self.use_ray = os.getenv("USE_RAY_SERVE", "False").lower() in ("true", "1")
        self.ray_url = os.getenv("RAY_SERVE_URL", "http://localhost:8000/generate")
        if self.use_ray:
            logger.info("Ray Serve integration enabled at %s", self.ray_url)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _ollama_call(self, model: str, prompt: str) -> Dict:
        async def call_api():
            response = await ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": settings.ai_model_temperature,
                    "top_p": 0.9,
                    "num_predict": 512,
                    "stream": False,
                },
            )
            return response

        return await cb.call(call_api)

    async def generate_response(self, prompt: str, task_type: str = "general") -> Dict:
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
                fallback = {
                    "text": "Ray Serve unavailable.",
                    "confidence": 0.0,
                    "tokens_used": 0,
                }
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
        else:
            model_name = (
                self.general_model
                if task_type.lower() == "general"
                else self.coding_model
            )
            logger.info(f"Using model {model_name} for prompt: {prompt}")
            try:
                response = await self._ollama_call(model_name, prompt)
            except RetryError:
                logger.error(
                    f"Retries exhausted for model {model_name} with prompt: {prompt}"
                )
                fallback = {
                    "text": "Unable to generate response at this time.",
                    "confidence": 0.0,
                    "tokens_used": 0,
                }
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
            except Exception as e:
                logger.error(f"Error during LLM call: {e}", exc_info=True)
                fallback = {
                    "text": "An error occurred while generating the response.",
                    "confidence": 0.0,
                    "tokens_used": 0,
                }
                await _RESPONSE_CACHE.set(cache_key, fallback, ttl=60)
                return fallback
        generated_text = response.get("content", "")
        confidence = self._calculate_confidence(generated_text)
        tokens_used = len(generated_text.split())
        result = {
            "text": generated_text,
            "confidence": confidence,
            "tokens_used": tokens_used,
        }
        await _RESPONSE_CACHE.set(cache_key, result, ttl=300)
        return result

    def _calculate_confidence(self, text: str) -> float:
        token_count = len(text.split())
        return min(1.0, token_count / 512)

    async def _ray_call(self, prompt: str, task_type: str) -> Dict:
        async def call_api() -> Dict:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.ray_url,
                    json={"prompt": prompt, "task_type": task_type},
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()

        return await cb.call(call_api)
