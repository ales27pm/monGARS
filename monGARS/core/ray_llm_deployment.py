from __future__ import annotations

import logging
import time
from typing import Any

try:
    import ray  # noqa: F401 - imported for side effects required by Serve
    from ray import serve
except ImportError as exc:  # pragma: no cover - enforced at deployment build time
    raise ImportError(
        "Ray Serve deployment module requires the 'ray[serve]' extra to be installed."
    ) from exc

try:  # pragma: no cover - optional based on Ray version
    from ray.serve.exceptions import RayServeException
except Exception:  # pragma: no cover - fallback when exceptions module changes

    class RayServeException(RuntimeError):
        """Fallback Ray Serve exception type."""


from monGARS.config import get_settings
from monGARS.core.llm_integration import LLMIntegration

settings = get_settings()
logger = logging.getLogger(__name__)


@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.5 if settings.llm.use_gpu else 0,
        "memory": 8 * 1024 * 1024 * 1024,  # 8GB
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
class RayLLMDeployment:
    def __init__(self):
        self.llm = LLMIntegration.instance()
        self.health_check_timestamp = time.time()

    async def health_check(self) -> dict[str, Any]:
        """Verify the underlying model is responsive."""

        now = time.time()
        if now - self.health_check_timestamp > 300:  # 5 minutes
            try:
                result = await self.__call__(
                    {"prompt": "health check", "max_new_tokens": 10}
                )
            except Exception as exc:  # pragma: no cover - unexpected runtime failure
                logger.exception("ray_llm.health_check_failed")
                return {
                    "status": "unhealthy",
                    "last_check": self.health_check_timestamp,
                    "detail": str(exc),
                    "model": settings.unified_model_dir.name,
                }
            if isinstance(result, dict) and result.get("error"):
                return {
                    "status": "unhealthy",
                    "last_check": self.health_check_timestamp,
                    "detail": result.get("message") or result.get("error"),
                    "model": settings.unified_model_dir.name,
                }
            self.health_check_timestamp = now
        return {
            "status": "healthy",
            "last_check": self.health_check_timestamp,
            "model": settings.unified_model_dir.name,
        }

    async def __call__(self, request_data: dict[str, Any]) -> dict[str, Any]:
        prompt = request_data.get("prompt", "")
        default_max_tokens = getattr(settings.model, "max_new_tokens", None)
        max_tokens = request_data.get("max_new_tokens", default_max_tokens)

        if not prompt:
            return {"error": "empty_prompt", "message": "Prompt cannot be empty"}

        try:
            kwargs: dict[str, Any] = {}
            if max_tokens is not None:
                kwargs["max_new_tokens"] = max_tokens
            response = self.llm.generate(prompt, **kwargs)
        except Exception as exc:  # pragma: no cover - unexpected runtime failure
            logger.exception(
                "ray_llm.generation_failed",
                extra={"prompt_length": len(prompt)},
            )
            return {"error": "generation_failed", "message": str(exc)}
        tokens_used: int | None = None
        tokenizer = getattr(self.llm, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "tokenize"):
            try:
                tokens_used = len(tokenizer.tokenize(prompt + response))
            except RayServeException:  # pragma: no cover - tokenizer failures are rare
                tokens_used = None
            except Exception:  # pragma: no cover - tokenizer failures are rare
                tokens_used = None
        payload: dict[str, Any] = {
            "response": response,
            "model": settings.unified_model_dir.name,
        }
        if tokens_used is not None:
            payload["tokens_used"] = tokens_used
        return payload


deployment = RayLLMDeployment.bind()
