"""Ray Serve deployment for monGARS LLM inference."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from modules.neurons.core import NeuronManager
from modules.neurons.registry import MANIFEST_FILENAME, load_manifest

logger = logging.getLogger(__name__)

try:  # pragma: no cover - ray is optional in tests
    import ray
    from ray import serve
except ImportError:  # pragma: no cover - fallback when ray not installed
    ray = None  # type: ignore[assignment]
    serve = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for inference
    import ollama
except ImportError:  # pragma: no cover - Ray deployment can still start without Ollama
    ollama = None

if serve:  # pragma: no cover - executed only when ray is installed
    try:
        from ray.serve.exceptions import RayServeException
    except Exception:  # pragma: no cover - defensive in case of API changes

        class RayServeException(RuntimeError):
            """Fallback Ray Serve exception type when the official one is unavailable."""

else:

    class RayServeException(RuntimeError):
        """Fallback Ray Serve exception used when Ray Serve is not installed."""


DEFAULT_BASE_MODEL = os.getenv(
    "LLM2VEC_BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_DEVICE_MAP = os.getenv("LLM2VEC_DEVICE_MAP", "cpu")
DEFAULT_REGISTRY = Path(os.getenv("LLM_ADAPTER_REGISTRY_PATH", "models/encoders"))
DEFAULT_ROUTE_PREFIX = os.getenv("RAY_ROUTE_PREFIX", "/generate")


def _resolve_registry_path(value: str | os.PathLike[str] | None) -> Path:
    path = Path(value) if value else DEFAULT_REGISTRY
    return path


def _safe_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class RayLLMDeployment:
    """Implementation backing the Ray Serve deployment."""

    def __init__(
        self,
        base_model_path: str = DEFAULT_BASE_MODEL,
        registry_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.registry_path = _resolve_registry_path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_path / MANIFEST_FILENAME
        self._manifest_mtime: float | None = None
        self._adapter_payload: dict[str, str] | None = None
        self._adapter_version = "baseline"
        self._lock = asyncio.Lock()
        self.general_model = os.getenv(
            "RAY_GENERAL_MODEL", "dolphin-mistral:7b-v2.8-q4_K_M"
        )
        self.coding_model = os.getenv(
            "RAY_CODING_MODEL", "qwen2.5-coder:7b-instruct-q6_K"
        )
        self.temperature = _safe_float(os.getenv("RAY_MODEL_TEMPERATURE"), default=0.7)
        self.top_p = _safe_float(os.getenv("RAY_MODEL_TOP_P"), default=0.9)
        self.max_tokens = _safe_int(os.getenv("RAY_MODEL_MAX_TOKENS"), default=512)
        manifest = load_manifest(self.registry_path)
        default_adapter: Optional[str] = None
        if manifest and manifest.current:
            payload = manifest.build_payload()
            if payload:
                default_adapter = payload.get("adapter_path")
                self._adapter_payload = payload
                self._adapter_version = payload.get("version", "baseline")
                self._manifest_mtime = self._stat_manifest()
        self.neuron_manager = NeuronManager(
            base_model_path=base_model_path,
            default_encoder_path=default_adapter,
            llm2vec_options={"device_map": DEFAULT_DEVICE_MAP},
        )

    def _stat_manifest(self) -> float | None:
        try:
            return self.manifest_path.stat().st_mtime
        except FileNotFoundError:
            return None

    async def _refresh_adapter(
        self, incoming: dict[str, Any] | None
    ) -> dict[str, str] | None:
        async with self._lock:
            if incoming and incoming.get("adapter_path"):
                root = self.registry_path.resolve()
                try:
                    requested_path = Path(str(incoming["adapter_path"])).resolve()
                except (OSError, RuntimeError, ValueError, TypeError):
                    requested_path = None

                is_within_registry = requested_path is not None and (
                    requested_path == root or root in requested_path.parents
                )

                try:
                    manifest = await asyncio.to_thread(
                        load_manifest, self.registry_path
                    )
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "llm.ray.manifest_unavailable",
                        extra={"registry_path": str(self.registry_path)},
                        exc_info=exc,
                    )
                    manifest = None

                manifest_version = (
                    manifest.current.version if manifest and manifest.current else None
                )
                requested_version = str(incoming.get("version") or "")

                if not is_within_registry or (
                    manifest_version
                    and requested_version
                    and requested_version != manifest_version
                ):
                    logger.warning(
                        "llm.ray.adapter.rejected",
                        extra={
                            "reason": "invalid_path_or_version",
                            "requested_path": str(incoming.get("adapter_path")),
                            "requested_version": requested_version,
                            "manifest_version": manifest_version,
                        },
                    )
                    return self._adapter_payload

                effective_version = requested_version or manifest_version or "baseline"
                if effective_version != self._adapter_version:
                    await asyncio.to_thread(
                        self.neuron_manager.switch_encoder, str(requested_path)
                    )
                    self._adapter_version = effective_version
                    logger.info(
                        "llm.ray.adapter.switched",
                        extra={
                            "adapter_version": self._adapter_version,
                            "adapter_path": str(requested_path),
                        },
                    )

                self._adapter_payload = {
                    "adapter_path": str(requested_path),
                    "version": self._adapter_version,
                }
                return self._adapter_payload

            current_mtime = self._stat_manifest()
            if not current_mtime:
                return self._adapter_payload
            if self._manifest_mtime and current_mtime <= self._manifest_mtime:
                return self._adapter_payload
            try:
                manifest = await asyncio.to_thread(load_manifest, self.registry_path)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "llm.ray.manifest_unavailable",
                    extra={"registry_path": str(self.registry_path)},
                    exc_info=exc,
                )
                return self._adapter_payload
            self._manifest_mtime = current_mtime
            if manifest and manifest.current:
                payload = manifest.build_payload()
                adapter_path = payload.get("adapter_path")
                if adapter_path:
                    await asyncio.to_thread(
                        self.neuron_manager.switch_encoder, adapter_path
                    )
                self._adapter_version = payload.get("version", "baseline")
                self._adapter_payload = payload if payload else None
            return self._adapter_payload

    def _encode_prompt(self, prompt: str) -> list[list[float]]:
        try:
            embedding = self.neuron_manager.encode([prompt])
        except (
            Exception
        ) as exc:  # pragma: no cover - fallback in production deployments
            logger.exception(
                "llm.ray.encode_failed",
                extra={"adapter_version": self._adapter_version},
            )
            raise RayServeException("Failed to encode prompt") from exc

        if not embedding:
            raise RayServeException("Encoder returned empty embedding")
        first = embedding[0]
        if not isinstance(first, list) or not first:
            raise RayServeException("Encoder returned malformed embedding")
        return embedding

    async def _render_response(
        self,
        prompt: str,
        embedding: list[list[float]],
        adapter: dict[str, Any] | None,
        task_type: str,
    ) -> dict[str, Any]:
        text, usage = await self._generate_text(prompt, task_type)
        adapter_version = (
            adapter.get("version", self._adapter_version)
            if adapter
            else self._adapter_version
        )
        embedding_summary = self._summarise_embedding(embedding)
        payload: dict[str, Any] = {
            "content": text,
            "message": {"role": "assistant", "content": text},
            "adapter_version": adapter_version,
            "usage": usage,
        }
        if embedding_summary:
            payload["embedding"] = embedding_summary
        if adapter:
            payload["adapter"] = {
                "version": adapter_version,
                "adapter_path": adapter.get("adapter_path"),
            }
        return payload

    async def _generate_text(
        self, prompt: str, task_type: str
    ) -> tuple[str, dict[str, Any]]:
        model = self._select_model(task_type)
        if not ollama:
            raise RayServeException("Ollama client is not available on the Ray replica")

        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": float(self.temperature),
                    "top_p": float(self.top_p),
                    "num_predict": int(self.max_tokens),
                    "stream": False,
                },
            )
        except Exception as exc:  # pragma: no cover - backend-specific failures
            logger.exception(
                "llm.ray.ollama_failure",
                extra={"model": model},
            )
            raise RayServeException("Text generation failed") from exc

        message = response.get("message") if isinstance(response, dict) else None
        text: Any
        if isinstance(message, dict):
            text = message.get("content")
        else:
            text = None
        if not isinstance(text, str):
            text = response.get("content") if isinstance(response, dict) else None
        if not isinstance(text, str):
            raise RayServeException("LLM response did not include textual content")

        usage_raw = response.get("usage") if isinstance(response, dict) else None
        usage: dict[str, Any]
        if isinstance(usage_raw, dict):
            usage = {
                key: value
                for key, value in usage_raw.items()
                if isinstance(key, str) and isinstance(value, (int, float))
            }
        else:
            usage = {}

        usage.setdefault("model", model)
        return text, usage

    def _summarise_embedding(
        self, embedding: list[list[float]]
    ) -> dict[str, Any] | None:
        if not embedding or not isinstance(embedding[0], list) or not embedding[0]:
            return None
        vector = embedding[0]
        magnitude = math.sqrt(sum(value * value for value in vector))
        mean_activation = sum(vector) / len(vector)
        return {
            "dimension": len(vector),
            "norm": magnitude,
            "mean": mean_activation,
        }

    def _select_model(self, task_type: str) -> str:
        if task_type.lower() in {"code", "coding", "developer"}:
            return self.coding_model
        return self.general_model

    async def __call__(self, request: Any) -> Dict[str, Any]:
        try:
            data = await request.json()
        except Exception as exc:  # pragma: no cover - defensive parsing
            logger.exception("llm.ray.invalid_request")
            return {
                "content": "",
                "error": "invalid_request",
                "detail": "Unable to parse JSON payload.",
                "adapter_version": self._adapter_version,
            }

        if not isinstance(data, dict):
            return {
                "content": "",
                "error": "invalid_request",
                "detail": "Request payload must be an object.",
                "adapter_version": self._adapter_version,
            }

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str) or not prompt:
            return {
                "content": "",
                "error": "prompt_missing",
                "adapter_version": self._adapter_version,
            }
        task_type = str(data.get("task_type") or "general")
        adapter_payload = await self._refresh_adapter(data.get("adapter"))
        try:
            embedding = await asyncio.to_thread(self._encode_prompt, prompt)
        except RayServeException as exc:
            return {
                "content": "",
                "error": "embedding_failed",
                "detail": str(exc),
                "adapter_version": self._adapter_version,
            }

        try:
            payload = await self._render_response(
                prompt, embedding, adapter_payload, task_type
            )
        except RayServeException as exc:
            logger.exception(
                "llm.ray.inference_failed",
                extra={"adapter_version": self._adapter_version},
            )
            return {
                "content": "",
                "error": "inference_failed",
                "detail": str(exc),
                "adapter_version": self._adapter_version,
            }

        embedding_dimension = len(embedding[0]) if embedding and embedding[0] else 0
        payload.setdefault("adapter_version", self._adapter_version)
        payload.setdefault("embedding_dimension", embedding_dimension)
        if "adapter" not in payload:
            if adapter_payload:
                payload["adapter"] = {
                    "version": adapter_payload.get("version", self._adapter_version)
                }
            else:
                payload["adapter"] = {"version": self._adapter_version}
        return payload


if serve:  # pragma: no cover - decorator requires ray
    try:
        LLMServeDeployment = serve.deployment(route_prefix=DEFAULT_ROUTE_PREFIX)(
            RayLLMDeployment
        )
        _SERVE_ROUTE_PREFIX_SUPPORTED = True
    except ValueError:
        LLMServeDeployment = serve.deployment(RayLLMDeployment)
        _SERVE_ROUTE_PREFIX_SUPPORTED = False
else:  # pragma: no cover - executed only when ray missing

    class LLMServeDeployment(RayLLMDeployment):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Ray Serve is not available; install ray[serve] to use.")

    _SERVE_ROUTE_PREFIX_SUPPORTED = False


def deploy_ray_service(
    *,
    base_model_path: str | None = None,
    registry_path: str | os.PathLike[str] | None = None,
) -> None:
    """Start Ray Serve and deploy the LLM service."""

    if serve is None or ray is None:  # pragma: no cover - environment dependent
        raise RuntimeError("Ray Serve is not available in this environment")
    if not ray.is_initialized():
        ray.init(include_dashboard=False, log_to_driver=False)
    deployment = LLMServeDeployment.bind(
        base_model_path or DEFAULT_BASE_MODEL, str(registry_path or DEFAULT_REGISTRY)
    )
    if _SERVE_ROUTE_PREFIX_SUPPORTED:
        serve.run(deployment)
    else:
        serve.run(deployment, route_prefix=DEFAULT_ROUTE_PREFIX)
    logger.info(
        "llm.ray.deployment.ready",
        extra={
            "route_prefix": DEFAULT_ROUTE_PREFIX,
            "registry_path": str(registry_path or DEFAULT_REGISTRY),
        },
    )


__all__ = [
    "LLMServeDeployment",
    "RayLLMDeployment",
    "deploy_ray_service",
]
