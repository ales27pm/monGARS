"""Ray Serve deployment for monGARS LLM inference."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from modules.neurons.core import NeuronManager
from modules.neurons.registry import MANIFEST_FILENAME, load_manifest
from monGARS.config import get_settings
from monGARS.core.model_manager import LLMModelManager

logger = logging.getLogger(__name__)

_ALLOWED_RAY_UPDATE_KEYS = {"adapter_path", "version", "weights_path"}

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
        settings = get_settings()
        self._model_manager = LLMModelManager(settings)
        general_definition = self._model_manager.get_model_definition("general")
        coding_definition = self._model_manager.get_model_definition("coding")
        self.general_model = os.getenv("RAY_GENERAL_MODEL", general_definition.name)
        self.coding_model = os.getenv("RAY_CODING_MODEL", coding_definition.name)
        resolved_temperature = self._model_manager.resolve_parameter(
            "general", "temperature", settings.AI_MODEL_TEMPERATURE
        )
        try:
            default_temperature = float(resolved_temperature)
        except (TypeError, ValueError):
            default_temperature = float(settings.AI_MODEL_TEMPERATURE)
        resolved_top_p = self._model_manager.resolve_parameter("general", "top_p", 0.9)
        try:
            default_top_p = float(resolved_top_p)
        except (TypeError, ValueError):
            default_top_p = 0.9
        resolved_tokens = self._model_manager.resolve_parameter(
            "general", "num_predict", 512
        )
        try:
            default_max_tokens = int(resolved_tokens)
        except (TypeError, ValueError):
            default_max_tokens = 512
        if default_max_tokens <= 0:
            default_max_tokens = 512
        self.temperature = _safe_float(
            os.getenv("RAY_MODEL_TEMPERATURE"), default=default_temperature
        )
        self.top_p = _safe_float(os.getenv("RAY_MODEL_TOP_P"), default=default_top_p)
        self.max_tokens = _safe_int(
            os.getenv("RAY_MODEL_MAX_TOKENS"), default=default_max_tokens
        )
        manifest = load_manifest(self.registry_path)
        default_adapter: Optional[str] = None
        default_wrapper: Optional[str] = None
        if manifest and manifest.current:
            payload = manifest.build_payload()
            if payload:
                default_adapter = payload.get("adapter_path")
                default_wrapper = payload.get("wrapper_path")
                self._adapter_payload = payload
                self._adapter_version = payload.get("version", "baseline")
                self._manifest_mtime = self._stat_manifest()
        self.neuron_manager = NeuronManager(
            base_model_path=base_model_path,
            default_encoder_path=default_adapter,
            llm2vec_options={"device_map": DEFAULT_DEVICE_MAP},
            wrapper_dir=default_wrapper,
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
                return await self._handle_requested_adapter(incoming)
            return await self._refresh_from_manifest()

    async def _handle_requested_adapter(
        self, incoming: dict[str, Any]
    ) -> dict[str, str] | None:
        requested_path = self._resolve_requested_path(incoming.get("adapter_path"))
        if requested_path is None:
            return self._adapter_payload

        manifest = await self._load_manifest()
        manifest_version = (
            manifest.current.version if manifest and manifest.current else None
        )
        requested_version = str(incoming.get("version") or "")
        if (
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
        wrapper_path_obj = self._resolve_wrapper_path(incoming.get("wrapper_path"))
        wrapper_path = str(wrapper_path_obj) if wrapper_path_obj else None
        if effective_version != self._adapter_version:
            await asyncio.to_thread(
                self.neuron_manager.switch_encoder,
                str(requested_path),
                wrapper_dir=wrapper_path,
            )
            self._adapter_version = effective_version
            logger.info(
                "llm.ray.adapter.switched",
                extra={
                    "adapter_version": self._adapter_version,
                    "adapter_path": str(requested_path),
                },
            )

        payload_snapshot = {
            "adapter_path": str(requested_path),
            "version": self._adapter_version,
        }
        if wrapper_path:
            payload_snapshot["wrapper_path"] = str(wrapper_path)
        self._adapter_payload = payload_snapshot
        return self._adapter_payload

    async def _refresh_from_manifest(self) -> dict[str, str] | None:
        current_mtime = self._stat_manifest()
        if not current_mtime:
            return self._adapter_payload
        if self._manifest_mtime and current_mtime <= self._manifest_mtime:
            return self._adapter_payload

        manifest = await self._load_manifest()
        if manifest is None:
            return self._adapter_payload

        self._manifest_mtime = current_mtime
        if manifest.current:
            payload = manifest.build_payload()
            if payload:
                adapter_candidate = self._resolve_requested_path(
                    payload.get("adapter_path")
                )
                wrapper_candidate = self._resolve_wrapper_path(
                    payload.get("wrapper_path")
                )
                if adapter_candidate:
                    await asyncio.to_thread(
                        self.neuron_manager.switch_encoder,
                        str(adapter_candidate),
                        wrapper_dir=(
                            str(wrapper_candidate) if wrapper_candidate else None
                        ),
                    )
                    payload["adapter_path"] = str(adapter_candidate)
                    if wrapper_candidate:
                        payload["wrapper_path"] = str(wrapper_candidate)
                    elif "wrapper_path" in payload:
                        payload.pop("wrapper_path")
                else:
                    logger.warning(
                        "llm.ray.adapter.rejected",
                        extra={
                            "reason": "invalid_manifest_path",
                            "requested_path": payload.get("adapter_path"),
                        },
                    )
            self._adapter_version = (
                payload.get("version", "baseline") if payload else "baseline"
            )
            self._adapter_payload = payload if payload else None
        return self._adapter_payload

    async def _load_manifest(self) -> Any | None:
        try:
            return await asyncio.to_thread(load_manifest, self.registry_path)
        except (OSError, ValueError) as exc:
            logger.warning(
                "llm.ray.manifest_unavailable",
                extra={"registry_path": str(self.registry_path)},
                exc_info=exc,
            )
            return None

    def _resolve_requested_path(self, adapter_path: Any) -> Path | None:
        if not adapter_path:
            return None
        root = self.registry_path.resolve()
        try:
            requested = Path(str(adapter_path)).resolve()
        except (OSError, RuntimeError, ValueError, TypeError):
            requested = None
        if requested is None:
            return None
        if requested != root and root not in requested.parents:
            logger.warning(
                "llm.ray.adapter.rejected",
                extra={
                    "reason": "outside_registry",
                    "requested_path": str(adapter_path),
                    "registry_root": str(root),
                },
            )
            return None
        return requested

    def _resolve_wrapper_path(self, wrapper_path: Any) -> Path | None:
        if not wrapper_path:
            return None
        root = self.registry_path.resolve()
        try:
            resolved = Path(str(wrapper_path)).resolve()
        except (OSError, RuntimeError, ValueError, TypeError):
            logger.warning(
                "llm.ray.wrapper.rejected",
                extra={"reason": "unresolvable", "wrapper_path": str(wrapper_path)},
            )
            return None
        if resolved != root and root not in resolved.parents:
            logger.warning(
                "llm.ray.wrapper.rejected",
                extra={
                    "reason": "outside_registry",
                    "wrapper_path": str(wrapper_path),
                    "registry_root": str(root),
                },
            )
            return None
        return resolved

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


def _normalise_ray_update_payload(user_config: Mapping[str, Any]) -> dict[str, Any]:
    unexpected = set(user_config) - _ALLOWED_RAY_UPDATE_KEYS
    if unexpected:
        joined = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Unsupported Ray Serve user_config keys: {joined}")

    payload: dict[str, Any] = {}
    for key in _ALLOWED_RAY_UPDATE_KEYS:
        if key not in user_config:
            continue
        value = user_config[key]
        if value is None:
            continue
        if isinstance(value, os.PathLike):
            payload[key] = os.fspath(value)
        elif isinstance(value, (str, int, float, bool)):
            payload[key] = value
        else:
            raise RuntimeError(
                "Unsupported value type for Ray Serve payload key"
                f" {key!r}: {type(value).__name__}"
            )

    if not payload:
        raise RuntimeError("Ray Serve deployment update payload is empty")

    return payload


def update_ray_deployment(user_config: Mapping[str, Any]) -> None:
    """Update the active Ray Serve deployment with the provided adapter payload."""

    if serve is None:  # pragma: no cover - environment dependent
        raise RuntimeError("Ray Serve is not available in this environment")

    try:
        deployment = serve.get_deployment("LLMServeDeployment")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to resolve Ray Serve deployment") from exc

    if deployment is None:  # pragma: no cover - deployment not registered
        raise RuntimeError("LLMServeDeployment is not registered")

    payload = _normalise_ray_update_payload(user_config)

    try:
        deployment.options(user_config=payload).deploy()
    except Exception as exc:  # pragma: no cover - Ray Serve API failure
        raise RuntimeError("Failed to update Ray Serve deployment") from exc

    logger.info(
        "llm.ray.deployment.updated",
        extra={
            "adapter_path": payload.get("adapter_path"),
            "version": payload.get("version"),
        },
    )


__all__ = [
    "LLMServeDeployment",
    "RayLLMDeployment",
    "deploy_ray_service",
    "update_ray_deployment",
]
