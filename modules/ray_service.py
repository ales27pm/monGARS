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

DEFAULT_BASE_MODEL = os.getenv(
    "LLM2VEC_BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_DEVICE_MAP = os.getenv("LLM2VEC_DEVICE_MAP", "cpu")
DEFAULT_REGISTRY = Path(os.getenv("LLM_ADAPTER_REGISTRY_PATH", "models/encoders"))
DEFAULT_ROUTE_PREFIX = os.getenv("RAY_ROUTE_PREFIX", "/generate")


def _resolve_registry_path(value: str | os.PathLike[str] | None) -> Path:
    path = Path(value) if value else DEFAULT_REGISTRY
    return path


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
                version = incoming.get("version") or "baseline"
                if version != self._adapter_version:
                    await asyncio.to_thread(
                        self.neuron_manager.switch_encoder, incoming["adapter_path"]
                    )
                    self._adapter_version = version
                self._adapter_payload = {str(k): str(v) for k, v in incoming.items()}
                return self._adapter_payload

            current_mtime = self._stat_manifest()
            if not current_mtime:
                return self._adapter_payload
            if self._manifest_mtime and current_mtime <= self._manifest_mtime:
                return self._adapter_payload
            manifest = await asyncio.to_thread(load_manifest, self.registry_path)
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
        return self.neuron_manager.encode([prompt])

    def _render_response(
        self, prompt: str, embedding: list[list[float]], adapter: dict[str, Any] | None
    ) -> str:
        if not embedding:
            return f"Adapter {self._adapter_version} received prompt: {prompt}"
        vector = embedding[0]
        if not vector:
            return f"Adapter {self._adapter_version} received prompt: {prompt}"
        magnitude = math.sqrt(sum(value * value for value in vector))
        mean_activation = sum(vector) / len(vector)
        adapter_version = (
            adapter.get("version", self._adapter_version)
            if adapter
            else self._adapter_version
        )
        prompt_length = len(prompt.split())
        return (
            f"Adapter {adapter_version} processed a prompt with {prompt_length} tokens. "
            f"Embedding magnitude {magnitude:.3f}, mean activation {mean_activation:.3f}."
        )

    async def __call__(self, request: Any) -> Dict[str, Any]:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not isinstance(prompt, str) or not prompt:
            return {
                "content": "",
                "error": "prompt_missing",
                "adapter_version": self._adapter_version,
            }
        adapter_payload = await self._refresh_adapter(data.get("adapter"))
        embedding = await asyncio.to_thread(self._encode_prompt, prompt)
        content = self._render_response(prompt, embedding, adapter_payload)
        return {
            "content": content,
            "adapter_version": self._adapter_version,
            "embedding_dimension": len(embedding[0]) if embedding else 0,
            "adapter": adapter_payload,
        }


if serve:  # pragma: no cover - decorator requires ray
    LLMServeDeployment = serve.deployment(route_prefix=DEFAULT_ROUTE_PREFIX)(
        RayLLMDeployment
    )
else:  # pragma: no cover - executed only when ray missing

    class LLMServeDeployment(RayLLMDeployment):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Ray Serve is not available; install ray[serve] to use.")


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
    serve.start(detached=False)
    deployment = LLMServeDeployment.bind(
        base_model_path or DEFAULT_BASE_MODEL, str(registry_path or DEFAULT_REGISTRY)
    )
    serve.run(deployment)
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
