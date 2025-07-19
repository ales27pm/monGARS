from __future__ import annotations

import asyncio
import logging

from kubernetes import client, config

from monGARS.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EvolutionEngine:
    async def diagnose_performance(self) -> list[str]:
        """Return a list of performance issues."""
        return ["High CPU", "Memory spike"]

    async def _scale_workers(
        self,
        delta: int,
        name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """Adjust worker replicas by ``delta`` safely."""
        if delta == 0:
            return

        if name is None:
            name = settings.worker_deployment_name
        if namespace is None:
            namespace = settings.worker_deployment_namespace

        config.load_incluster_config()
        apps_v1 = client.AppsV1Api()
        loop = asyncio.get_running_loop()

        try:
            deployment = await loop.run_in_executor(
                None, apps_v1.read_namespaced_deployment, name, namespace
            )
        except client.exceptions.ApiException as exc:
            logger.error("Failed to read deployment: %s", exc)
            raise

        current = deployment.spec.replicas or 0
        new_count = current + delta
        if new_count < 0:
            raise ValueError("Resulting replica count cannot be negative")

        patch = {"spec": {"replicas": new_count}}
        try:
            await loop.run_in_executor(
                None,
                apps_v1.patch_namespaced_deployment,
                name,
                namespace,
                patch,
            )
        except client.exceptions.ApiException as exc:
            logger.error("Failed to patch deployment: %s", exc)
            raise

        logger.info("Scaled workers from %s to %s", current, new_count)

    async def _clear_caches(self) -> None:
        from monGARS.core.caching.tiered_cache import clear_cache

        await clear_cache("default")
        logger.info("Caches cleared.")

    async def apply_optimizations(self) -> None:
        suggestions = await self.diagnose_performance()
        if "High CPU" in suggestions:
            await self._scale_workers(1)
        if "Memory spike" in suggestions:
            await self._clear_caches()

    async def safe_apply_optimizations(self) -> bool:
        """Run optimizations and log failures."""
        try:
            await self.apply_optimizations()
            return True
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Optimization failed")
            return False
