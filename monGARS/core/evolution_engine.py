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

    async def _scale_workers(self, delta: int) -> None:
        config.load_incluster_config()
        apps_v1 = client.AppsV1Api()
        deployment = apps_v1.read_namespaced_deployment(
            name="mongars-workers", namespace="default"
        )
        current = deployment.spec.replicas or 0
        patch = {"spec": {"replicas": current + delta}}
        apps_v1.patch_namespaced_deployment(
            name="mongars-workers", namespace="default", body=patch
        )
        logger.info("Scaled workers by %s", delta)

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
