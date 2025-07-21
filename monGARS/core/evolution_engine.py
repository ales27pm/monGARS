from __future__ import annotations

import asyncio
import logging

from kubernetes import client, config

from monGARS.config import get_settings
from monGARS.core.monitor import SystemMonitor

logger = logging.getLogger(__name__)
settings = get_settings()


class EvolutionEngine:
    def __init__(self) -> None:
        self.monitor = SystemMonitor(update_interval=1)

    async def diagnose_performance(self) -> list[str]:
        """Analyze real system stats and return a list of issues."""
        issues = []
        try:
            stats = await self.monitor.get_system_stats()
            logger.info(f"Diagnosing performance with stats: {stats}")

            if stats.cpu_usage and stats.cpu_usage > 85.0:
                issues.append("High CPU")
                logger.warning(
                    "Performance issue detected: High CPU usage (%.2f%%)",
                    stats.cpu_usage,
                )

            if stats.memory_usage and stats.memory_usage > 90.0:
                issues.append("Memory spike")
                logger.warning(
                    "Performance issue detected: High memory usage (%.2f%%)",
                    stats.memory_usage,
                )

            if stats.gpu_usage and stats.gpu_usage > 90.0:
                issues.append("High GPU")
                logger.warning(
                    "Performance issue detected: High GPU usage (%.2f%%)",
                    stats.gpu_usage,
                )
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to diagnose performance: %s", exc)

        return issues

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

        try:
            config.load_incluster_config()
        except config.ConfigException:
            logger.warning(
                "Could not load in-cluster K8s config, falling back to kube_config."
            )
            config.load_kube_config()
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
        from monGARS.core.caching.tiered_cache import TieredCache

        cache = TieredCache()
        await cache.clear_all()
        logger.info("All cache tiers cleared.")

    async def apply_optimizations(self) -> None:
        suggestions = await self.diagnose_performance()
        if "High CPU" in suggestions or "High GPU" in suggestions:
            logger.info("Applying optimization: Scaling up workers due to high load.")
            await self._scale_workers(1)
        if "Memory spike" in suggestions:
            logger.info("Applying optimization: Clearing caches due to memory spike.")
            await self._clear_caches()

    async def safe_apply_optimizations(self) -> bool:
        """Run optimizations and log failures."""
        try:
            await self.apply_optimizations()
            return True
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Optimization failed")
            return False
