from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from kubernetes import client, config

from monGARS.config import get_settings
from monGARS.core.monitor import SystemMonitor, SystemStats

from .ui_events import event_bus, make_event

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class PerformanceIssue:
    """Represents an optimization trigger with contextual metadata."""

    identifier: str
    severity: str
    details: dict[str, float | int]


class EvolutionEngine:
    def __init__(self) -> None:
        self.monitor = SystemMonitor(update_interval=1)
        self._stat_history: deque[SystemStats] = deque(maxlen=10)
        self._last_scale_timestamp: float = 0.0
        self._scale_cooldown_seconds: int = 60

    async def diagnose_performance(self) -> list[PerformanceIssue]:
        """Analyze system statistics and return actionable performance issues."""
        issues: list[PerformanceIssue] = []
        try:
            stats = await self.monitor.get_system_stats()
            logger.info("Diagnosing performance with stats: %s", stats)

            self._stat_history.append(stats)
            history = list(self._stat_history)

            cpu_samples = _collect_numeric(s.cpu_usage for s in history)
            mem_samples = _collect_numeric(s.memory_usage for s in history)
            disk_samples = _collect_numeric(s.disk_usage for s in history)
            gpu_samples = _collect_numeric(s.gpu_usage for s in history)
            gpu_mem_samples = _collect_numeric(s.gpu_memory_usage for s in history)

            if cpu_issue := _detect_cpu_pressure(cpu_samples):
                issues.append(cpu_issue)
            if memory_issue := _detect_memory_pressure(mem_samples):
                issues.append(memory_issue)
            if leak_issue := _detect_memory_leak(mem_samples):
                issues.append(leak_issue)
            if gpu_issue := _detect_gpu_pressure(gpu_samples, gpu_mem_samples):
                issues.append(gpu_issue)
            if disk_issue := _detect_disk_pressure(disk_samples):
                issues.append(disk_issue)
            if under_utilized := _detect_underutilization(
                cpu_samples, mem_samples, gpu_samples
            ):
                issues.append(under_utilized)
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

        apps_v1 = self._get_kubernetes_client()
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

    def _get_kubernetes_client(self) -> client.AppsV1Api:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            logger.warning(
                "Could not load in-cluster K8s config, falling back to kube_config."
            )
            try:
                config.load_kube_config()
            except config.ConfigException as exc:
                logger.error("Failed to load Kubernetes configuration: %s", exc)
                raise
        return client.AppsV1Api()

    async def _get_worker_replicas(
        self, name: str | None = None, namespace: str | None = None
    ) -> int:
        if name is None:
            name = settings.worker_deployment_name
        if namespace is None:
            namespace = settings.worker_deployment_namespace

        apps_v1 = self._get_kubernetes_client()
        loop = asyncio.get_running_loop()
        try:
            deployment = await loop.run_in_executor(
                None, apps_v1.read_namespaced_deployment, name, namespace
            )
        except client.exceptions.ApiException as exc:
            logger.error("Failed to read deployment for replica count: %s", exc)
            raise

        return deployment.spec.replicas or 0

    async def _clear_caches(self) -> None:
        from monGARS.core.caching.tiered_cache import TieredCache

        cache = TieredCache()
        await cache.clear_all()
        logger.info("All cache tiers cleared.")

    async def _maybe_scale_workers(
        self,
        delta: int,
        reason: str,
        *,
        current_replicas: int | None = None,
        name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        if delta == 0:
            return

        if current_replicas is not None and current_replicas + delta < 1:
            logger.info(
                "Skipping worker scale %s to avoid zero replicas",
                delta,
                extra={"reason": reason, "current": current_replicas},
            )
            return

        now = time.monotonic()
        if now - self._last_scale_timestamp < self._scale_cooldown_seconds:
            logger.info(
                "Skipping worker scale due to cooldown",
                extra={
                    "reason": reason,
                    "cooldown_seconds": self._scale_cooldown_seconds,
                },
            )
            return

        try:
            await self._scale_workers(delta, name=name, namespace=namespace)
        except ValueError:
            logger.info(
                "Worker scale request skipped due to invalid replica target",
                extra={"reason": reason, "delta": delta, "current": current_replicas},
            )
            return

        self._last_scale_timestamp = now
        logger.info(
            "Worker pool adjusted",
            extra={"delta": delta, "reason": reason, "current": current_replicas},
        )

    async def apply_optimizations(self) -> None:
        suggestions = await self.diagnose_performance()
        if not suggestions:
            logger.info("No performance issues detected; skipping optimizations.")
            return

        logger.info(
            "Optimization suggestions",
            extra={"issues": [issue.__dict__ for issue in suggestions]},
        )

        try:
            replicas = await self._get_worker_replicas()
        except Exception:
            logger.warning(
                "Unable to determine current worker replicas.", exc_info=True
            )
            replicas = None

        issue_map = {issue.identifier: issue for issue in suggestions}

        if cpu_issue := issue_map.get("cpu_sustained_high"):
            delta = 2 if cpu_issue.severity == "critical" else 1
            await self._maybe_scale_workers(
                delta,
                "cpu pressure",
                current_replicas=replicas,
            )
        elif gpu_issue := issue_map.get("gpu_sustained_high"):
            delta = 2 if gpu_issue.severity == "critical" else 1
            await self._maybe_scale_workers(
                delta,
                "gpu pressure",
                current_replicas=replicas,
            )

        if issue_map.get("workers_underutilized") and replicas and replicas > 1:
            await self._maybe_scale_workers(
                -1,
                "underutilized workers",
                current_replicas=replicas,
            )

        if issue_map.get("memory_pressure") or issue_map.get("memory_leak_suspected"):
            logger.info("Clearing caches to mitigate memory pressure.")
            await self._clear_caches()

        if issue_map.get("disk_capacity_pressure"):
            logger.info("Clearing caches to free disk space.")
            await self._clear_caches()

    async def safe_apply_optimizations(self) -> bool:
        """Run optimizations and log failures."""
        try:
            await self.apply_optimizations()
            return True
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Optimization failed")
            return False

    async def train_cycle(
        self,
        user_id: str | None = None,
        version: str = "enc_2025_09_29",
    ) -> None:
        """Execute a training cycle and publish lifecycle events.

        Args:
            user_id: Identifier for the user initiating the training cycle.
            version: Identifier describing the training routine being executed.
        """

        logger.info(
            "evolution.train_cycle.start",
            extra={"user_id": user_id, "version": version},
        )
        try:
            await self.apply_optimizations()
        except Exception as exc:
            await event_bus().publish(
                make_event(
                    "evolution_engine.training_failed",
                    user_id,
                    {"error": str(exc), "version": version},
                )
            )
            logger.exception(
                "evolution.train_cycle.failed",
                extra={"user_id": user_id, "version": version},
            )
            raise

        await event_bus().publish(
            make_event(
                "evolution_engine.training_complete",
                user_id,
                {"version": version},
            )
        )
        logger.info(
            "evolution.train_cycle.complete",
            extra={"user_id": user_id, "version": version},
        )


def _collect_numeric(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _detect_cpu_pressure(samples: list[float]) -> PerformanceIssue | None:
    if not samples:
        return None

    window = samples[-min(len(samples), 3) :]
    if window and min(window) > 85.0:
        severity = "critical" if fmean(window) >= 95.0 else "high"
        return PerformanceIssue(
            "cpu_sustained_high",
            severity,
            {"average": round(fmean(window), 2), "window": len(window)},
        )

    latest = samples[-1]
    if latest >= 97.0:
        return PerformanceIssue(
            "cpu_sustained_high",
            "critical",
            {"latest": round(latest, 2), "window": 1},
        )
    if latest >= 92.0:
        return PerformanceIssue(
            "cpu_sustained_high",
            "high",
            {"latest": round(latest, 2), "window": 1},
        )
    return None


def _detect_memory_pressure(samples: list[float]) -> PerformanceIssue | None:
    if not samples:
        return None

    window = samples[-min(len(samples), 3) :]
    latest = samples[-1]
    if window and fmean(window) >= 90.0:
        severity = "critical" if latest >= 95.0 else "high"
        return PerformanceIssue(
            "memory_pressure",
            severity,
            {"average": round(fmean(window), 2), "latest": round(latest, 2)},
        )
    if latest >= 94.0:
        return PerformanceIssue(
            "memory_pressure",
            "high",
            {"latest": round(latest, 2)},
        )
    return None


def _detect_memory_leak(samples: list[float]) -> PerformanceIssue | None:
    if len(samples) < 4:
        return None

    recent = samples[-4:]
    if all(a < b for a, b in zip(recent, recent[1:])) and recent[-1] >= 80.0:
        growth = recent[-1] - recent[0]
        if growth >= 12.0:
            return PerformanceIssue(
                "memory_leak_suspected",
                "medium",
                {"growth": round(growth, 2), "latest": round(recent[-1], 2)},
            )
    return None


def _detect_gpu_pressure(
    usage_samples: list[float], gpu_mem_samples: list[float]
) -> PerformanceIssue | None:
    window = usage_samples[-min(len(usage_samples), 3) :] if usage_samples else []
    usage_latest = usage_samples[-1] if usage_samples else None
    memory_latest = gpu_mem_samples[-1] if gpu_mem_samples else None

    high_usage = bool(window) and min(window) >= 85.0
    high_memory = memory_latest is not None and memory_latest >= 90.0

    if not high_usage and not high_memory:
        return None

    severity = (
        "critical"
        if (
            (window and fmean(window) >= 95.0)
            or (memory_latest and memory_latest >= 95.0)
        )
        else "high"
    )

    details: dict[str, float | int] = {"window": len(window) or 1}
    if usage_latest is not None:
        details["usage_latest"] = round(usage_latest, 2)
    if memory_latest is not None:
        details["memory_latest"] = round(memory_latest, 2)

    return PerformanceIssue("gpu_sustained_high", severity, details)


def _detect_disk_pressure(samples: list[float]) -> PerformanceIssue | None:
    if not samples:
        return None

    latest = samples[-1]
    if latest < 85.0:
        return None

    severity = "critical" if latest >= 92.0 else "high"
    return PerformanceIssue(
        "disk_capacity_pressure",
        severity,
        {"latest": round(latest, 2)},
    )


def _detect_underutilization(
    cpu_samples: list[float],
    mem_samples: list[float],
    gpu_samples: list[float],
) -> PerformanceIssue | None:
    if len(cpu_samples) < 3 or not mem_samples:
        return None

    window_length = min(4, len(cpu_samples), len(mem_samples))
    cpu_window = cpu_samples[-window_length:]
    mem_window = mem_samples[-window_length:]
    gpu_window = gpu_samples[-window_length:] if gpu_samples else []

    if (
        max(cpu_window) <= 30.0
        and max(mem_window) <= 60.0
        and (not gpu_window or max(gpu_window) <= 40.0)
    ):
        return PerformanceIssue(
            "workers_underutilized",
            "info",
            {
                "cpu_average": round(fmean(cpu_window), 2),
                "mem_average": round(fmean(mem_window), 2),
                "window": window_length,
            },
        )
    return None
