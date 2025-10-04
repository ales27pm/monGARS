import asyncio
import logging
from dataclasses import dataclass

import GPUtil
import psutil
from opentelemetry import metrics

from .ui_events import event_bus, make_event

logger = logging.getLogger(__name__)

meter = metrics.get_meter(__name__)
TRAINING_CYCLE_COUNTER = meter.create_counter(
    "llm.training.cycles",
    description="Number of MNTP training cycles started and completed.",
)
TRAINING_FAILURE_COUNTER = meter.create_counter(
    "llm.training.failures",
    description="Count of MNTP training cycles that failed.",
)
TRAINING_TOKEN_COUNTER = meter.create_counter(
    "llm.training.tokens",
    unit="token",
    description="Approximate number of tokens processed during MNTP fine-tuning.",
)


@dataclass
class SystemStats:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: float = None
    gpu_memory_usage: float = None


class SystemMonitor:
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval

    async def get_system_stats(self) -> SystemStats:
        cpu = await asyncio.to_thread(psutil.cpu_percent, self.update_interval)
        memory = await asyncio.to_thread(psutil.virtual_memory)
        disk = await asyncio.to_thread(psutil.disk_usage, "/")
        gpu_stats = await asyncio.to_thread(self._get_gpu_stats)
        return SystemStats(
            cpu_usage=cpu,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            gpu_usage=gpu_stats.get("gpu_usage"),
            gpu_memory_usage=gpu_stats.get("gpu_memory_usage"),
        )

    def _get_gpu_stats(self) -> dict:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "gpu_usage": min(gpu.load * 100, 85),
                    "gpu_memory_usage": gpu.memoryUtil * 100,
                }
        except Exception as exc:  # pragma: no cover - optional GPU dependency
            logger.exception("Failed to query GPU stats", exc_info=exc)
        return {"gpu_usage": None, "gpu_memory_usage": None}


async def maybe_alert(
    user_id: str | None = None,
    cpu: float | None = None,
    ttfb_ms: int | None = None,
) -> None:
    """Publish performance alerts when metrics are provided."""

    data: dict[str, float | int] = {}
    if cpu is not None:
        data["cpu"] = cpu
    if ttfb_ms is not None:
        data["ttfb_ms"] = ttfb_ms
    if not data:
        return
    await event_bus().publish(make_event("performance.alert", user_id, data))


__all__ = [
    "SystemMonitor",
    "SystemStats",
    "maybe_alert",
    "TRAINING_CYCLE_COUNTER",
    "TRAINING_FAILURE_COUNTER",
    "TRAINING_TOKEN_COUNTER",
]
