import asyncio
import logging
from dataclasses import dataclass

import GPUtil
import psutil

logger = logging.getLogger(__name__)


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
