import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
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
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def get_system_stats(self) -> SystemStats:
        loop = asyncio.get_running_loop()
        cpu = await loop.run_in_executor(
            self._executor, psutil.cpu_percent, self.update_interval
        )
        memory = await loop.run_in_executor(self._executor, psutil.virtual_memory)
        disk = await loop.run_in_executor(self._executor, psutil.disk_usage, "/")
        gpu_stats = await loop.run_in_executor(self._executor, self._get_gpu_stats)
        return SystemStats(
            cpu_usage=cpu,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            gpu_usage=gpu_stats.get("gpu_usage"),
            gpu_memory_usage=gpu_stats.get("gpu_memory_usage"),
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

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
