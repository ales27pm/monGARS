import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import GPUtil
import psutil

from monGARS.config import get_settings


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
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def get_system_stats(self) -> SystemStats:
        loop = asyncio.get_running_loop()
        cpu = await loop.run_in_executor(None, psutil.cpu_percent, self.update_interval)
        memory = await loop.run_in_executor(None, psutil.virtual_memory)
        disk = await loop.run_in_executor(None, psutil.disk_usage, "/")
        gpu_stats = await asyncio.get_running_loop().run_in_executor(
            self.executor, self._get_gpu_stats
        )
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
        except Exception:
            pass
        return {"gpu_usage": None, "gpu_memory_usage": None}
