from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

class _CpuTimes:
    user: float
    system: float

class _VirtualMemory:
    total: int
    available: int
    percent: float

class _DiskUsage:
    total: int
    used: int
    free: int
    percent: float

class Process:
    def __init__(self, pid: int | None = ...) -> None: ...
    def cpu_times(self) -> _CpuTimes: ...

def cpu_count(logical: bool | None = ...) -> int | None: ...
def cpu_percent(
    interval: float | int | None = ..., percpu: bool | None = ...
) -> float | Sequence[float]: ...
def virtual_memory() -> _VirtualMemory: ...
def disk_usage(path: str) -> _DiskUsage: ...
def process_iter(attrs: Iterable[str] | None = ...) -> Iterator[Any]: ...

__all__ = [
    "Process",
    "cpu_count",
    "cpu_percent",
    "virtual_memory",
    "disk_usage",
    "process_iter",
]
