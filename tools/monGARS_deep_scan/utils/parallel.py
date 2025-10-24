from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    items: Sequence[T],
    func: Callable[[T], R],
    max_workers: int,
) -> Iterator[R]:
    if max_workers <= 1:
        for item in items:
            yield func(item)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(func, item): item for item in items}
        for future in as_completed(future_map):
            yield future.result()
