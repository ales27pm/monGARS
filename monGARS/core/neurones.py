"""Minimal embedding system placeholder."""

from __future__ import annotations


class EmbeddingSystem:
    """Simple embedding system with a no-op database driver."""

    class _NoOpDriver:
        def session(self) -> "EmbeddingSystem._NoOpDriver":
            return self

        async def __aenter__(self) -> "EmbeddingSystem._NoOpDriver":
            return self

        async def __aexit__(
            self, exc_type: type | None, exc: Exception | None, tb: object | None
        ) -> None:
            pass

        async def run(self, *args, **kwargs) -> "EmbeddingSystem._NoOpDriver":
            return self

        async def single(self) -> dict:
            return {"exists": False}

    def __init__(self) -> None:
        self.driver = EmbeddingSystem._NoOpDriver()

    async def encode(self, text: str) -> list[float]:
        return [0.0]
