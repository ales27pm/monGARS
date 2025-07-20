"""Minimal embedding system placeholder."""

from __future__ import annotations


class EmbeddingSystem:
    """Simple embedding system with a no-op database driver."""

    class _NoOpResult:
        async def single(self) -> dict:
            return {"exists": False}

    class _NoOpSession:
        async def __aenter__(self) -> "EmbeddingSystem._NoOpSession":
            return self

        async def __aexit__(
            self,
            exc_type: type | None,
            exc: Exception | None,
            tb: object | None,
        ) -> None:
            return None

        async def run(self, *args, **kwargs) -> "EmbeddingSystem._NoOpResult":
            return EmbeddingSystem._NoOpResult()

    class _NoOpDriver:
        def session(self) -> "EmbeddingSystem._NoOpSession":
            return EmbeddingSystem._NoOpSession()

    def __init__(self) -> None:
        self.driver = EmbeddingSystem._NoOpDriver()

    async def encode(self, text: str) -> list[float]:
        return [0.0]
