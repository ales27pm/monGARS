from __future__ import annotations

from typing import Any, Iterable, Sequence

class LLM2Vec:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> LLM2Vec: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def encode(
        self,
        texts: Sequence[str],
        *,
        instruction: str | None = ...,
        batch_size: int | None = ...,
        device: str | None = ...,
        **kwargs: Any,
    ) -> list[list[float]]: ...
    def close(self) -> None: ...

__all__ = ["LLM2Vec"]
