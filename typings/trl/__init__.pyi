from __future__ import annotations

from typing import Any, Protocol

class SFTConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class _TrainerProtocol(Protocol):
    def train(self) -> Any: ...

class SFTTrainer:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def train(self) -> Any: ...

__all__ = ["SFTConfig", "SFTTrainer"]
