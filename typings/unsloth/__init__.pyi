from __future__ import annotations

from typing import Any, Tuple

class FastLanguageModel:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Tuple[Any, Any]: ...
    @classmethod
    def get_peft_model(cls, *args: Any, **kwargs: Any) -> Any: ...
    @classmethod
    def merge_and_unload(cls, *args: Any, **kwargs: Any) -> Any: ...

__all__ = ["FastLanguageModel"]
