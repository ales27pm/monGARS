from __future__ import annotations

from typing import Any, Sequence

class SamplingParams:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LLM:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

__all__ = ["LLM", "SamplingParams"]
