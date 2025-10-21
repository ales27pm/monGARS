"""monGARS package initialisation hooks.

This module provides runtime compatibility shims that need to be in place before
the broader package is imported. The current shim restores the
``PytorchGELUTanh`` activation that AutoAWQ/PEFT expect from older versions of
🤗 Transformers. The symbol was removed upstream in Transformers >= 4.45, which
caused AutoAWQ to fail during import and broke our fine-tuning tests.

By defining the module here we ensure the symbol is available as soon as the
``monGARS`` package loads, keeping the rest of the codebase agnostic of the
underlying dependency change.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from types import ModuleType
from typing import Callable, Literal, cast

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    importlib.import_module("unsloth")
except Exception:  # pragma: no cover - optional dependency missing or failing
    pass

ModuleBase = nn.Module

SimpleFilterAction = Literal[
    "default", "error", "ignore", "always", "all", "module", "once"
]
SimpleFilter = Callable[[SimpleFilterAction, type[Warning], int, bool], None]

_original_simplefilter = cast(SimpleFilter, warnings.simplefilter)


def _awq_safe_simplefilter(
    action: SimpleFilterAction,
    category: type[Warning] = Warning,
    lineno: int = 0,
    append: bool = False,
) -> None:
    if action == "default" and category is DeprecationWarning:
        for frame in inspect.stack():
            if "awq/__init__.py" in frame.filename:
                _original_simplefilter("ignore", category, lineno, append)
                return
    _original_simplefilter(action, category, lineno, append)


warnings.simplefilter = _awq_safe_simplefilter

_transformers_activations = cast(
    ModuleType, importlib.import_module("transformers.activations")
)

if not hasattr(_transformers_activations, "PytorchGELUTanh"):

    class PytorchGELUTanh(ModuleBase):
        """Compatibility shim mirroring the removed Transformers activation."""

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(input_tensor, approximate="tanh")

    setattr(_transformers_activations, "PytorchGELUTanh", PytorchGELUTanh)

    symbols = getattr(_transformers_activations, "__all__", ())
    symbol_list = list(symbols) if symbols is not None else []
    if "PytorchGELUTanh" not in symbol_list:
        symbol_list.append("PytorchGELUTanh")
        setattr(_transformers_activations, "__all__", symbol_list)
