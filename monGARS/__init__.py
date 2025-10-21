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
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

torch_module: ModuleType | None
torch_nn: ModuleType | None

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch as torch_module
    from torch import nn as torch_nn
else:  # pragma: no cover - optional dependency handling
    try:
        import torch as torch_module  # type: ignore[import-not-found]
        from torch import nn as torch_nn  # type: ignore[import-not-found]
    except ImportError:
        torch_module = None
        torch_nn = None

try:  # pragma: no cover - optional dependency
    importlib.import_module("unsloth")
except Exception:  # pragma: no cover - optional dependency missing or failing
    pass

ModuleBase: type[Any] | None
if TYPE_CHECKING:
    ModuleBase = torch_nn.Module
else:
    ModuleBase = torch_nn.Module if torch_nn is not None else None

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

if (
    ModuleBase is not None
    and torch_module is not None
    and not hasattr(_transformers_activations, "PytorchGELUTanh")
):

    class PytorchGELUTanh(ModuleBase):
        """Compatibility shim mirroring the removed Transformers activation."""

        def forward(self, input_tensor: "torch.Tensor") -> "torch.Tensor":
            return torch_module.nn.functional.gelu(  # type: ignore[union-attr]
                input_tensor,
                approximate="tanh",
            )

    _transformers_activations.PytorchGELUTanh = PytorchGELUTanh

    symbol_list = list(getattr(_transformers_activations, "__all__", ()))
    if "PytorchGELUTanh" not in symbol_list:
        symbol_list.append("PytorchGELUTanh")
        _transformers_activations.__all__ = symbol_list
