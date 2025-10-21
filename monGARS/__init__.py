"""monGARS package initialisation hooks.

This module provides runtime compatibility shims that need to be in place before
the broader package is imported. The current shim restores the
``PytorchGELUTanh`` activation that AutoAWQ/PEFT expect from older versions of
Hugging Face Transformers. The symbol was removed upstream in Transformers >= 4.45,
which caused AutoAWQ to fail during import and broke our fine-tuning tests.

By defining the module here we ensure the symbol is available as soon as the
``monGARS`` package loads, keeping the rest of the codebase agnostic of the
underlying dependency change.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import warnings
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence, cast


def _is_static_analysis_process(argv: Sequence[str], env: Mapping[str, str]) -> bool:
    """Best-effort detection for type-checker invocations.

    The previous heuristic only inspected the executable name, which missed
    `python -m mypy` and language-server launches. Those cases pulled optional
    dependencies (Torch, Unsloth) into type-checker sessions and caused
    aggressive startup delays. This helper cross-references common environment
    flags alongside the argument vector so static analyzers never trigger
    heavy imports.
    """

    if not argv and not env:
        return False

    analysis_env_flags = (
        "MYPY",
        "MYPY_FORCE_COLOR",
        "MYPY_FORCE_TERMINAL_WIDTH",
        "PYRIGHT_VERSION_INFO",
        "PYTYPE_ANALYZE",
    )
    if any(flag in env for flag in analysis_env_flags):
        return True

    analyzer_tokens = {
        "dmypy",
        "mypy",
        "pyre",
        "pyright",
        "stubgen",
        "stubtest",
    }

    for arg in argv:
        if arg == "-m":
            continue
        token = os.path.basename(arg) if arg else ""
        if not token:
            continue
        if token in analyzer_tokens:
            return True
        if token.endswith((".exe", ".py")):
            stripped = token.rsplit(".", 1)[0]
            if stripped in analyzer_tokens:
                return True

    return False


_RUNNING_STATIC_ANALYSIS = _is_static_analysis_process(tuple(sys.argv), os.environ)

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

PytorchGELUTanh: type[Any] | None
ModuleBase: type[Any] | None

torch_module: ModuleType | None
torch_nn: ModuleType | None

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn as _torch_nn_module

    class _PytorchGELUTanhStub(_torch_nn_module.Module):
        """Typing stub mirroring the runtime activation shim."""

        def forward(self, input_tensor: "torch.Tensor") -> "torch.Tensor": ...

    PytorchGELUTanh = _PytorchGELUTanhStub
    torch_module = cast(ModuleType, torch)
    torch_nn = cast(ModuleType, _torch_nn_module)
    ModuleBase = cast("type[Any]", _torch_nn_module.Module)
else:  # pragma: no cover - optional dependency handling
    torch_module = None
    torch_nn = None
    ModuleBase = None
    PytorchGELUTanh = None

    if not _RUNNING_STATIC_ANALYSIS and importlib.util.find_spec("torch") is not None:
        try:
            torch_module = cast(ModuleType, importlib.import_module("torch"))
            torch_nn = cast(ModuleType, importlib.import_module("torch.nn"))
        except Exception:
            torch_module = None
            torch_nn = None

    if not _RUNNING_STATIC_ANALYSIS:
        try:  # pragma: no cover - optional dependency
            importlib.import_module("unsloth")
        except Exception:  # pragma: no cover - optional dependency missing or failing
            pass

    if torch_nn is not None:
        module_candidate = getattr(torch_nn, "Module", None)
        if isinstance(module_candidate, type):
            ModuleBase = cast("type[Any]", module_candidate)

    def _load_transformers_activations() -> ModuleType | None:
        """Safely resolve the Transformers activation registry."""

        try:
            return cast(ModuleType, importlib.import_module("transformers.activations"))
        except ModuleNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - defensive guard
            warnings.warn(
                f"Failed to import transformers.activations ({exc!r}); "
                "PytorchGELUTanh shim will not be installed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    transformers_activations: ModuleType | None = None
    if ModuleBase is not None and torch_module is not None:
        transformers_activations = _load_transformers_activations()

    if (
        ModuleBase is not None
        and torch_module is not None
        and transformers_activations is not None
        and not hasattr(transformers_activations, "PytorchGELUTanh")
    ):
        activations_module = cast(Any, transformers_activations)
        torch_api = cast(Any, torch_module)

        class _RuntimePytorchGELUTanh(ModuleBase):
            """Compatibility shim mirroring the removed Transformers activation."""

            def forward(self, input_tensor: "torch.Tensor") -> "torch.Tensor":
                gelu_fn = getattr(torch_api.nn.functional, "gelu")
                return gelu_fn(input_tensor, approximate="tanh")

        PytorchGELUTanh = _RuntimePytorchGELUTanh
        activations_module.PytorchGELUTanh = _RuntimePytorchGELUTanh

        symbol_list = list(getattr(activations_module, "__all__", ()))
        if "PytorchGELUTanh" not in symbol_list:
            symbol_list.append("PytorchGELUTanh")
            activations_module.__all__ = symbol_list
