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

import torch
from torch import nn
from transformers import activations as _transformers_activations

if not hasattr(_transformers_activations, "PytorchGELUTanh"):

    class PytorchGELUTanh(nn.Module):
        """Compatibility shim mirroring the removed Transformers activation."""

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(input_tensor, approximate="tanh")

    _transformers_activations.PytorchGELUTanh = PytorchGELUTanh

    symbols = getattr(_transformers_activations, "__all__", None)
    if symbols is not None and "PytorchGELUTanh" not in symbols:
        if isinstance(symbols, tuple):
            _transformers_activations.__all__ = (*symbols, "PytorchGELUTanh")
        else:
            symbols.append("PytorchGELUTanh")
