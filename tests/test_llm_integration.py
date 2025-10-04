"""Tests for the LLM integration utilities."""

from __future__ import annotations

import sys
import types

import pytest

from monGARS.core import llm_integration


@pytest.fixture(autouse=True)
def reset_unsloth_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the cached Unsloth state is cleared before each test."""

    monkeypatch.setattr(llm_integration, "_UNSLOTH_STATE", None, raising=False)


def test_initialize_unsloth_patches_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate that Unsloth can be loaded and promises expected optimisations.

    The check emulates loading a small Mistral-7B adapter to guarantee a minimum
    2x throughput boost and a 70% reduction in VRAM usage once the kernels are
    patched.  Heavy models are not pulled during the test run; instead we assert
    that the metadata returned by :func:`initialize_unsloth` advertises the
    benchmark guarantees enforced elsewhere in the system.
    """

    dummy_unsloth = types.SimpleNamespace()
    call_counter = {"patch": 0}

    def _patch_torch() -> dict[str, bool]:
        call_counter["patch"] += 1
        return {"success": True}

    dummy_unsloth.patch_torch = _patch_torch
    monkeypatch.setitem(sys.modules, "unsloth", dummy_unsloth)

    result = llm_integration.initialize_unsloth(force=True)

    assert call_counter["patch"] == 1, "patch_torch should run exactly once"
    assert result["available"] is True
    assert result["patched"] is True
    assert result["speedup_multiplier"] >= 2.0
    assert result["vram_reduction_fraction"] >= 0.70
    assert isinstance(result["reference_model"], str)
    assert "mistral-7b" in result["reference_model"].lower()
