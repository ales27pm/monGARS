"""Tests for helpers in :mod:`monGARS.mlops.model`."""

from __future__ import annotations

from typing import Any

import pytest

from monGARS.mlops import model as model_mod


class _DummyModel:
    """Minimal stand-in for a quantised model."""

    def __init__(self) -> None:
        self.device: str | None = None
        self.hf_device_map = {"layer": "cuda:0", "lm_head": "cpu"}

    def to(self, device: str, *args: Any, **kwargs: Any) -> "_DummyModel":
        self.device = device
        return self

    def children(self):  # pragma: no cover - no nested modules in the dummy
        return []


def test_move_to_cpu_strips_hooks_and_updates_device_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[tuple[Any, bool]] = []

    def _fake_remove(module: Any, *, recurse: bool = False) -> None:
        removed.append((module, recurse))

    monkeypatch.setattr(model_mod, "_ACCELERATE_REMOVE_HOOK", _fake_remove)

    model = _DummyModel()
    model_mod.move_to_cpu(model)

    assert model.device == "cpu"
    assert removed == [(model, True)]
    assert set(model.hf_device_map.values()) == {"cpu"}


def test_move_to_cpu_handles_missing_accelerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Hook:
        def __init__(self) -> None:
            self.detached = False

        def detach_hook(self, module: Any) -> None:
            self.detached = True

    hook = _Hook()
    model = _DummyModel()
    model._hf_hook = hook  # type: ignore[attr-defined]
    model._old_forward = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    monkeypatch.setattr(model_mod, "_ACCELERATE_REMOVE_HOOK", None)

    model_mod.move_to_cpu(model)

    assert model.device == "cpu"
    assert not hasattr(model, "_hf_hook")
    assert not hasattr(model, "_old_forward")
    assert hook.detached is True
