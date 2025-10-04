from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from monGARS.core import model_slot_manager
from monGARS.core.model_slot_manager import ModelSlotManager
from monGARS.core.persistence import ModelSnapshot


class _DummyModel:
    load_state_calls: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.device = "cuda:0"
        self.config = SimpleNamespace(use_cache=True)

    def eval(self) -> None:  # pragma: no cover - simple setter
        self.config.use_cache = False

    def state_dict(self) -> dict[str, Any]:
        return {"weights": 1}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        self.__class__.load_state_calls.append(
            {"state_dict": state_dict, "strict": strict}
        )
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def generate(self, **_: Any) -> list[list[int]]:
        return [[0, 1, 2]]


class _DummyTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, Any]:
        assert return_tensors == "pt"
        return {"input_ids": prompt}

    def save_pretrained(
        self, path: Any
    ) -> None:  # pragma: no cover - not used directly
        path.mkdir(parents=True, exist_ok=True)

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        return "decoded"


class _DummyFastLanguageModel:
    load_count = 0

    @classmethod
    def from_pretrained(cls, *_: Any, **__: Any) -> tuple[_DummyModel, _DummyTokenizer]:
        cls.load_count += 1
        return _DummyModel(), _DummyTokenizer()

    @staticmethod
    def get_peft_model(model: _DummyModel, **_: Any) -> _DummyModel:
        return model


@pytest.fixture(autouse=True)
def reset_slots() -> None:
    ModelSlotManager._slots.clear()
    _DummyFastLanguageModel.load_count = 0
    _DummyModel.load_state_calls.clear()
    yield
    ModelSlotManager._slots.clear()
    _DummyFastLanguageModel.load_count = 0
    _DummyModel.load_state_calls.clear()


def _apply_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_slot_manager, "FastLanguageModel", _DummyFastLanguageModel
    )
    monkeypatch.setattr(model_slot_manager.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_slot_manager.torch.cuda, "empty_cache", lambda: None)


def test_model_slot_reuses_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_patches(monkeypatch)

    with ModelSlotManager("primary") as (model_a, tok_a):
        assert isinstance(model_a, _DummyModel)
        assert isinstance(tok_a, _DummyTokenizer)

    with ModelSlotManager("primary") as (model_b, tok_b):
        assert model_a is model_b
        assert tok_a is tok_b

    assert _DummyFastLanguageModel.load_count == 1


def test_snapshot_triggered_on_high_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_patches(monkeypatch)

    def _memory_allocated(_: int) -> int:
        return int(7.0 * 1024**3)

    class _Props:
        total_memory = int(8.0 * 1024**3)

    snapshot_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(model_slot_manager.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_slot_manager.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        model_slot_manager.torch.cuda, "memory_allocated", _memory_allocated
    )
    monkeypatch.setattr(
        model_slot_manager.torch.cuda, "get_device_properties", lambda _: _Props()
    )
    monkeypatch.setattr(model_slot_manager.torch.cuda, "empty_cache", lambda: None)

    def _snapshot(model: Any, tokenizer: Any, **kwargs: Any) -> None:
        snapshot_calls.append({"model": model, "tokenizer": tokenizer, **kwargs})

    monkeypatch.setattr(
        model_slot_manager.PersistenceManager,
        "snapshot_model",
        staticmethod(_snapshot),
    )

    with ModelSlotManager("primary") as (model, tokenizer):
        assert isinstance(model, _DummyModel)
        assert isinstance(tokenizer, _DummyTokenizer)

    slot_state = ModelSlotManager._slots["primary"]
    assert slot_state.model is None
    assert slot_state.tokenizer is None
    assert snapshot_calls, "snapshot should be recorded when VRAM threshold exceeded"


def test_restores_from_snapshot_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_patches(monkeypatch)

    fake_snapshot_path = Path("/fake/snapshot")
    state_dict = {"weights": 99}
    restored_tokenizer = _DummyTokenizer()

    monkeypatch.setattr(
        model_slot_manager.PersistenceManager,
        "find_latest_snapshot",
        staticmethod(
            lambda slot_name, base_path=None: fake_snapshot_path
            if slot_name == "primary"
            else None
        ),
    )

    def _fake_load_snapshot(
        snapshot_path: Path,
        *,
        map_location: Any | None = None,
        load_tokenizer: bool = True,
    ) -> ModelSnapshot:
        assert snapshot_path == fake_snapshot_path
        assert map_location == "cpu"
        assert load_tokenizer
        return ModelSnapshot(
            path=snapshot_path,
            state_dict=state_dict,
            tokenizer=restored_tokenizer,
            metadata={"model_id": model_slot_manager._DEFAULT_MODEL_ID},
        )

    monkeypatch.setattr(
        model_slot_manager.PersistenceManager,
        "load_snapshot",
        staticmethod(_fake_load_snapshot),
    )

    load_into_slot_calls: list[None] = []

    original_load_into_slot = ModelSlotManager._load_into_slot

    def _spy_load_into_slot(self: ModelSlotManager, slot: Any) -> None:
        load_into_slot_calls.append(None)
        original_load_into_slot(self, slot)

    monkeypatch.setattr(ModelSlotManager, "_load_into_slot", _spy_load_into_slot)

    with ModelSlotManager("primary") as (model, tokenizer):
        assert isinstance(model, _DummyModel)
        assert tokenizer is restored_tokenizer

    assert not load_into_slot_calls, "expected snapshot restoration to bypass cold load"
    assert _DummyModel.load_state_calls
    assert _DummyModel.load_state_calls[-1]["state_dict"] == state_dict


def test_gpu_stats_via_gputil(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_patches(monkeypatch)

    class _GPU:
        memoryTotal = 8192
        memoryUsed = 4096

    class _GPUModule:
        @staticmethod
        def getGPUs() -> list[_GPU]:
            return [_GPU()]

    monkeypatch.setattr(model_slot_manager, "GPUtil", _GPUModule)

    with ModelSlotManager("primary") as _:
        pass

    slot_state = ModelSlotManager._slots["primary"]
    assert slot_state.last_usage_fraction is not None
    assert slot_state.last_usage_fraction < 0.7
