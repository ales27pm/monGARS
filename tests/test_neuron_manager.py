from __future__ import annotations

import math
from typing import Any

import pytest

from modules.neurons.core import NeuronManager


class _DummyModel:
    def __init__(self, base_model: str, encoder_path: str | None) -> None:
        self.base_model = base_model
        self.encoder_path = encoder_path
        self.calls: list[tuple[str, ...]] = []

    @classmethod
    def from_pretrained(
        cls,
        *,
        base_model_name_or_path: str,
        peft_model_name_or_path: str | None = None,
        **_: Any,
    ) -> "_DummyModel":
        return cls(base_model_name_or_path, peft_model_name_or_path)

    def encode(
        self,
        formatted_texts: list[list[str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[list[float]]:  # noqa: D401 - simple helper
        self.calls.append(tuple(text for _, text in formatted_texts))
        return [[float(len(text)), float(batch_size)] for _, text in formatted_texts]


def _factory(base: str, encoder: str | None) -> _DummyModel:
    options = {
        "base_model_name_or_path": base,
        "peft_model_name_or_path": encoder,
    }
    return _DummyModel.from_pretrained(**options)


def test_neuron_manager_uses_fallback_when_model_unavailable() -> None:
    manager = NeuronManager(
        base_model_path="does/not/matter",
        fallback_dimensions=12,
        fallback_cache_size=4,
        llm2vec_factory=lambda *_: None,
    )

    vectors = manager.encode(["bonjour"], instruction="salut")

    assert len(vectors) == 1
    assert len(vectors[0]) == 12
    magnitude = math.sqrt(sum(component**2 for component in vectors[0]))
    assert math.isclose(magnitude, 1.0, rel_tol=1e-6)

    # Deterministic fallback
    assert vectors[0] == manager.encode(["bonjour"], instruction="salut")[0]


def test_neuron_manager_loads_and_encodes_with_custom_factory() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    assert manager.is_ready is True

    outputs = manager.encode(["hello", "world"], instruction="test")
    assert outputs == [[5.0, 8.0], [5.0, 8.0]]

    assert isinstance(manager.model, _DummyModel)
    assert manager.model.calls == [("hello", "world")]


def test_neuron_manager_loads_base_model_without_adapter() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path=None,
        llm2vec_factory=_factory,
    )

    assert manager.is_ready is True
    assert isinstance(manager.model, _DummyModel)
    assert manager.model.encoder_path is None

    outputs = manager.encode(["hello"], instruction="greet")
    assert outputs == [[5.0, 8.0]]


def test_fallback_cache_eviction_behavior() -> None:
    cache_size = 3
    manager = NeuronManager(
        base_model_path="base/model",
        fallback_dimensions=8,
        fallback_cache_size=cache_size,
        llm2vec_factory=lambda *_: None,
    )

    instruction = "test"
    texts = [f"text_{index}" for index in range(cache_size + 2)]
    for text in texts:
        manager.encode([text], instruction=instruction)

    cached_keys = list(
        manager._fallback_cache.keys()
    )  # noqa: SLF001 - intentional inspection
    expected_keys = [(instruction, text) for text in texts[-cache_size:]]
    assert cached_keys == expected_keys

    oldest_key = (instruction, texts[0])
    newest_key = (instruction, texts[-1])
    assert oldest_key not in manager._fallback_cache
    assert newest_key in manager._fallback_cache


def test_switch_encoder_reloads_model() -> None:
    calls: list[tuple[str, str | None]] = []

    def factory(base: str, encoder: str | None) -> _DummyModel:
        calls.append((base, encoder))
        return _DummyModel(base, encoder)

    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="first",
        llm2vec_factory=factory,
    )

    assert calls == [("base/model", "first")]

    manager.switch_encoder("second")
    assert calls[-1] == ("base/model", "second")
    assert isinstance(manager.model, _DummyModel)


def test_encode_fallback_on_model_error() -> None:
    class _ErrorModel:
        def __init__(self, base_model: str, encoder_path: str | None) -> None:
            self.base_model = base_model
            self.encoder_path = encoder_path

        def encode(
            self,
            formatted_texts: list[list[str]],
            *,
            batch_size: int,
            show_progress_bar: bool,
        ) -> list[list[float]]:
            raise RuntimeError("Encoding failed!")

    def error_factory(base: str, encoder: str | None) -> _ErrorModel:
        return _ErrorModel(base, encoder)

    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        fallback_dimensions=6,
        llm2vec_factory=error_factory,
    )

    outputs = manager.encode(["hello", "world"], instruction="test")
    assert len(outputs) == 2
    for vector in outputs:
        assert len(vector) == 6
        magnitude = math.sqrt(sum(component**2 for component in vector))
        assert math.isclose(magnitude, 1.0, rel_tol=1e-6)


def test_invalid_configuration_raises() -> None:
    with pytest.raises(ValueError):
        NeuronManager(base_model_path="base", fallback_dimensions=0)

    with pytest.raises(ValueError):
        NeuronManager(base_model_path="base", fallback_cache_size=0)
