from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from modules.neurons.core import NeuronManager


def _create_wrapper(tmp_path: Path) -> Path:
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    (wrapper_dir / "project_wrapper.py").write_text(
        "class ChatAndEmbed:\n"
        "    def embed(self, texts):\n"
        "        if isinstance(texts, str):\n"
        "            texts = [texts]\n"
        "        return [[float(len(text)), 1.0] for text in texts]\n"
    )
    (wrapper_dir / "config.json").write_text(
        json.dumps(
            {
                "base_model_id": "base-model",
                "lora_dir": (tmp_path / "adapter").as_posix(),
                "max_seq_len": 256,
                "quantized_4bit": True,
                "vram_budget_mb": 4096,
                "offload_dir": (tmp_path / "offload").as_posix(),
            }
        )
    )
    return wrapper_dir


class _DummyModel:
    def __init__(self, base_model: str, encoder_path: str | None) -> None:
        self.base_model = base_model
        self.encoder_path = encoder_path
        self.calls: list[tuple[str, ...]] = []
        self.formatted_texts: list[list[str]] | None = None
        self.last_kwargs: dict[str, Any] | None = None

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
        **kwargs: Any,
    ) -> list[list[float]]:  # noqa: D401 - simple helper
        self.formatted_texts = [list(item) for item in formatted_texts]
        self.calls.append(tuple(text for _, text in formatted_texts))
        self.last_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            **kwargs,
        }
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
    assert manager.model.formatted_texts == [["test", "hello"], ["test", "world"]]
    assert manager.model.last_kwargs == {"batch_size": 8, "show_progress_bar": False}


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


def test_neuron_manager_uses_wrapper_bundle(tmp_path: Path) -> None:
    wrapper_dir = _create_wrapper(tmp_path)
    manager = NeuronManager(
        base_model_path="base-model",
        default_encoder_path=None,
        wrapper_dir=str(wrapper_dir),
        llm2vec_factory=lambda *_: None,
    )

    vectors = manager.encode(["hello"], instruction="")
    assert vectors == [[5.0, 1.0]]
    assert manager.encoder_path == (tmp_path / "adapter").as_posix()


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


def test_string_torch_dtype_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _StubLLM2Vec:
        @staticmethod
        def from_pretrained(**options: Any) -> "_StubLLM2Vec":
            captured["options"] = options
            return _StubLLM2Vec()

    class _StubTorch:
        bfloat16 = object()

    monkeypatch.setattr("modules.neurons.core.LLM2Vec", _StubLLM2Vec)
    monkeypatch.setattr("modules.neurons.core._get_torch_module", lambda: _StubTorch())

    manager = NeuronManager(base_model_path="base/model", default_encoder_path="enc")

    assert isinstance(manager.model, _StubLLM2Vec)
    assert captured["options"]["dtype"] is _StubTorch.bfloat16


def test_invalid_string_torch_dtype_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _StubLLM2Vec:
        @staticmethod
        def from_pretrained(**options: Any) -> "_StubLLM2Vec":
            captured["options"] = options
            return _StubLLM2Vec()

    class _StubTorch:
        pass

    monkeypatch.setattr("modules.neurons.core.LLM2Vec", _StubLLM2Vec)
    monkeypatch.setattr("modules.neurons.core._get_torch_module", lambda: _StubTorch())

    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="enc",
        llm2vec_options={"torch_dtype": "unknown_type"},
    )

    assert isinstance(manager.model, _StubLLM2Vec)
    assert "dtype" not in captured["options"]


def test_dotted_torch_dtype_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _StubLLM2Vec:
        @staticmethod
        def from_pretrained(**options: Any) -> "_StubLLM2Vec":
            captured["options"] = options
            return _StubLLM2Vec()

    class _StubTorch:
        float32 = object()

    monkeypatch.setattr("modules.neurons.core.LLM2Vec", _StubLLM2Vec)
    monkeypatch.setattr("modules.neurons.core._get_torch_module", lambda: _StubTorch())

    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="enc",
        llm2vec_options={"torch_dtype": "torch.float32"},
    )

    assert isinstance(manager.model, _StubLLM2Vec)
    assert captured["options"]["dtype"] is _StubTorch.float32


def test_auto_torch_dtype_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _StubLLM2Vec:
        @staticmethod
        def from_pretrained(**options: Any) -> "_StubLLM2Vec":
            captured["options"] = options
            return _StubLLM2Vec()

    monkeypatch.setattr("modules.neurons.core.LLM2Vec", _StubLLM2Vec)
    monkeypatch.setattr("modules.neurons.core._get_torch_module", lambda: None)

    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="enc",
        llm2vec_options={"torch_dtype": "auto"},
    )

    assert isinstance(manager.model, _StubLLM2Vec)
    assert captured["options"].get("dtype") == "auto"


def test_encode_allows_instruction_sequence() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    outputs = manager.encode(["hello", "world"], instruction=["one", "two"])
    assert outputs == [[5.0, 8.0], [5.0, 8.0]]
    assert manager.model is not None
    assert manager.model.formatted_texts == [["one", "hello"], ["two", "world"]]


def test_encode_supports_preformatted_prompts() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    prompts = [("inst-1", "hello"), ("inst-2", "world")]
    outputs = manager.encode(prompts)
    assert outputs == [[5.0, 8.0], [5.0, 8.0]]
    assert manager.model is not None
    assert manager.model.formatted_texts == [["inst-1", "hello"], ["inst-2", "world"]]


def test_encode_rejects_instruction_with_preformatted_prompts() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    with pytest.raises(ValueError):
        manager.encode([("one", "hello")], instruction="oops")


def test_encode_rejects_mismatched_instruction_list() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    with pytest.raises(ValueError):
        manager.encode(["hello", "world"], instruction=["only-one"])


def test_encode_rejects_non_sequence_input() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    with pytest.raises(TypeError):
        manager.encode("hello")  # type: ignore[arg-type]


def test_encode_raises_on_invalid_batch_size() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=lambda *_: None,
    )

    with pytest.raises(ValueError):
        manager.encode(["hello"], batch_size=0)


def test_encode_kwargs_override_defaults() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
        encode_options={"batch_size": 4, "show_progress_bar": True},
    )

    manager.encode(["hello"], convert_to_numpy=True, batch_size=2)

    assert manager.model is not None
    assert manager.model.last_kwargs == {
        "batch_size": 2,
        "show_progress_bar": True,
        "convert_to_numpy": True,
    }


def test_set_encode_options_updates_defaults() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    manager.set_encode_options(batch_size=3, show_progress_bar=True)
    manager.encode(["hello"])

    assert manager.model is not None
    assert manager.model.last_kwargs == {"batch_size": 3, "show_progress_bar": True}


def test_set_encode_options_removes_option_with_none() -> None:
    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=_factory,
    )

    # Set an option
    manager.set_encode_options(batch_size=4, show_progress_bar=True)
    manager.encode(["test"])
    assert manager.model.last_kwargs == {"batch_size": 4, "show_progress_bar": True}

    # Remove batch_size by setting it to None
    manager.set_encode_options(batch_size=None)
    manager.encode(["test2"])
    # Only show_progress_bar should remain
    assert manager.model.last_kwargs == {"show_progress_bar": True}


def test_reload_after_initial_failure() -> None:
    attempts: list[str | None] = []

    def flaky_factory(base: str, encoder: str | None) -> _DummyModel:
        attempts.append(encoder)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return _DummyModel(base, encoder)

    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
        llm2vec_factory=flaky_factory,
    )

    assert manager.model is None

    # First call falls back
    fallback = manager.encode(["hello"], instruction="test")
    assert len(fallback) == 1

    manager.reload()
    assert isinstance(manager.model, _DummyModel)

    outputs = manager.encode(["hello"], instruction="test")
    assert outputs == [[5.0, 8.0]]
    assert attempts == ["adapter/path", "adapter/path"]


def test_missing_from_pretrained_gracefully_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubLLM2Vec:
        pass

    monkeypatch.setattr("modules.neurons.core.LLM2Vec", _StubLLM2Vec)

    manager = NeuronManager(
        base_model_path="base/model",
        default_encoder_path="adapter/path",
    )

    assert manager.model is None
