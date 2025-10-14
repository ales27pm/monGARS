from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pytest


class FakeTensor:
    """Minimal tensor stub supporting the ops used by the wrapper."""

    __array_priority__ = 1000

    def __init__(self, data: Iterable[float] | np.ndarray, *, device: str = "cpu") -> None:
        self._data = np.array(data, dtype=float)
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def to(self, device: str) -> "FakeTensor":
        self.device = device
        return self

    def unsqueeze(self, dim: int) -> "FakeTensor":
        if dim < 0:
            dim = self._data.ndim + dim + 1
        expanded = np.expand_dims(self._data, axis=dim)
        return FakeTensor(expanded, device=self.device)

    def sum(self, dim: int | None = None) -> "FakeTensor":
        if dim is None:
            summed = self._data.sum()
        else:
            summed = self._data.sum(axis=dim)
        return FakeTensor(summed, device=self.device)

    def clamp(self, *, min: float | None = None) -> "FakeTensor":
        data = self._data
        if min is not None:
            data = np.maximum(data, min)
        return FakeTensor(data, device=self.device)

    def cpu(self) -> "FakeTensor":
        return FakeTensor(self._data, device="cpu")

    def numpy(self) -> np.ndarray:
        return np.array(self._data, copy=True)

    def __mul__(self, other: Any) -> "FakeTensor":
        if isinstance(other, FakeTensor):
            product = self._data * other._data
        else:
            product = self._data * other
        return FakeTensor(product, device=self.device)

    def __truediv__(self, other: Any) -> "FakeTensor":
        if isinstance(other, FakeTensor):
            quotient = self._data / other._data
        else:
            quotient = self._data / other
        return FakeTensor(quotient, device=self.device)

    def __array__(self) -> np.ndarray:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"FakeTensor(shape={self._data.shape}, device={self.device})"


class FakeBatchEncoding(dict[str, FakeTensor]):
    def to(self, device: str) -> "FakeBatchEncoding":
        for key, value in list(self.items()):
            self[key] = value.to(device)
        return self


class FakeOutput:
    def __init__(
        self,
        *,
        hidden_states: list[FakeTensor] | None = None,
        last_hidden_state: FakeTensor | None = None,
    ) -> None:
        self.hidden_states = hidden_states
        self.last_hidden_state = last_hidden_state


class FakeModel:
    hidden_size = 4

    def __init__(self) -> None:
        self.fail_direct = False
        self.return_last_only = False
        self.forward_calls: list[dict[str, Any]] = []
        self.base_model: FakeModel | None = None
        self.model: FakeModel | None = None
        self.transformer: FakeModel | None = None
        self.last_hidden_data: np.ndarray | None = None

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def __call__(self, *, input_ids: FakeTensor, attention_mask: FakeTensor, **kwargs: Any) -> FakeOutput:
        if self.fail_direct:
            raise TypeError("direct call unsupported")
        self.forward_calls.append(kwargs)
        batch, seq = input_ids.shape
        data = np.arange(batch * seq * self.hidden_size, dtype=float).reshape(batch, seq, self.hidden_size)
        hidden = FakeTensor(data + 1, device=input_ids.device)
        self.last_hidden_data = hidden.numpy()
        if self.return_last_only:
            return FakeOutput(hidden_states=None, last_hidden_state=hidden)
        return FakeOutput(hidden_states=[FakeTensor(np.zeros_like(hidden.numpy())), hidden])


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.eos_token = 2

    @classmethod
    def from_pretrained(cls, path: str, use_fast: bool = True) -> "FakeTokenizer":
        tokenizer = cls()
        tokenizer.path = path
        return tokenizer

    def __call__(self, texts: Iterable[str], **_: Any) -> FakeBatchEncoding:
        if isinstance(texts, str):
            batch_texts = [texts]
        else:
            batch_texts = list(texts)
        batch = len(batch_texts)
        seq_len = 3
        input_ids = FakeTensor(np.arange(batch * seq_len).reshape(batch, seq_len))
        attention_mask = FakeTensor(np.ones((batch, seq_len)))
        return FakeBatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })


class FakeAutoModelForCausalLM:
    last_created: FakeModel | None = None

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeModel:
        model = FakeModel()
        cls.last_created = model
        return model


class FakePeftModel:
    @staticmethod
    def from_pretrained(model: FakeModel, adapter_dir: str) -> FakeModel:
        model.adapter_dir = adapter_dir
        return model


@pytest.fixture()
def llm2vec_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_torch = ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

    def _inference_mode():
        def decorator(fn):
            def wrapper(*args: Any, **kwargs: Any):
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    fake_torch.inference_mode = _inference_mode

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = FakeTokenizer

    fake_peft = ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    sys.modules.pop("scripts.export_llm2vec_wrapper", None)
    module = importlib.import_module("scripts.export_llm2vec_wrapper")

    namespace: dict[str, Any] = {}
    exec(module.WRAPPER_PY, namespace)
    llm_cls = namespace["LLM2Vec"]

    base_dir = tmp_path / "model"
    (base_dir / "tokenizer").mkdir(parents=True)
    (base_dir / "merged").mkdir(parents=True)

    return {
        "LLM2Vec": llm_cls,
        "base_dir": base_dir,
        "auto_model_cls": FakeAutoModelForCausalLM,
        "model_factory": FakeModel,
    }


def test_embed_mean_pools_hidden_states(llm2vec_fixture: dict[str, Any]) -> None:
    llm_cls = llm2vec_fixture["LLM2Vec"]
    base_dir = llm2vec_fixture["base_dir"]
    auto_model_cls = llm2vec_fixture["auto_model_cls"]

    wrapper = llm_cls(base_dir=base_dir, prefer_merged=True, device="cpu", load_in_4bit=False)

    embeddings = wrapper.embed(["alpha", "beta"])
    expected = auto_model_cls.last_created.last_hidden_data.mean(axis=1)
    np.testing.assert_allclose(embeddings.numpy(), expected)

    for call_kwargs in auto_model_cls.last_created.forward_calls:
        assert call_kwargs["output_hidden_states"] is True
        assert call_kwargs["use_cache"] is False
        assert call_kwargs["return_dict"] is True


def test_embed_falls_back_to_base_model(llm2vec_fixture: dict[str, Any]) -> None:
    llm_cls = llm2vec_fixture["LLM2Vec"]
    base_dir = llm2vec_fixture["base_dir"]
    auto_model_cls = llm2vec_fixture["auto_model_cls"]
    model_factory = llm2vec_fixture["model_factory"]

    wrapper = llm_cls(base_dir=base_dir, prefer_merged=True, device="cpu", load_in_4bit=False)

    primary = auto_model_cls.last_created
    fallback = model_factory()
    fallback.return_last_only = True
    primary.fail_direct = True
    primary.base_model = fallback

    result = wrapper.embed(["single"])
    assert fallback.forward_calls, "fallback model should receive the forward call"
    expected = fallback.last_hidden_data.mean(axis=1)
    np.testing.assert_allclose(result.numpy(), expected)


def test_embed_raises_for_missing_hidden_states(llm2vec_fixture: dict[str, Any]) -> None:
    llm_cls = llm2vec_fixture["LLM2Vec"]
    base_dir = llm2vec_fixture["base_dir"]
    auto_model_cls = llm2vec_fixture["auto_model_cls"]

    wrapper = llm_cls(base_dir=base_dir, prefer_merged=True, device="cpu", load_in_4bit=False)

    class BrokenOutput:
        hidden_states = None
        last_hidden_state = None

    class BrokenModel(FakeModel):
        def __call__(self, *args: Any, **kwargs: Any) -> BrokenOutput:  # type: ignore[override]
            return BrokenOutput()

    auto_model_cls.last_created.fail_direct = True
    auto_model_cls.last_created.base_model = BrokenModel()

    with pytest.raises(RuntimeError):
        wrapper.embed(["noop"])


def test_embed_rejects_empty_inputs(llm2vec_fixture: dict[str, Any]) -> None:
    llm_cls = llm2vec_fixture["LLM2Vec"]
    base_dir = llm2vec_fixture["base_dir"]

    wrapper = llm_cls(base_dir=base_dir, prefer_merged=True, device="cpu", load_in_4bit=False)

    with pytest.raises(ValueError):
        wrapper.embed([])
