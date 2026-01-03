import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from monGARS.config import LLMPooling, LLMQuantization
from monGARS.core.llm_integration import UnifiedLLMRuntime


class FakeUnifiedLLMRuntime:
    """Minimal stub used to document the runtime contract in unit tests."""

    def generate(self, prompt: str, **_: object) -> str:
        return "42"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        return [[0.1, 0.2, 0.3]] * len(texts)


def _make_settings(tmp_path: Path) -> SimpleNamespace:
    model = SimpleNamespace(
        quantize_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        max_new_tokens=32,
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.05,
    )
    llm = SimpleNamespace(
        quantization=LLMQuantization.NF4,
        load_in_4bit=True,
        embedding_pooling=LLMPooling.MEAN,
    )
    return SimpleNamespace(unified_model_dir=tmp_path, model=model, llm=llm)


def test_fake_runtime_contract() -> None:
    runtime = FakeUnifiedLLMRuntime()
    assert runtime.generate("anything") == "42"
    vectors = runtime.embed(["hello"])
    assert vectors == [[0.1, 0.2, 0.3]]


def test_runtime_singleton(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    UnifiedLLMRuntime.reset_for_tests()
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(
        UnifiedLLMRuntime,
        "_load_components",
        lambda self: setattr(self, "_encoder", object()),
    )
    first = UnifiedLLMRuntime.instance(settings)
    second = UnifiedLLMRuntime.instance(settings)
    assert first is second
    UnifiedLLMRuntime.reset_for_tests()


def test_cpu_quantisation_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedLLMRuntime.reset_for_tests()
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = UnifiedLLMRuntime.instance(settings)
    quant_config = runtime._build_quantization_config()
    assert quant_config is None
    UnifiedLLMRuntime.reset_for_tests()


def test_model_quantize_flag_is_respected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedLLMRuntime.reset_for_tests()
    settings = _make_settings(tmp_path)
    settings.model.quantize_4bit = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runtime = UnifiedLLMRuntime.instance(settings)
    assert runtime._build_quantization_config() is None
    UnifiedLLMRuntime.reset_for_tests()


def test_unsupported_quantization_modes_are_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedLLMRuntime.reset_for_tests()
    settings = _make_settings(tmp_path)
    settings.llm.quantization = LLMQuantization.GPTQ
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runtime = UnifiedLLMRuntime.instance(settings)
    assert runtime._build_quantization_config() is None
    UnifiedLLMRuntime.reset_for_tests()


@pytest.mark.skipif(
    os.getenv("CI", "false").lower() == "true",
    reason="Tiny fixture models are only loaded in local environments",
)
def test_runtime_generates_and_embeds_from_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    UnifiedLLMRuntime.reset_for_tests()
    fixture_dir = Path("tests/fixtures/tiny_dolphin").resolve()

    class _FakeEncoding(dict):
        def __init__(self, prompt: str) -> None:
            super().__init__(input_ids=torch.tensor([[len(prompt.split()), 1]]))

        def to(self, *_: object, **__: object) -> "_FakeEncoding":
            return self

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(  # noqa: ARG002
            self, prompt: str, return_tensors: str | None = None
        ) -> _FakeEncoding:
            return _FakeEncoding(prompt)

        def decode(  # noqa: ARG002
            self, tokens, skip_special_tokens: bool = True
        ) -> str:
            return "2+2=4"

    class _FakeLLM2Vec:
        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()

        @staticmethod
        def from_pretrained(path: str, **_: object) -> "_FakeLLM2Vec":
            assert Path(path) == fixture_dir
            return _FakeLLM2Vec()

        def encode(  # noqa: ARG002
            self,
            texts: list[str],
            *,
            batch_size: int,
            show_progress_bar: bool,
            convert_to_tensor: bool,
            device: str,
        ) -> torch.Tensor:
            vectors = torch.ones((len(texts), 4096), dtype=torch.float32)
            return vectors

    class _FakeGenerator:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, path: str, **_: object) -> "_FakeGenerator":
            assert Path(path) == fixture_dir
            return cls()

        def eval(self) -> "_FakeGenerator":
            return self

        def generate(self, **kwargs: object) -> SimpleNamespace:
            tokens = torch.tensor([[1, 2, 3, 4]])
            return SimpleNamespace(sequences=[tokens])

    monkeypatch.setattr(
        "monGARS.core.llm_integration.AutoModelForCausalLM", _FakeGenerator
    )
    monkeypatch.setattr("monGARS.core.llm_integration.AutoTokenizer", _FakeTokenizer)
    monkeypatch.setitem(sys.modules, "llm2vec", SimpleNamespace(LLM2Vec=_FakeLLM2Vec))

    settings = _make_settings(fixture_dir)
    runtime = UnifiedLLMRuntime.instance(settings)
    text = runtime.generate("2+2=")
    assert "4" in text
    embeddings = runtime.embed(["test"])
    assert len(embeddings[0]) == 4096
    UnifiedLLMRuntime.reset_for_tests()
