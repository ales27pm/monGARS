"""Tests for `monGARS.mlops.model` helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from monGARS.mlops import model as model_module


class _DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.hf_device_map = {"model.layers": 0}


class _DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.eos_token_id = 2
        self.eos_token = "</s>"


@pytest.fixture(autouse=True)
def _patch_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_tokenizer_from_pretrained(
        model_id: str, use_fast: bool = True
    ) -> Any:  # noqa: ARG001
        return _DummyTokenizer()

    monkeypatch.setattr(
        model_module.AutoTokenizer, "from_pretrained", _fake_tokenizer_from_pretrained
    )


@pytest.fixture(autouse=True)
def _patch_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeBitsAndBytesConfig:
        def __init__(
            self,
            *,
            load_in_4bit: bool,
            bnb_4bit_use_double_quant: bool,
            bnb_4bit_quant_type: str,
            bnb_4bit_compute_dtype,
            llm_int8_enable_fp32_cpu_offload: bool = False,
        ) -> None:
            self.load_in_4bit = load_in_4bit
            self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
            self.bnb_4bit_quant_type = bnb_4bit_quant_type
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
            self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload

    monkeypatch.setattr(model_module, "BitsAndBytesConfig", _FakeBitsAndBytesConfig)


def test_load_4bit_causal_lm_prefers_torch_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    recorded_kwargs: dict[str, Any] = {}

    def _fake_from_pretrained(
        model_id: str,
        *,
        device_map: dict[str, Any],
        max_memory: dict[Any, str],
        offload_folder: str,
        quantization_config: Any,
        low_cpu_mem_usage: bool,
        trust_remote_code: bool,
        torch_dtype,
    ) -> Any:  # noqa: ARG001
        recorded_kwargs.update(
            {
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
            }
        )
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained
    )

    model, tokenizer = model_module.load_4bit_causal_lm(
        "meta-llama/Llama-2-7b-hf", offload_dir=tmp_path
    )

    assert isinstance(model, _DummyModel)
    assert isinstance(tokenizer, _DummyTokenizer)
    assert recorded_kwargs["torch_dtype"] is model_module.torch.float16
    assert recorded_kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload is True


def test_load_4bit_causal_lm_falls_back_to_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    calls: list[dict[str, Any]] = []

    def _legacy_from_pretrained(
        model_id: str,
        *,
        device_map: dict[str, Any],
        max_memory: dict[Any, str],
        offload_folder: str,
        quantization_config: Any,
        low_cpu_mem_usage: bool,
        trust_remote_code: bool,
        **kwargs,
    ) -> Any:  # noqa: ARG001
        calls.append(kwargs)
        if "torch_dtype" in kwargs:
            raise TypeError("unexpected keyword argument 'torch_dtype'")
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _legacy_from_pretrained
    )

    model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf", offload_dir=tmp_path)

    assert len(calls) == 2
    assert calls[0]["torch_dtype"] is model_module.torch.float16
    assert calls[1]["dtype"] is model_module.torch.float16


def test_load_4bit_causal_lm_handles_legacy_bitsandbytes_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    quant_configs: list[Any] = []

    class _LegacyBitsAndBytesConfig:
        def __init__(
            self,
            *,
            load_in_4bit: bool,
            bnb_4bit_use_double_quant: bool,
            bnb_4bit_quant_type: str,
            bnb_4bit_compute_dtype,
            **kwargs,
        ) -> None:
            if "llm_int8_enable_fp32_cpu_offload" in kwargs:
                raise TypeError("unexpected keyword argument")
            self.load_in_4bit = load_in_4bit
            self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
            self.bnb_4bit_quant_type = bnb_4bit_quant_type
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype

    monkeypatch.setattr(model_module, "BitsAndBytesConfig", _LegacyBitsAndBytesConfig)

    def _fake_from_pretrained(
        *_, quantization_config: Any, **__
    ) -> Any:  # noqa: ANN002
        quant_configs.append(quantization_config)
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained
    )

    model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf", offload_dir=tmp_path)

    assert len(quant_configs) == 1
    cfg = quant_configs[0]
    assert not hasattr(cfg, "llm_int8_enable_fp32_cpu_offload")


def test_load_4bit_causal_lm_sets_tokenizer_pad_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_, **__: _DummyModel(),
    )

    _, tokenizer = model_module.load_4bit_causal_lm(
        "meta-llama/Llama-2-7b-hf", offload_dir=tmp_path
    )

    assert tokenizer.pad_token == tokenizer.eos_token
