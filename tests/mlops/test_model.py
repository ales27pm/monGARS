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


def test_load_4bit_causal_lm_uses_dtype_kwarg(
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
        dtype,
    ) -> Any:  # noqa: ARG001
        recorded_kwargs.update(
            {
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "quantization_config": quantization_config,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "trust_remote_code": trust_remote_code,
                "dtype": dtype,
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
    assert recorded_kwargs["dtype"] is model_module.torch.float16
    assert "llm_int8_enable_fp32_cpu_offload" not in recorded_kwargs


def test_load_4bit_causal_lm_falls_back_to_torch_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    recorded_kwargs: dict[str, Any] = {}

    def _legacy_from_pretrained(
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
        model_module.AutoModelForCausalLM, "from_pretrained", _legacy_from_pretrained
    )

    model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf", offload_dir=tmp_path)

    assert "torch_dtype" in recorded_kwargs
    assert "dtype" not in recorded_kwargs
    assert recorded_kwargs["torch_dtype"] is model_module.torch.float16
