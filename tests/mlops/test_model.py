"""Tests for `monGARS.mlops.model` helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from packaging.version import Version

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
        model_id: str, use_fast: bool = True, **__: Any
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


@pytest.fixture(autouse=True)
def _force_quantization_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: True)


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
    assert (
        recorded_kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload is True
    )
    assert recorded_kwargs["max_memory"][0] == "5308MiB"


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


def test_load_4bit_causal_lm_reserves_activation_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    recorded: dict[str, Any] = {}

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
        recorded.update(
            {
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "torch_dtype": torch_dtype,
                "quantization_config": quantization_config,
            }
        )
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained
    )

    model_module.load_4bit_causal_lm(
        "meta-llama/Llama-2-7b-hf",
        vram_budget_mb=5000,
        activation_buffer_mb=1500,
        offload_dir=tmp_path,
    )

    assert recorded["max_memory"][0] == "2732MiB"


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
    ) -> Any:
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


def test_load_4bit_causal_lm_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: False)

    recorded_kwargs: dict[str, Any] = {}

    def _fake_from_pretrained(*_, **kwargs) -> Any:  # noqa: ANN002
        recorded_kwargs.update(kwargs)
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained
    )

    model, tokenizer = model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf")

    assert isinstance(model, _DummyModel)
    assert isinstance(tokenizer, _DummyTokenizer)
    assert recorded_kwargs.get("quantization_config") is None
    assert recorded_kwargs.get("device_map") is None
    assert recorded_kwargs.get("torch_dtype") is model_module.torch.float32


def test_load_4bit_causal_lm_cpu_fallback_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: False)

    def _failing_from_pretrained(*_, **__):  # noqa: ANN002
        raise RuntimeError("Simulated model loading failure")

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _failing_from_pretrained
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="Simulated model loading failure"):
            model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf")

    assert any(
        "Failed to load model on CPU fallback" in message for message in caplog.messages
    )


def test_build_model_kwargs_candidates_skips_unsupported() -> None:
    def _loader(model_id: str, *, low_cpu_mem_usage: bool) -> Any:  # noqa: ARG001
        return _DummyModel()

    candidates = model_module._build_model_kwargs_candidates(
        _loader,
        base_kwargs={"low_cpu_mem_usage": True},
        optional_kwargs=(
            {"torch_dtype": model_module.torch.float32},
            {"dtype": model_module.torch.float32},
        ),
    )

    assert candidates == [{"low_cpu_mem_usage": True}]


def test_build_model_kwargs_candidates_prefers_first_supported() -> None:
    def _loader(
        model_id: str,
        *,
        low_cpu_mem_usage: bool,
        torch_dtype: Any = None,
        dtype: Any = None,
    ) -> Any:  # noqa: ARG001
        return _DummyModel()

    candidates = model_module._build_model_kwargs_candidates(
        _loader,
        base_kwargs={"low_cpu_mem_usage": True},
        optional_kwargs=(
            {"torch_dtype": model_module.torch.float16},
            {"dtype": model_module.torch.float16},
        ),
    )

    assert len(candidates) == 3
    assert candidates[0]["torch_dtype"] is model_module.torch.float16
    assert "dtype" not in candidates[0]
    assert candidates[1]["dtype"] is model_module.torch.float16
    assert candidates[2] == {"low_cpu_mem_usage": True}


def test_get_min_bitsandbytes_version_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONGARS_MIN_BITSANDBYTES_VERSION", "0.45.0")

    assert model_module._get_min_bitsandbytes_version() == Version("0.45.0")


def test_get_min_bitsandbytes_version_invalid_override(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MONGARS_MIN_BITSANDBYTES_VERSION", "not-a-version")

    with caplog.at_level("WARNING"):
        resolved = model_module._get_min_bitsandbytes_version()

    assert resolved == model_module._DEFAULT_MIN_BITSANDBYTES_VERSION
    assert any(
        "Invalid bitsandbytes minimum version override" in message
        for message in caplog.messages
    )
