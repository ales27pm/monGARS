"""Tests for `monGARS.mlops.model` helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from packaging.version import Version
from pytest import FixtureRequest

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
def _force_quantization_available(
    monkeypatch: pytest.MonkeyPatch, request: FixtureRequest
) -> None:
    if request.node.get_closest_marker("no_quantization_patch"):
        return
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: True)


@pytest.fixture(autouse=True)
def _patch_auto_config(monkeypatch: pytest.MonkeyPatch):
    state = SimpleNamespace(dtype=None, alt_dtype=None, error=None)

    class _FakeConfig:
        def __init__(self) -> None:
            self.torch_dtype = state.dtype
            self.dtype = state.alt_dtype

    def _from_pretrained(*_, **__):
        if state.error is not None:
            raise state.error
        return _FakeConfig()

    monkeypatch.setattr(
        model_module,
        "AutoConfig",
        SimpleNamespace(from_pretrained=_from_pretrained),
    )
    return state


@pytest.fixture()
def auto_config_state(_patch_auto_config) -> SimpleNamespace:  # type: ignore[misc]
    return _patch_auto_config


def _dtype_from_kwargs(kwargs: dict[str, Any]) -> Any:
    """Return the dtype argument recorded in a kwargs dictionary."""

    return kwargs.get("dtype") or kwargs.get("torch_dtype")


@pytest.mark.no_quantization_patch
def test_is_quantization_unavailable_due_to_cuda(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)

    with caplog.at_level("WARNING"):
        assert model_module._is_quantization_available() is False

    assert any(
        "CUDA unavailable; using CPU execution without 4-bit quantization" in message
        for message in caplog.messages
    )


def test_load_4bit_causal_lm_prefers_dtype_kwarg(
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
    assert (
        recorded_kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload is True
    )
    assert recorded_kwargs["max_memory"][0] == "5308MiB"
    assert getattr(model, "_mongars_quantized_4bit", None) is True


def test_load_4bit_causal_lm_falls_back_to_torch_dtype_kwarg(
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
        if "dtype" in kwargs:
            raise TypeError("unexpected keyword argument 'dtype'")
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _legacy_from_pretrained
    )

    model = model_module.load_4bit_causal_lm(
        "meta-llama/Llama-2-7b-hf", offload_dir=tmp_path
    )[0]

    assert len(calls) == 2
    assert calls[0]["dtype"] is model_module.torch.float16
    assert calls[1]["torch_dtype"] is model_module.torch.float16
    assert getattr(model, "_mongars_quantized_4bit", None) is True


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
        dtype,
    ) -> Any:  # noqa: ARG001
        recorded.update(
            {
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "dtype": dtype,
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

    def _fake_from_pretrained(*_, quantization_config: Any, **__) -> Any:
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

    def _fake_from_pretrained(*_, **kwargs) -> Any:
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
    assert recorded_kwargs.get("dtype") is model_module.torch.float16
    assert getattr(model, "_mongars_quantized_4bit", None) is False


def test_load_4bit_causal_lm_cpu_fallback_respects_requested_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: False)

    recorded_kwargs: dict[str, Any] = {}

    def _fake_from_pretrained(*_, **kwargs) -> Any:  # noqa: ANN002
        recorded_kwargs.update(kwargs)
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained
    )

    preferred_dtype = getattr(
        model_module.torch, "bfloat16", model_module.torch.float16
    )

    model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf", dtype=preferred_dtype)

    assert recorded_kwargs.get("dtype") is preferred_dtype


def test_load_4bit_causal_lm_cpu_fallback_prefers_config_dtype(
    monkeypatch: pytest.MonkeyPatch, auto_config_state: SimpleNamespace
) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: False)

    bfloat16 = getattr(model_module.torch, "bfloat16", None)
    if bfloat16 is None:
        pytest.skip("torch build does not expose bfloat16")

    auto_config_state.dtype = "bfloat16"

    calls: list[dict[str, Any]] = []

    def _maybe_fail_from_pretrained(*_, **kwargs):  # noqa: ANN002
        calls.append(kwargs)
        dtype = _dtype_from_kwargs(kwargs)
        if dtype is bfloat16:
            raise RuntimeError("Simulated failure for config dtype")
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        _maybe_fail_from_pretrained,
    )

    model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf")

    assert _dtype_from_kwargs(calls[0]) is bfloat16
    assert _dtype_from_kwargs(calls[1]) is model_module.torch.float16


def test_load_4bit_causal_lm_cpu_fallback_ignores_unknown_config_dtype(
    monkeypatch: pytest.MonkeyPatch, auto_config_state: SimpleNamespace
) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: False)
    auto_config_state.dtype = "fp8"

    calls: list[dict[str, Any]] = []

    def _capture_kwargs(*_, **kwargs):  # noqa: ANN002
        calls.append(kwargs)
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _capture_kwargs
    )

    model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf")

    assert _dtype_from_kwargs(calls[0]) is model_module.torch.float16


def test_load_4bit_causal_lm_cpu_fallback_tries_multiple_dtypes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: False)

    calls: list[dict[str, Any]] = []

    def _maybe_fail_from_pretrained(*_, **kwargs):  # noqa: ANN002
        calls.append(kwargs)
        dtype = _dtype_from_kwargs(kwargs)
        if dtype is not model_module.torch.float32:
            raise RuntimeError("Unsupported dtype for CPU fallback")
        return _DummyModel()

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        _maybe_fail_from_pretrained,
    )

    with caplog.at_level("WARNING"):
        model = model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf")[0]

    assert _dtype_from_kwargs(calls[-1]) is model_module.torch.float32
    assert any(
        "CPU fallback load failed for dtype" in record.message
        for record in caplog.records
    )
    assert getattr(model, "_mongars_quantized_4bit", None) is False


def test_load_4bit_causal_lm_cpu_fallback_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(model_module, "_is_quantization_available", lambda: False)

    def _failing_from_pretrained(*_, **__):
        raise RuntimeError("Simulated model loading failure")

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", _failing_from_pretrained
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="Simulated model loading failure"):
            model_module.load_4bit_causal_lm("meta-llama/Llama-2-7b-hf")

    assert any(
        "Failed to load model on CPU fallback" in record.message
        for record in caplog.records
    )


def test_resolve_config_dtype_handles_exceptions(
    caplog: pytest.LogCaptureFixture, auto_config_state: SimpleNamespace
) -> None:
    auto_config_state.error = RuntimeError("boom")

    with caplog.at_level("DEBUG"):
        dtype = model_module._resolve_config_dtype("hf/test", trust_remote_code=True)

    assert dtype is None
    assert any(
        "Unable to resolve model config dtype" in record.message
        for record in caplog.records
    )


def test_build_model_kwargs_candidates_skips_unsupported() -> None:
    def _loader(model_id: str, *, low_cpu_mem_usage: bool) -> Any:  # noqa: ARG001
        return _DummyModel()

    candidates = model_module._build_model_kwargs_candidates(
        _loader,
        base_kwargs={"low_cpu_mem_usage": True},
        optional_kwargs=(
            {"dtype": model_module.torch.float32},
            {"torch_dtype": model_module.torch.float32},
        ),
    )

    assert candidates == [{"low_cpu_mem_usage": True}]


def test_build_model_kwargs_candidates_includes_all_for_kwargs_loader() -> None:
    def _loader(model_id: str, **kwargs: Any) -> Any:  # noqa: ARG001
        return _DummyModel()

    candidates = model_module._build_model_kwargs_candidates(
        _loader,
        base_kwargs={"low_cpu_mem_usage": True},
        optional_kwargs=(
            {"dtype": model_module.torch.float32},
            {"torch_dtype": model_module.torch.float32},
        ),
    )

    assert candidates == [
        {"low_cpu_mem_usage": True, "dtype": model_module.torch.float32},
        {"low_cpu_mem_usage": True, "torch_dtype": model_module.torch.float32},
        {"low_cpu_mem_usage": True},
    ]


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
            {"dtype": model_module.torch.float16},
            {"torch_dtype": model_module.torch.float16},
        ),
    )

    assert len(candidates) == 3
    assert candidates[0]["dtype"] is model_module.torch.float16
    assert "torch_dtype" not in candidates[0]
    assert candidates[1]["torch_dtype"] is model_module.torch.float16
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
