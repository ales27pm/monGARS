import importlib.util
import sys
import types
from pathlib import Path
from typing import cast

import pytest
from torch import nn


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    package_roots = [
        ("vendor", repo_root / "vendor"),
        ("vendor.llm2vec_monGARS", repo_root / "vendor/llm2vec_monGARS"),
        (
            "vendor.llm2vec_monGARS.llm2vec",
            repo_root / "vendor/llm2vec_monGARS/llm2vec",
        ),
        (
            "vendor.llm2vec_monGARS.llm2vec.models",
            repo_root / "vendor/llm2vec_monGARS/llm2vec/models",
        ),
    ]

    for name, path in package_roots:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = module

    module_name = "vendor.llm2vec_monGARS.llm2vec.models.bidirectional_qwen2"
    if module_name in sys.modules:
        return sys.modules[module_name]

    qwen_module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    if not hasattr(qwen_module, "Qwen2FlashAttention2"):

        class _StubQwen2FlashAttention2(qwen_module.Qwen2Attention):
            """Backfill FlashAttention2 for lean transformer builds."""

        qwen_module.Qwen2FlashAttention2 = _StubQwen2FlashAttention2

    if not hasattr(qwen_module, "Qwen2SdpaAttention"):

        class _StubQwen2SdpaAttention(qwen_module.Qwen2Attention):
            """Backfill SDPA attention for lean transformer builds."""

        qwen_module.Qwen2SdpaAttention = _StubQwen2SdpaAttention

    spec = importlib.util.spec_from_file_location(
        module_name,
        repo_root / "vendor/llm2vec_monGARS/llm2vec/models/bidirectional_qwen2.py",
    )
    if (
        spec is None or spec.loader is None
    ):  # pragma: no cover - fatal configuration error
        raise RuntimeError("Unable to load bidirectional_qwen2 module for testing")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_module()
ModifiedQwen2DecoderLayer = _module.ModifiedQwen2DecoderLayer
Qwen2BiForMNTP = _module.Qwen2BiForMNTP
QWEN2_ATTENTION_CLASSES = _module.QWEN2_ATTENTION_CLASSES


class _DummyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saved_paths: list[str] = []

    def save_pretrained(
        self, save_directory: str
    ) -> None:  # pragma: no cover - trivial
        self.saved_paths.append(save_directory)


def _minimal_config(attn_impl: str = "eager"):
    from transformers import Qwen2Config

    config = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["full_attention"],
        pad_token_id=0,
    )
    config._attn_implementation = attn_impl
    return config


def test_modified_decoder_layer_uses_bidirectional_attention():
    config = _minimal_config()
    layer = ModifiedQwen2DecoderLayer(config, layer_idx=0)

    attention_cls = QWEN2_ATTENTION_CLASSES[config._attn_implementation]
    assert isinstance(layer.self_attn, attention_cls)
    assert getattr(layer.self_attn, "is_causal", True) is False


def test_modified_decoder_layer_rejects_unknown_attention_impl():
    config = _minimal_config()
    config._attn_implementation = "unknown"

    with pytest.raises(ValueError):
        ModifiedQwen2DecoderLayer(config, layer_idx=0)


def test_peft_model_roundtrip(tmp_path: Path):
    config = _minimal_config()
    model = Qwen2BiForMNTP(config)

    dummy_adapter = _DummyAdapter()
    model.set_model_for_peft(dummy_adapter)

    retrieved = model.get_model_for_peft()
    assert retrieved is dummy_adapter

    target_dir = tmp_path / "adapter"
    model.save_peft_model(target_dir)
    assert dummy_adapter.saved_paths == [str(target_dir)]


def test_set_model_for_peft_validates_module_type():
    config = _minimal_config()
    model = Qwen2BiForMNTP(config)

    with pytest.raises(TypeError):
        model.set_model_for_peft(cast(nn.Module, object()))
