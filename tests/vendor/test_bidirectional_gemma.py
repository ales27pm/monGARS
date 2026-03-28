import importlib
import importlib.util
import sys
import types
from pathlib import Path


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

    module_name = "vendor.llm2vec_monGARS.llm2vec.models.bidirectional_gemma"
    sys.modules.pop(module_name, None)

    gemma_module = importlib.import_module("transformers.models.gemma.modeling_gemma")
    if hasattr(gemma_module, "GemmaFlashAttention2"):
        delattr(gemma_module, "GemmaFlashAttention2")

    spec = importlib.util.spec_from_file_location(
        module_name,
        repo_root / "vendor/llm2vec_monGARS/llm2vec/models/bidirectional_gemma.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - fatal config
        raise RuntimeError("Unable to load bidirectional_gemma module for testing")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_module()
ModifiedGemmaDecoderLayer = _module.ModifiedGemmaDecoderLayer
GEMMA_ATTENTION_CLASSES = _module.GEMMA_ATTENTION_CLASSES


def _minimal_config(attn_impl: str = "eager"):
    from transformers import GemmaConfig

    config = GemmaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    config._attn_implementation = attn_impl
    return config


def test_modified_decoder_layer_uses_flash_fallback_when_symbol_is_missing():
    config = _minimal_config(attn_impl="flash_attention_2")
    layer = ModifiedGemmaDecoderLayer(config, layer_idx=0)

    attention_cls = GEMMA_ATTENTION_CLASSES[config._attn_implementation]
    assert isinstance(layer.self_attn, attention_cls)
    assert getattr(layer.self_attn, "is_causal", True) is False
