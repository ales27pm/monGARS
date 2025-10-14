from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from monGARS.mlops.pipelines import unsloth as unsloth_mod


class FakeTrainer:
    def __init__(self, model: Any) -> None:
        self.model = model


@pytest.fixture()
def patch_unsloth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"

    monkeypatch.setattr(unsloth_mod, "configure_cuda_allocator", lambda: None)
    monkeypatch.setattr(unsloth_mod, "ensure_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(unsloth_mod, "describe_environment", lambda: None)
    monkeypatch.setattr(
        unsloth_mod,
        "ensure_directory",
        lambda path: Path(path).mkdir(parents=True, exist_ok=True),
    )

    class Model:
        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    monkeypatch.setattr(
        unsloth_mod,
        "load_4bit_causal_lm",
        lambda *a, **k: (Model(), Tokenizer()),
    )
    monkeypatch.setattr(unsloth_mod, "summarise_device_map", lambda *a, **k: None)
    monkeypatch.setattr(
        unsloth_mod, "prepare_lora_model_light", lambda model, *a, **k: model
    )
    monkeypatch.setattr(unsloth_mod, "_activate_unsloth", lambda model: (model, True))
    monkeypatch.setattr(unsloth_mod, "_load_dataset", lambda **k: object())
    monkeypatch.setattr(
        unsloth_mod,
        "train_qlora",
        lambda model, dataset, config: FakeTrainer(model),
    )
    monkeypatch.setattr(
        unsloth_mod,
        "save_lora_artifacts",
        lambda model, tokenizer, output_dir: Path(output_dir).mkdir(
            parents=True, exist_ok=True
        ),
    )
    monkeypatch.setattr(
        unsloth_mod, "disable_training_mode", lambda model: model.eval()
    )
    monkeypatch.setattr(unsloth_mod, "merge_lora_adapters", lambda *a, **k: None)
    monkeypatch.setattr(
        unsloth_mod, "run_embedding_smoke_test", lambda encoder, texts: (len(texts), 3)
    )

    class FakeLLM2Vec:
        def __init__(self, model: Any, tokenizer: Any, pooling_mode: str) -> None:
            self.model = model
            self.pooling_mode = pooling_mode

        def encode(self, texts: list[str]) -> list[list[int]]:
            return [[1, 2, 3] for _ in texts]

    monkeypatch.setattr(unsloth_mod, "LLM2Vec", FakeLLM2Vec)

    def fake_write_wrapper_bundle(config, output_root):
        wrapper_dir = Path(output_root) / "wrapper"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        module_path = wrapper_dir / "project_wrapper.py"
        module_path.write_text("pass")
        config_path = wrapper_dir / "config.json"
        config_path.write_text("{}")
        return {
            "module": module_path,
            "config": config_path,
            "readme": wrapper_dir / "README.md",
        }

    monkeypatch.setattr(unsloth_mod, "write_wrapper_bundle", fake_write_wrapper_bundle)

    return output_root


def test_run_unsloth_finetune_generates_wrapper(patch_unsloth: Path) -> None:
    results = unsloth_mod.run_unsloth_finetune(
        model_id="hf/test",
        output_dir=patch_unsloth,
        dataset_id="hf/dataset",
        run_smoke_tests=True,
    )

    assert results["chat_lora_dir"].exists()
    assert (patch_unsloth / "run_metadata.json").exists()
    assert results["wrapper_module"].name == "project_wrapper.py"
