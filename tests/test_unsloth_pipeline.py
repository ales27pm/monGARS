from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from monGARS.mlops.pipelines import unsloth as unsloth_mod


class FakeTrainer:
    def __init__(self, model: Any, extra_args: dict[str, Any] | None = None) -> None:
        self.model = model
        self.extra_args = extra_args or {}
        self.train_calls = 0
        self.evaluate_inputs: list[Any] = []

    def train(self) -> None:
        self.train_calls += 1

    def evaluate(self, dataset: Any) -> dict[str, float]:
        self.evaluate_inputs.append(dataset)
        return {"eval_loss": 0.123, "eval_runtime": 1.5}


def _initialise_model_with_quant_state(model_cls: type, state: SimpleNamespace) -> Any:
    model = model_cls()
    setattr(model, "_mongars_quantized_4bit", state.value)
    return model


@pytest.fixture()
def patch_unsloth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
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

    quantized_state = SimpleNamespace(value=True)

    monkeypatch.setattr(
        unsloth_mod,
        "load_4bit_causal_lm",
        lambda *a, **k: (
            _initialise_model_with_quant_state(Model, quantized_state),
            Tokenizer(),
        ),
    )
    monkeypatch.setattr(unsloth_mod, "summarise_device_map", lambda *a, **k: None)
    monkeypatch.setattr(
        unsloth_mod, "prepare_lora_model_light", lambda model, *a, **k: model
    )
    monkeypatch.setattr(unsloth_mod, "_activate_unsloth", lambda model: (model, True))

    class DummyDataset:
        def __init__(self, name: str, size: int) -> None:
            self.name = name
            self._size = size

        def __len__(self) -> int:
            return self._size

    dataset_calls: list[tuple[str, int]] = []

    def fake_load_dataset(**kwargs: Any) -> DummyDataset:
        dataset_path = kwargs.get("dataset_path")
        dataset_id = kwargs.get("dataset_id")
        identifier = (
            Path(dataset_path).name if dataset_path else str(dataset_id or "train")
        )
        if "val" in identifier:
            size = 16
        else:
            size = 128
        dataset_calls.append((identifier, size))
        return DummyDataset(identifier, size)

    monkeypatch.setattr(unsloth_mod, "_load_dataset", fake_load_dataset)

    trainer_instances: list[FakeTrainer] = []

    def fake_train_qlora(
        model: Any,
        dataset: Any,
        *,
        config: Any,
        extra_args: dict[str, Any] | None = None,
        trainer_cls: Any | None = None,
    ) -> FakeTrainer:
        trainer = FakeTrainer(model, extra_args)
        trainer_instances.append(trainer)
        trainer.train()
        return trainer

    monkeypatch.setattr(unsloth_mod, "train_qlora", fake_train_qlora)
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
    merge_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _record_merge(*args: Any, **kwargs: Any) -> bool:
        merge_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(unsloth_mod, "merge_lora_adapters", _record_merge)
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

    return SimpleNamespace(
        output_root=output_root,
        dataset_calls=dataset_calls,
        trainers=trainer_instances,
        quantized_state=quantized_state,
        merge_calls=merge_calls,
    )


def test_run_unsloth_finetune_generates_wrapper(patch_unsloth: SimpleNamespace) -> None:
    results = unsloth_mod.run_unsloth_finetune(
        model_id="hf/test",
        output_dir=patch_unsloth.output_root,
        dataset_id="hf/dataset",
        run_smoke_tests=True,
    )

    assert results["chat_lora_dir"].exists()
    assert (patch_unsloth.output_root / "run_metadata.json").exists()
    assert results["wrapper_module"].name == "project_wrapper.py"
    assert results["merged_dir"] is not None
    assert patch_unsloth.trainers[0].train_calls == 1
    assert len(patch_unsloth.merge_calls) == 1
    assert results["quantized_4bit"] is True


def test_run_unsloth_finetune_records_eval_metrics(
    patch_unsloth: SimpleNamespace,
) -> None:
    results = unsloth_mod.run_unsloth_finetune(
        model_id="hf/test",
        output_dir=patch_unsloth.output_root,
        dataset_path=Path("monGARS_llm_train.jsonl"),
        eval_dataset_path=Path("monGARS_llm_val.jsonl"),
        eval_batch_size=2,
        run_smoke_tests=False,
        merge_to_fp16=False,
    )

    assert patch_unsloth.trainers[0].extra_args["per_device_eval_batch_size"] == 2
    assert patch_unsloth.trainers[0].evaluate_inputs
    metadata = json.loads((patch_unsloth.output_root / "run_metadata.json").read_text())
    assert metadata["dataset_size"] == 128
    assert metadata["eval_dataset_size"] == 16
    assert metadata["evaluation_metrics"]["eval_loss"] == pytest.approx(0.123)
    assert results["evaluation_metrics"]["eval_runtime"] == pytest.approx(1.5)
    assert metadata["quantized_4bit"] is True
    assert results["quantized_4bit"] is True


def test_run_unsloth_finetune_skips_merge_when_quantization_disabled(
    patch_unsloth: SimpleNamespace,
) -> None:
    patch_unsloth.quantized_state.value = False

    results = unsloth_mod.run_unsloth_finetune(
        model_id="hf/test",
        output_dir=patch_unsloth.output_root,
        dataset_id="hf/dataset",
        run_smoke_tests=False,
    )

    assert patch_unsloth.merge_calls == []
    assert results["merged_dir"] is None
    assert results["quantized_4bit"] is False
    metadata = json.loads((patch_unsloth.output_root / "run_metadata.json").read_text())
    assert metadata["quantized_4bit"] is False
