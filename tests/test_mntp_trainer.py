from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from models.datasets.catalog import DatasetCatalog
from modules.neurons.training.mntp_trainer import MNTPTrainer


class _CounterStub:
    def __init__(self) -> None:
        self.calls: list[tuple[int | float, dict[str, Any]]] = []

    def add(self, value: int | float, attributes: dict[str, Any]) -> None:
        self.calls.append((value, dict(attributes)))


class _FakeCuda:
    def __init__(self) -> None:
        self._summary_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "NVIDIA GeForce RTX 2070"

    def memory_summary(self) -> str:
        self._summary_calls += 1
        return "Allocated: 2048 MB"

    @staticmethod
    def max_memory_allocated() -> int:
        return 2 * 1024**3


class _TorchStub:
    def __init__(self) -> None:
        self.float32 = "float32"
        self.float16 = "float16"
        self.cuda = _FakeCuda()


class _DummyParam:
    def __init__(self) -> None:
        self.requires_grad = True


class _DummyModel:
    def __init__(self) -> None:
        self._params = {
            "model.layers.0.weight": _DummyParam(),
            "model.layers.1.weight": _DummyParam(),
            "model.layers.2.weight": _DummyParam(),
            "model.layers.3.weight": _DummyParam(),
            "model.layers.4.weight": _DummyParam(),
            "model.layers.5.weight": _DummyParam(),
        }

    def named_parameters(self):  # pragma: no cover - simple iterator
        return list(self._params.items())

    def save_pretrained(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"stub")


class _DummyTokenizer:
    pad_token_id = 1


class _DummySFTConfig:
    def __init__(
        self,
        *,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: int,
        optim: str,
        fp16: bool,
        bf16: bool,
        seed: int,
        output_dir: str,
        gradient_checkpointing: bool,
        max_seq_length: int,
        **_: Any,
    ) -> None:
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.optim = optim
        self.fp16 = fp16
        self.bf16 = bf16
        self.seed = seed
        self.output_dir = output_dir
        self.gradient_checkpointing = gradient_checkpointing
        self.max_seq_length = max_seq_length


class _DummySFTTrainer:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        args: Any,
        train_dataset: Any,
        data_collator: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.state = SimpleNamespace(log_history=[{"loss": 0.25}])

    def train(self) -> SimpleNamespace:
        return SimpleNamespace(metrics={"train_loss": 0.05, "train_runtime": 0.3})


class _DummySlotManager:
    last_model: _DummyModel | None = None

    def __init__(
        self, slot_name: str, *, model_id: str | None = None, max_seq_length: int = 0
    ):
        self.slot_name = slot_name
        self.model_id = model_id
        self.max_seq_length = max_seq_length

    def __enter__(self) -> tuple[_DummyModel, _DummyTokenizer]:
        model = _DummyModel()
        _DummySlotManager.last_model = model
        return model, _DummyTokenizer()

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeDataset:
    def __init__(self) -> None:
        self._items = [
            {"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1], "labels": 2},
            {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": 4},
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


@pytest.fixture()
def trainer_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name_or_path": "stub-model",
                "dataset_name": "stub-dataset",
                "max_seq_length": 64,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    trainer = MNTPTrainer(str(config_path), str(output_dir))
    trainer.config = {
        "model_name_or_path": "stub-model",
        "dataset_name": "stub-dataset",
        "max_seq_length": 64,
    }

    torch_stub = _TorchStub()
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.torch", torch_stub, raising=False
    )
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.ModelSlotManager",
        _DummySlotManager,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.SFTConfig",
        _DummySFTConfig,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.SFTTrainer",
        _DummySFTTrainer,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.default_data_collator",
        lambda features: features,
        raising=False,
    )

    cycle_counter = _CounterStub()
    failure_counter = _CounterStub()
    token_counter = _CounterStub()
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.TRAINING_CYCLE_COUNTER",
        cycle_counter,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.TRAINING_FAILURE_COUNTER",
        failure_counter,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.neurons.training.mntp_trainer.TRAINING_TOKEN_COUNTER",
        token_counter,
        raising=False,
    )

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_file = dataset_root / "curated_batch.jsonl"
    dataset_file.write_text("{}\n", encoding="utf-8")
    catalog = DatasetCatalog(dataset_root)
    catalog.register(
        run_id="test-run",
        dataset_dir=dataset_root,
        dataset_file=dataset_file,
        record_count=1,
    )

    return {
        "trainer": trainer,
        "torch_stub": torch_stub,
        "cycle_counter": cycle_counter,
        "failure_counter": failure_counter,
        "token_counter": token_counter,
        "output_dir": output_dir,
    }


def test_fit_saves_adapter_and_emits_metrics(trainer_setup: dict[str, Any]):
    trainer = trainer_setup["trainer"]
    dataset = _FakeDataset()

    summary = trainer.fit(dataset, epochs=2)

    adapter_dir = Path(summary["artifacts"]["adapter"])
    assert adapter_dir.exists()
    weights_path = Path(summary["artifacts"]["weights"])
    assert weights_path.exists()

    max_cuda = summary["metrics"].get("max_cuda_memory_gb")
    assert max_cuda is not None and max_cuda < 5

    cycle_calls = trainer_setup["cycle_counter"].calls
    assert len(cycle_calls) == 2
    assert cycle_calls[0][1]["cycle"] == "start"
    assert cycle_calls[1][1]["cycle"] == "complete"

    token_calls = trainer_setup["token_counter"].calls
    assert token_calls and token_calls[0][0] == 5

    assert trainer_setup["failure_counter"].calls == []

    model = _DummySlotManager.last_model
    assert model is not None
    frozen_flags = {
        name: param.requires_grad for name, param in model.named_parameters()
    }
    assert frozen_flags["model.layers.0.weight"] is False
    assert frozen_flags["model.layers.5.weight"] is True

    assert summary["metrics"]["training_examples"] == len(dataset)
    assert summary["metrics"]["estimated_tokens"] == 5
    assert summary["metrics"]["per_device_train_batch_size"] == 1
    assert summary["metrics"]["gradient_accumulation_steps"] == 16
    assert summary["metrics"]["max_seq_length"] == trainer.config["max_seq_length"]
