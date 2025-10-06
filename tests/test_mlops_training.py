from __future__ import annotations

import pytest
import torch

from monGARS.mlops.training import OOMRetryEvent, TrainerConfig, train_qlora


class DummyTrainer:
    """Test double that mimics ``transformers.Trainer``."""

    failures_remaining = 0
    failure_factory = staticmethod(lambda: torch.cuda.OutOfMemoryError("mock OOM"))
    instances: list["DummyTrainer"] = []

    def __init__(
        self, *, model, args, train_dataset, data_collator
    ):  # noqa: D401 - signature mirrors Trainer
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.train_calls = 0
        DummyTrainer.instances.append(self)

    def train(self) -> None:
        self.train_calls += 1
        if DummyTrainer.failures_remaining > 0:
            DummyTrainer.failures_remaining -= 1
            raise DummyTrainer.failure_factory()


@pytest.fixture(autouse=True)
def _reset_dummy_trainer():
    DummyTrainer.failures_remaining = 0
    DummyTrainer.failure_factory = staticmethod(
        lambda: torch.cuda.OutOfMemoryError("mock OOM")
    )
    DummyTrainer.instances.clear()
    yield
    DummyTrainer.failures_remaining = 0
    DummyTrainer.failure_factory = staticmethod(
        lambda: torch.cuda.OutOfMemoryError("mock OOM")
    )
    DummyTrainer.instances.clear()


@pytest.fixture
def trainer_config(tmp_path):
    return TrainerConfig(
        output_dir=tmp_path,
        batch_size=4,
        grad_accum=2,
        learning_rate=2e-4,
        epochs=1.0,
        max_steps=-1,
    )


def test_train_qlora_success(trainer_config):
    trainer = train_qlora(
        object(),
        dataset=[{"input_ids": [1, 2]}],
        config=trainer_config,
        trainer_cls=DummyTrainer,
    )

    assert isinstance(trainer, DummyTrainer)
    assert trainer.args.per_device_train_batch_size == 4
    assert trainer.args.gradient_accumulation_steps == 2
    assert trainer.train_calls == 1


def test_train_qlora_retries_with_smaller_batch(trainer_config):
    DummyTrainer.failures_remaining = 1
    trainer = train_qlora(
        object(),
        dataset=[{"input_ids": [1, 2]}],
        config=trainer_config,
        trainer_cls=DummyTrainer,
    )

    # Two trainer instances are created: the initial attempt and the retry
    assert len(DummyTrainer.instances) == 2
    assert DummyTrainer.instances[0].args.per_device_train_batch_size == 4
    assert DummyTrainer.instances[1].args.per_device_train_batch_size == 2
    assert trainer.args.per_device_train_batch_size == 2


def test_train_qlora_reduces_gradient_accumulation_when_batch_is_one(trainer_config):
    DummyTrainer.failures_remaining = 1
    trainer_config.batch_size = 1
    trainer_config.grad_accum = 8

    trainer = train_qlora(
        object(),
        dataset=[{"input_ids": [1]}],
        config=trainer_config,
        trainer_cls=DummyTrainer,
    )

    assert len(DummyTrainer.instances) == 2
    assert DummyTrainer.instances[0].args.gradient_accumulation_steps == 8
    assert DummyTrainer.instances[1].args.gradient_accumulation_steps == 4
    assert trainer.args.gradient_accumulation_steps == 4


def test_train_qlora_raises_after_exhausting_retries(trainer_config):
    DummyTrainer.failures_remaining = 2
    captured_events: list[OOMRetryEvent] = []

    with pytest.raises(torch.cuda.OutOfMemoryError):
        train_qlora(
            object(),
            dataset=[{"input_ids": [1]}],
            config=trainer_config,
            extra_args={"oom_retries": 0, "oom_event_hooks": captured_events.append},
            trainer_cls=DummyTrainer,
        )

    assert len(DummyTrainer.instances) == 1
    assert len(captured_events) == 1
    assert captured_events[0].will_retry is False


def test_train_qlora_does_not_retry_on_non_oom(trainer_config):
    DummyTrainer.failures_remaining = 1
    DummyTrainer.failure_factory = staticmethod(lambda: RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        train_qlora(
            object(),
            dataset=[{"input_ids": [1]}],
            config=trainer_config,
            trainer_cls=DummyTrainer,
        )

    assert len(DummyTrainer.instances) == 1


def test_train_qlora_honours_custom_backoff_factor(trainer_config):
    DummyTrainer.failures_remaining = 1
    trainer_config.batch_size = 8

    trainer = train_qlora(
        object(),
        dataset=[{"input_ids": [1, 2]}],
        config=trainer_config,
        extra_args={"oom_backoff_factor": 0.25},
        trainer_cls=DummyTrainer,
    )

    assert len(DummyTrainer.instances) == 2
    assert DummyTrainer.instances[1].args.per_device_train_batch_size == 2
    assert trainer.args.per_device_train_batch_size == 2


def test_train_qlora_emits_oom_events(trainer_config):
    DummyTrainer.failures_remaining = 1
    captured_events: list[OOMRetryEvent] = []

    def _hook(event: OOMRetryEvent) -> None:
        captured_events.append(event)

    train_qlora(
        object(),
        dataset=[{"input_ids": [1, 2]}],
        config=trainer_config,
        extra_args={"oom_event_hooks": [_hook]},
        trainer_cls=DummyTrainer,
    )

    assert len(captured_events) == 1
    event = captured_events[0]
    assert event.will_retry is True
    assert event.next_batch_size < event.batch_size
    assert event.remaining_retries >= 1
