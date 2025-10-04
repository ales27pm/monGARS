import argparse
import sys

import pytest

from scripts import provision_models


@pytest.mark.asyncio
async def test_prepare_reasoning_assets_curates_and_warms(monkeypatch):
    class DummyEngine:
        last_call = None

        def __init__(self) -> None:
            pass

        def curate_reasoning_dataset(self, num_samples: int, internal_ratio: float):
            type(self).last_call = (num_samples, internal_ratio)
            return [1, 2], [3]

    warmed: dict[str, int | str] = {}

    class DummySlotManager:
        def __init__(self, *, slot_name: str, model_id: str, max_seq_length: int) -> None:
            warmed["slot_name"] = slot_name
            warmed["model_id"] = model_id
            warmed["max_seq_length"] = max_seq_length

        def __enter__(self):
            return object(), object()

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - context manager protocol
            return None

    monkeypatch.setitem(
        sys.modules,
        "monGARS.core.self_training",
        type("mod", (), {"SelfTrainingEngine": DummyEngine}),
    )
    monkeypatch.setitem(
        sys.modules,
        "monGARS.core.model_slot_manager",
        type("slot_mod", (), {"ModelSlotManager": DummySlotManager}),
    )

    args = argparse.Namespace(
        reasoning_samples=5,
        reasoning_internal_ratio=0.75,
        reasoning_slot="slot-alpha",
        reasoning_model_id="model-beta",
        reasoning_max_seq=1024,
    )

    summary = await provision_models._prepare_reasoning_assets(args)

    assert summary["dataset"]["status"] == "ok"
    assert summary["dataset"]["train_samples"] == 2
    assert summary["dataset"]["eval_samples"] == 1
    assert DummyEngine.last_call == (5, 0.75)
    assert summary["slot"]["status"] == "ok"
    assert warmed == {
        "slot_name": "slot-alpha",
        "model_id": "model-beta",
        "max_seq_length": 1024,
    }


def test_emit_reasoning_summary_reports_outcomes(capsys):
    provision_models._emit_reasoning_summary(
        {
            "dataset": {"status": "ok", "train_samples": 3, "eval_samples": 1},
            "slot": {"status": "failed", "error": "hardware not available"},
        }
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert "Reasoning dataset curated" in out[0]
    assert "Reasoning slot preparation failed" in out[1]
