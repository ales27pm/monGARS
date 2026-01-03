from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_mongars_llm_pipeline as pipeline


def test_cmd_finetune_updates_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chat_dir = tmp_path / "chat_lora"
    chat_dir.mkdir()
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    wrapper_module = wrapper_dir / "project_wrapper.py"
    wrapper_module.write_text("pass")
    wrapper_config = wrapper_dir / "config.json"
    wrapper_config.write_text("{}")
    merged_dir = tmp_path / "merged_fp16"
    merged_dir.mkdir()

    captured_kwargs: dict[str, object] = {}

    def fake_run_unsloth(**kwargs: object) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {
            "chat_lora_dir": chat_dir,
            "wrapper_module": wrapper_module,
            "wrapper_config": wrapper_config,
            "wrapper_dir": wrapper_dir,
            "merged_dir": merged_dir,
            "dataset_size": 42,
            "eval_dataset_size": 7,
            "evaluation_metrics": {"eval_loss": 0.1},
        }

    monkeypatch.setattr(pipeline, "run_unsloth_finetune", fake_run_unsloth)

    summary_args: dict[str, object] = {}

    def fake_build_adapter_summary(**kwargs: object) -> dict[str, object]:
        summary_args.update(kwargs)
        return {"artifacts": {"adapter": str(chat_dir)}}

    monkeypatch.setattr(pipeline, "build_adapter_summary", fake_build_adapter_summary)

    manifest_path = tmp_path / "registry" / "adapter_manifest.json"
    manifest_path.parent.mkdir()
    manifest_calls: list[tuple[Path, dict[str, object]]] = []

    def fake_update_manifest(path: Path, summary: dict[str, object]) -> SimpleNamespace:
        manifest_calls.append((path, summary))
        return SimpleNamespace(path=manifest_path)

    monkeypatch.setattr(pipeline, "update_manifest", fake_update_manifest)

    dataset_path = tmp_path / "datasets" / "train.jsonl"
    eval_path = tmp_path / "datasets" / "val.jsonl"

    args = Namespace(
        model_id="hf/test",
        dataset_id=None,
        dataset_path=str(dataset_path),
        output_dir=str(tmp_path / "out"),
        max_seq_len=2048,
        vram_budget_mb=6144,
        activation_buffer_mb=768,
        batch_size=2,
        grad_accum=4,
        learning_rate=2.5e-4,
        epochs=1.5,
        max_steps=1000,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        train_fraction=0.75,
        eval_dataset_id="hf/val",
        eval_dataset_path=str(eval_path),
        eval_batch_size=3,
        skip_smoke_tests=False,
        skip_metadata=False,
        skip_merge=False,
        registry_path=str(tmp_path / "registry"),
    )

    pipeline.cmd_finetune(args)

    assert Path(captured_kwargs["dataset_path"]) == dataset_path.resolve()
    assert captured_kwargs["train_fraction"] == 0.75
    assert captured_kwargs["eval_dataset_id"] == "hf/val"
    assert Path(captured_kwargs["eval_dataset_path"]) == eval_path.resolve()
    assert manifest_calls and manifest_calls[0][0] == (tmp_path / "registry").resolve()
    summary = manifest_calls[0][1]
    assert summary["artifacts"]["merged_fp16"] == str(merged_dir)
    assert summary_args["labels"]["pipeline"] == "unsloth_llm2vec"
    assert summary_args["training"]["train_fraction"] == 0.75
    assert summary_args["training"]["eval_batch_size"] == 3
    assert summary_args["metrics"]["evaluation"]["eval_loss"] == 0.1
