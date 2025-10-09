"""Unit tests for the Dolphin Unsloth automation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_dolphin_unsloth_workflow as workflow


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _base_config(
    tmp_path: Path, *, minimum_train_records: int = 1
) -> workflow.WorkflowConfig:
    repo_dataset = tmp_path / "repo.jsonl"
    formatted_dataset = tmp_path / "formatted.jsonl"

    _write_jsonl(
        repo_dataset,
        [
            {"instruction": "Hello", "output": "World"},
            {"instruction": "Duplicate", "output": "Value"},
        ],
    )
    _write_jsonl(
        formatted_dataset,
        [
            {"instruction": "Duplicate", "output": "Value"},
            {"instruction": "Another", "output": "Example"},
        ],
    )

    return workflow.WorkflowConfig(
        refresh_analysis=False,
        skip_analysis=True,
        analyzer_script=tmp_path / "does_not_matter.py",
        analyzer_output=repo_dataset,
        formatted_dataset=formatted_dataset,
        dataset_output_dir=tmp_path / "dataset_out",
        validation_ratio=0.25,
        shuffle_seed=123,
        training_output_dir=tmp_path / "train_out",
        max_seq_length=2048,
        learning_rate=1e-4,
        num_train_epochs=1.0,
        gradient_accumulation_steps=1,
        hf_token=None,
        hf_token_source=None,
        allow_cpu_fallback=False,
        max_retries=3,
        minimum_train_records=minimum_train_records,
        dry_run=True,
    )


def test_parse_arguments_env_token(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    args = [
        "--skip-analysis",
        "--analyzer-output",
        str(tmp_path / "repo.jsonl"),
        "--formatted-dataset",
        str(tmp_path / "formatted.jsonl"),
        "--dataset-output-dir",
        str(tmp_path / "dataset"),
        "--training-output-dir",
        str(tmp_path / "train"),
        "--minimum-train-records",
        "1",
    ]

    config = workflow.parse_arguments(args)

    assert config.hf_token == "secret-token"
    assert config.hf_token_source == "env:HF_TOKEN"


def test_parse_arguments_conflicting_flags(tmp_path):
    args = [
        "--skip-analysis",
        "--refresh-analysis",
        "--analyzer-output",
        str(tmp_path / "repo.jsonl"),
        "--formatted-dataset",
        str(tmp_path / "formatted.jsonl"),
        "--dataset-output-dir",
        str(tmp_path / "dataset"),
        "--training-output-dir",
        str(tmp_path / "train"),
    ]

    with pytest.raises(SystemExit):
        workflow.parse_arguments(args)


def test_build_datasets_deduplicates_and_respects_minimum(tmp_path):
    config = _base_config(tmp_path)

    train_path, validation_path = workflow.build_datasets(config)

    train_records = list(workflow._load_jsonl_records(train_path))
    assert len(train_records) >= config.minimum_train_records

    if validation_path:
        validation_records = list(workflow._load_jsonl_records(validation_path))
        assert all(
            record["instruction"] != "Duplicate" for record in validation_records
        )


def test_build_datasets_raises_when_minimum_not_met(tmp_path):
    config = _base_config(tmp_path, minimum_train_records=5)

    with pytest.raises(RuntimeError):
        workflow.build_datasets(config)
