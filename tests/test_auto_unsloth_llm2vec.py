import json
from pathlib import Path

import pytest

from scripts import auto_unsloth_llm2vec as workflow


def test_main_skip_train_uses_default_seed(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    dataset_dir.mkdir(parents=True)
    train_path.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n")
    validation_path.write_text(json.dumps({"prompt": "c", "completion": "d"}) + "\n")

    monkeypatch.setattr(workflow, "scan_llm_usage", lambda *_, **__: ["usage"])
    monkeypatch.setattr(
        workflow, "scan_module_interactions", lambda *_, **__: ["interaction"]
    )

    seed_file = tmp_path / "seed.jsonl"
    seed_file.write_text(json.dumps({"prompt": "seed", "completion": "seed"}) + "\n")
    monkeypatch.setattr(workflow, "DEFAULT_SEED_DATASET", seed_file)

    captured: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        captured["extra_datasets"] = kwargs["extra_datasets"]
        return {"train": train_path, "validation": validation_path}

    monkeypatch.setattr(workflow, "build_unsloth_llm2vec_dataset", fake_build)

    result = workflow.main(
        [
            "--root",
            str(tmp_path),
            "--dataset-dir",
            str(dataset_dir),
            "--metadata-path",
            str(tmp_path / "meta.json"),
            "--skip-train",
        ]
    )

    assert result["train"] == train_path
    assert captured["extra_datasets"] == [seed_file]


def test_main_runs_training(monkeypatch, tmp_path: Path) -> None:
    dataset_paths = {"train": tmp_path / "train.jsonl", "validation": None}
    dataset_paths["train"].write_text(
        json.dumps({"prompt": "a", "completion": "b"}) + "\n"
    )

    monkeypatch.setattr(workflow, "_build_dataset", lambda *_: dataset_paths)

    captured: dict[str, object] = {}

    def fake_run(args, paths):
        captured["args"] = args
        captured["paths"] = paths
        return {"chat_lora_dir": Path("chat"), "wrapper_dir": Path("wrapper")}

    monkeypatch.setattr(workflow, "_run_training", fake_run)

    result = workflow.main(
        ["--root", str(tmp_path), "--dataset-dir", str(tmp_path / "ds")]
    )

    assert result["chat_lora_dir"] == Path("chat")
    assert captured["paths"] == dataset_paths


def test_run_training_invokes_unsloth(monkeypatch, tmp_path: Path) -> None:
    args = workflow.parse_args(
        ["--root", str(tmp_path), "--output-dir", str(tmp_path / "out")]
    )
    dataset_paths = {"train": tmp_path / "train.jsonl", "validation": None}
    dataset_paths["train"].write_text(
        json.dumps({"prompt": "z", "completion": "y"}) + "\n"
    )

    captured: dict[str, object] = {}

    def fake_run_unsloth_finetune(**kwargs):
        captured.update(kwargs)
        return {"chat_lora_dir": Path("chat"), "wrapper_dir": Path("wrapper")}

    monkeypatch.setattr(workflow, "run_unsloth_finetune", fake_run_unsloth_finetune)

    result = workflow._run_training(args, dataset_paths)

    assert captured["dataset_path"] == dataset_paths["train"]
    assert result["wrapper_dir"] == Path("wrapper")


def test_run_training_uses_validation_dataset(monkeypatch, tmp_path: Path) -> None:
    args = workflow.parse_args(
        [
            "--root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--eval-batch-size",
            "4",
        ]
    )
    dataset_paths = {
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
    }
    dataset_paths["train"].write_text(
        json.dumps({"prompt": "train", "completion": "resp"}) + "\n"
    )
    dataset_paths["validation"].write_text(
        json.dumps({"prompt": "val", "completion": "resp"}) + "\n"
    )

    captured: dict[str, object] = {}

    def fake_run_unsloth_finetune(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(workflow, "run_unsloth_finetune", fake_run_unsloth_finetune)

    workflow._run_training(args, dataset_paths)

    assert captured["eval_dataset_path"] == dataset_paths["validation"]
    assert captured["eval_batch_size"] == 4


def test_parse_args_rejects_invalid_ratio() -> None:
    with pytest.raises(SystemExit) as excinfo:
        workflow.parse_args(["--validation-ratio", "0"])

    assert "(0, 1)" in str(excinfo.value)


def test_parse_args_rejects_unknown_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        workflow.parse_args(["--not-an-arg"])

    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err.lower()
