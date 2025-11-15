"""Tests for the fine-tune to deployment automation script."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import finetune_deploy_launch as automation


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    (tmp_path / ".env.example").write_text(
        "SECRET_KEY=dev-secret-change-me\n", encoding="utf8"
    )
    (tmp_path / "docker-compose.yml").write_text(
        json.dumps({"version": "3", "services": {}}), encoding="utf8"
    )
    unsloth_dir = tmp_path / "datasets" / "unsloth"
    unsloth_dir.mkdir(parents=True)
    unsloth_dir.joinpath("monGARS_unsloth_dataset.jsonl").write_text(
        "{}\n", encoding="utf8"
    )
    eval_dir = tmp_path / "datasets" / "monGARS_llm"
    eval_dir.mkdir(parents=True)
    eval_dir.joinpath("monGARS_llm_val.jsonl").write_text("{}\n", encoding="utf8")
    registry_dir = tmp_path / "models" / "encoders" / "monGARS_unsloth"
    registry_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def fake_run(monkeypatch: pytest.MonkeyPatch, project_root: Path) -> list[list[str]]:
    commands: list[list[str]] = []

    def _fake_run_command(
        self: automation.FineTuneDeploymentAutomation, command: list[str]
    ) -> None:
        commands.append(command)

    monkeypatch.setattr(
        automation.FineTuneDeploymentAutomation, "_run_command", _fake_run_command
    )
    monkeypatch.setattr(
        automation.FineTuneDeploymentAutomation,
        "_compose_invocation",
        lambda self: ("docker", "compose"),
    )
    monkeypatch.setattr(
        automation.FineTuneDeploymentAutomation,
        "_wait_for_url",
        lambda self, url, label: None,
    )
    monkeypatch.setattr(
        automation.webbrowser, "open", lambda url, new=2, autoraise=True: True
    )
    monkeypatch.setattr(
        automation.shutil,
        "which",
        lambda command: f"/usr/bin/{command}",
    )
    return commands


@pytest.fixture()
def fake_training(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> dict[str, object]:
    result: dict[str, object] = {}

    def _fake_run_unsloth_finetune(**kwargs):
        output_dir: Path = kwargs["output_dir"]
        chat_dir = output_dir / "chat_lora"
        chat_dir.mkdir(parents=True, exist_ok=True)
        (chat_dir / "adapter.safetensors").write_text("", encoding="utf8")
        wrapper_dir = output_dir / "wrapper"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        (wrapper_dir / "module.py").write_text("", encoding="utf8")
        merged_dir = output_dir / "merged_fp16"
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "weights.safetensors").write_text("", encoding="utf8")
        local_result = {
            "output_dir": output_dir,
            "chat_lora_dir": chat_dir,
            "wrapper_module": wrapper_dir / "module.py",
            "wrapper_config": wrapper_dir / "config.json",
            "wrapper_dir": wrapper_dir,
            "merged_dir": merged_dir,
            "dataset_size": 42,
            "eval_dataset_size": 7,
            "evaluation_metrics": {"loss": 0.1},
            "quantized_4bit": True,
        }
        result.update(local_result)
        return local_result

    monkeypatch.setattr(automation, "run_unsloth_finetune", _fake_run_unsloth_finetune)
    return result


@pytest.fixture()
def fake_manifest(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    payload: dict[str, object] = {}

    def _fake_update_manifest(path: Path, summary: dict[str, object]):
        payload["path"] = Path(path) / "adapter_manifest.json"
        payload["summary"] = summary
        payload["path"].parent.mkdir(parents=True, exist_ok=True)
        payload["path"].write_text("{}", encoding="utf8")

        class _Manifest:
            def __init__(self, manifest_path: Path) -> None:
                self.path = manifest_path
                self.current = SimpleNamespace(version="stub-version")

        return _Manifest(payload["path"])

    monkeypatch.setattr(automation, "update_manifest", _fake_update_manifest)
    return payload


@pytest.fixture()
def fake_snapshot(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> dict[str, object]:
    payload: dict[str, object] = {}

    def _fake_snapshot_download(**kwargs):
        target = kwargs.get("local_dir")
        if target is None:
            target = project_root / "hf-cache"
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "config.json").write_text("{}", encoding="utf8")
        payload["path"] = target_path
        payload["kwargs"] = kwargs
        return str(target_path)

    module = SimpleNamespace(snapshot_download=_fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return payload


def test_automation_runs_full_pipeline(
    project_root: Path,
    fake_run: list[list[str]],
    fake_training: dict[str, object],
    fake_manifest: dict[str, object],
    fake_snapshot: dict[str, object],
) -> None:
    args = automation.build_parser().parse_args(
        [
            "--project-root",
            str(project_root),
            "--webapp-url",
            "http://localhost:9000/ui/",
        ]
    )
    automation._resolve_paths(args)
    runner = automation.FineTuneDeploymentAutomation(args)

    exit_code = runner.run()

    assert exit_code == 0
    assert fake_training["chat_lora_dir"].exists()
    assert fake_manifest["path"].exists()
    summary = fake_manifest["summary"]
    assert summary["metrics"]["dataset_size"] == 42
    assert summary["metrics"]["quantized_4bit"] is True
    assert summary["labels"]["pipeline"] == "unsloth_llm2vec"
    assert summary["labels"]["quantization"] == "4bit"
    assert summary["labels"]["llm2vec_export"] == "enabled"
    assert "base_snapshot" in summary["labels"]
    assert fake_run[0][-1] in {"pull", "build"}
    assert any(command[-2:] == ["up", "-d"] for command in fake_run)
    assert Path(fake_snapshot["path"]).exists()
    assert fake_snapshot["kwargs"]["repo_id"] == automation.DEFAULT_MODEL_ID
