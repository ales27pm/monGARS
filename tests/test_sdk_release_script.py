from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import sdk_release


def test_build_python_sdk_invokes_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    (repo_root / "sdks" / "python").mkdir(parents=True)
    output_dir = tmp_path / "artifacts"

    monkeypatch.setattr(sdk_release.importlib.util, "find_spec", lambda name: object())
    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_run(command: list[str], cwd: Path, check: bool) -> None:
        calls.append((tuple(command), cwd))

    monkeypatch.setattr(sdk_release.subprocess, "run", fake_run)

    artefact_dir = sdk_release.build_python_sdk(repo_root, output_dir=output_dir)

    assert artefact_dir == output_dir
    assert output_dir.exists()
    assert calls
    command, cwd = calls[0]
    assert command[0] == sys.executable
    assert command[1:4] == ("-m", "build", "--wheel")
    assert command[-2:] == ("--outdir", str(output_dir))
    assert cwd == repo_root / "sdks" / "python"


def test_build_python_sdk_requires_build_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    (repo_root / "sdks" / "python").mkdir(parents=True)

    monkeypatch.setattr(sdk_release.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(sdk_release.BuildError):
        sdk_release.build_python_sdk(repo_root)


def test_build_typescript_sdk_runs_expected_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    sdk_dir = repo_root / "sdks" / "typescript"
    sdk_dir.mkdir(parents=True)

    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_run(command: list[str], cwd: Path, check: bool) -> None:
        calls.append((tuple(command), cwd))

    monkeypatch.setattr(sdk_release.subprocess, "run", fake_run)

    artefact_dir = sdk_release.build_typescript_sdk(repo_root)

    assert artefact_dir == sdk_dir / "dist"
    assert artefact_dir.exists()
    assert [c[0] for c in calls] == [
        ("npm", "ci"),
        ("npm", "run", "build"),
        ("npm", "pack", "--pack-destination", str(artefact_dir)),
    ]
    assert all(cwd == sdk_dir for _, cwd in calls)
