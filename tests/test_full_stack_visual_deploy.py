from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.full_stack_visual_deploy import EnvFileManager, VisualDeployer


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key] = value
    return values


def test_env_manager_creates_and_updates(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text(
        "\n".join(
            [
                "SECRET_KEY=dev-secret-change-me",
                "DJANGO_SECRET_KEY=django-insecure-change-me",
                "DB_PASSWORD=changeme",
                "SEARXNG_SECRET=",
                "SEARCH_SEARX_API_KEY=",
            ]
        )
        + "\n",
        encoding="utf8",
    )
    manager = EnvFileManager(tmp_path, logging.getLogger("test"))
    manager.ensure_env_file()
    manager.ensure_secure_defaults()

    env_file = tmp_path / ".env"
    assert env_file.exists()
    values = _read_env(env_file)

    assert values["SECRET_KEY"] not in {"", "dev-secret-change-me"}
    assert values["DJANGO_SECRET_KEY"] not in {"", "django-insecure-change-me"}
    assert values["DB_PASSWORD"] not in {"", "changeme", "password"}
    assert values["SEARXNG_SECRET"] not in {"", "change-me"}
    assert values["SEARCH_SEARX_API_KEY"] not in {"", "change-me"}
    assert values["SEARCH_SEARX_ENABLED"].lower() == "true"
    assert values["SEARXNG_PORT"] == "8082"
    assert values["SEARXNG_BASE_URL"] == "http://localhost:8082"
    assert values["SEARCH_SEARX_BASE_URL"] == "http://localhost:8082"


def test_compose_invocation_prefers_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deployer = VisualDeployer(
        project_root=tmp_path,
        include_searx=True,
        non_interactive=False,
        step_filter=set(),
    )

    def fake_which(command: str) -> str | None:
        return {
            "docker": "/usr/bin/docker",
            "docker-compose": "/usr/local/bin/docker-compose",
        }.get(command)

    class FakeCompletedProcess:
        returncode = 0

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: FakeCompletedProcess()
    )

    assert deployer._compose_invocation() == ("/usr/bin/docker", "compose")


def test_compose_invocation_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deployer = VisualDeployer(
        project_root=tmp_path,
        include_searx=False,
        non_interactive=True,
        step_filter=set(),
    )

    def fake_which(command: str) -> str | None:
        if command == "docker-compose":
            return "/usr/local/bin/docker-compose"
        return None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1)
    )

    assert deployer._compose_invocation() == ("/usr/local/bin/docker-compose",)
