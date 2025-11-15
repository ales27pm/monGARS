import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts import docker_menu


def test_extract_compose_version_handles_various_formats():
    assert (
        docker_menu.DockerMenu._extract_compose_version(
            "Docker Compose version v2.24.6"
        )
        == "2.24.6"
    )
    assert docker_menu.DockerMenu._extract_compose_version("v2.5.1") == "2.5.1"
    assert docker_menu.DockerMenu._extract_compose_version("2.0.0") == "2.0.0"
    assert docker_menu.DockerMenu._extract_compose_version("version vv2.3.1") == "2.3.1"
    assert docker_menu.DockerMenu._extract_compose_version("") == ""
    assert (
        docker_menu.DockerMenu._extract_compose_version("Docker version without digits")
        == "Docker version without digits"
    )


def test_mask_env_value_redacts_sensitive_tokens():
    assert (
        docker_menu.DockerMenu._mask_env_value("SECRET_KEY", "abc1234567") == "abc1…67"
    )
    assert docker_menu.DockerMenu._mask_env_value("PASSWORD", "short") == "***"
    assert docker_menu.DockerMenu._mask_env_value("PORT", "8000") == "8000"
    assert (
        docker_menu.DockerMenu._mask_env_value("TOKEN", "tokensecretvalue") == "toke…ue"
    )
    assert (
        docker_menu.DockerMenu._mask_env_value("DB_PASSWORD", "dbpass1234") == "dbpa…34"
    )
    assert (
        docker_menu.DockerMenu._mask_env_value("Api_Token", "mixedcasevalue")
        == "mixe…ue"
    )
    assert docker_menu.DockerMenu._mask_env_value("ACCESS_TOKEN", "xy") == "***"
    assert docker_menu.DockerMenu._mask_env_value("SECRET_KEY", "") == "***"
    assert docker_menu.DockerMenu._mask_env_value("USERNAME", "") == ""
    assert docker_menu.DockerMenu._mask_env_value("TOKEN", "abcdef") == "***"
    assert docker_menu.DockerMenu._mask_env_value("TOKEN", "abcdefg") == "abcd…fg"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            """\nNAME SERVICE STATUS PORTS\napp-api-1 api running (healthy) 8000/tcp\napp-db-1 postgres exited (1) \n""",
            [
                {
                    "name": "app-api-1",
                    "service": "api",
                    "status": "running (healthy)",
                    "ports": "8000/tcp",
                },
                {
                    "name": "app-db-1",
                    "service": "postgres",
                    "status": "exited (1)",
                    "ports": "",
                },
            ],
        ),
        ("", []),
        (
            """\nNAME SERVICE STATUS PORTS\napp redis running 0.0.0.0:6379->6379/tcp\n""",
            [
                {
                    "name": "app",
                    "service": "redis",
                    "status": "running",
                    "ports": "0.0.0.0:6379->6379/tcp",
                }
            ],
        ),
        (
            """\nNAME SERVICE STATUS PORTS\napp redis restarting (1) 0.0.0.0:6379->6379/tcp\n""",
            [
                {
                    "name": "app",
                    "service": "redis",
                    "status": "restarting (1)",
                    "ports": "0.0.0.0:6379->6379/tcp",
                }
            ],
        ),
        (
            """\nNAME SERVICE STATUS PORTS\napp redis running\n""",
            [
                {
                    "name": "app",
                    "service": "redis",
                    "status": "running",
                    "ports": "",
                }
            ],
        ),
    ],
)
def test_parse_compose_ps_table(raw: str, expected: list[dict[str, str]]):
    parsed = docker_menu.DockerMenu._parse_compose_ps_table(raw)
    simplified = [
        {
            "name": entry.get("name"),
            "service": entry.get("service"),
            "status": entry.get("status"),
            "ports": entry.get("ports"),
        }
        for entry in parsed
    ]
    assert simplified == expected


def test_parse_env_file_ignores_comments(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("""# comment\nKEY=value\nEMPTY=\n#ANOTHER=\n""")
    parsed = docker_menu.DockerMenu._parse_env_file(env_file)
    assert parsed == {"KEY": "value", "EMPTY": ""}


def test_prepare_ports_updates_searx_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        docker_menu.DockerMenu,
        "_resolve_compose_binary",
        lambda self: ["docker", "compose"],
    )

    menu = docker_menu.DockerMenu()
    menu.env_file = tmp_path / ".env"
    menu.env_template = tmp_path / ".env.example"
    menu.env_file.write_text(
        "\n".join(
            [
                "SEARXNG_PORT=8082",
                "SEARXNG_BASE_URL=http://localhost:8082",
                "SEARCH_SEARX_BASE_URL=http://localhost:8082",
                "SEARCH_SEARX_INTERNAL_BASE_URL=http://searxng:8080",
            ]
        )
        + "\n",
        encoding="utf8",
    )

    def fake_find_available_port(start: int, _reserved: set[int]) -> int:
        return 9090 if start == 8082 else start

    monkeypatch.setattr(menu, "_find_available_port", fake_find_available_port)

    menu.prepare_ports()
    values = docker_menu.DockerMenu._parse_env_file(menu.env_file)

    assert values["SEARXNG_PORT"] == "9090"
    assert values["SEARXNG_BASE_URL"] == "http://localhost:9090"
    assert values["SEARCH_SEARX_BASE_URL"] == "http://localhost:9090"


def test_synchronise_searx_urls_skips_remote_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker_menu.DockerMenu,
        "_resolve_compose_binary",
        lambda self: ["docker", "compose"],
    )

    menu = docker_menu.DockerMenu()
    env_values = {
        "SEARXNG_PORT": "8082",
        "SEARXNG_BASE_URL": "https://search.example",
        "SEARCH_SEARX_BASE_URL": "https://search.example",
    }
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        menu, "_write_env_updates", lambda updates: captured.update(updates)
    )

    menu._synchronise_searx_urls(env_values)

    assert captured == {}
    assert env_values["SEARXNG_BASE_URL"] == "https://search.example"
    assert env_values["SEARCH_SEARX_BASE_URL"] == "https://search.example"


def test_generate_base_model_bundle_invokes_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        docker_menu.DockerMenu,
        "_resolve_compose_binary",
        lambda self: ["docker", "compose"],
    )

    menu = docker_menu.DockerMenu()

    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text('{"prompt": "p", "completion": "c"}\n')
    install_root = tmp_path / "install"

    monkeypatch.setattr(menu, "_default_dataset_path", lambda: dataset_file)
    monkeypatch.setattr(menu, "_model_install_dir", lambda: install_root)

    inputs = iter(["", "hf/test-model"])
    monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

    recorded = {}

    def fake_run(
        command, *, capture_output: bool, check: bool
    ) -> CompletedProcess[str]:
        recorded["command"] = command
        install_root.mkdir(parents=True, exist_ok=True)
        (install_root / "wrapper").mkdir(parents=True, exist_ok=True)
        (install_root / "chat_lora").mkdir(parents=True, exist_ok=True)
        return CompletedProcess(command, 0)

    monkeypatch.setattr(menu, "_run_command", fake_run)

    logs: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        menu, "log", lambda message, error=False: logs.append((message, error))
    )
    blocks: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        menu,
        "log_block",
        lambda heading, lines: blocks.append((heading, list(lines))),
    )

    menu.generate_base_model_bundle()

    command = recorded["command"]
    assert command[0] == sys.executable
    assert command[1].endswith("run_mongars_llm_pipeline.py")
    assert "--dataset-path" in command
    assert "--output-dir" in command
    assert "--skip-smoke-tests" in command
    assert "--skip-merge" in command
    assert blocks[-1][0] == "Base model + wrapper installation complete"


def test_generate_base_model_bundle_validates_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        docker_menu.DockerMenu,
        "_resolve_compose_binary",
        lambda self: ["docker", "compose"],
    )

    menu = docker_menu.DockerMenu()

    missing_dataset = tmp_path / "missing.jsonl"
    monkeypatch.setattr(menu, "_default_dataset_path", lambda: missing_dataset)
    monkeypatch.setattr("builtins.input", lambda *_args: "")

    logs: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        menu, "log", lambda message, error=False: logs.append((message, error))
    )

    menu.generate_base_model_bundle()

    assert logs
    assert any(error for _, error in logs)
    assert "Dataset not found" in logs[-1][0]
