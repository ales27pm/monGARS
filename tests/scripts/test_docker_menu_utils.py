from pathlib import Path

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
