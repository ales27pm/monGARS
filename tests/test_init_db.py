"""Tests for the lightweight database bootstrap helpers."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest

try:
    from sqlalchemy.engine.url import make_url
    from monGARS import init_db
except ModuleNotFoundError:
    missing_dependencies = [
        name
        for name in (
            "sqlalchemy",
            "hvac",
            "dotenv",
            "opentelemetry",
            "pydantic",
            "pydantic_settings",
        )
        if importlib.util.find_spec(name) is None
    ]

    pytest.skip(
        (
            "Skipping init_db tests because required dependencies are missing: "
            f"{', '.join(sorted(missing_dependencies))}. "
            "Install monGARS[test] or the full requirements to run these tests."
        ),
        allow_module_level=True,
    )


def _load_init_db_script(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "init_db.py"
    fake_alembic = types.ModuleType("alembic")
    fake_alembic.command = types.SimpleNamespace(upgrade=lambda *args, **kwargs: None)
    fake_config_module = types.ModuleType("alembic.config")
    fake_config_module.Config = object
    monkeypatch.setitem(sys.modules, "alembic", fake_alembic)
    monkeypatch.setitem(sys.modules, "alembic.config", fake_config_module)
    spec = importlib.util.spec_from_file_location("mongars_init_db_script", script_path)
    assert spec and spec.loader  # pragma: no cover - sanity check
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_driver_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_init_db_script(monkeypatch)

    if importlib.util.find_spec("psycopg") is not None:
        expected = "postgresql+psycopg"
    elif importlib.util.find_spec("psycopg2") is not None:
        expected = "postgresql+psycopg2"
    else:
        expected = "postgresql"

    for key in {
        "DATABASE_URL",
        "DJANGO_DATABASE_URL",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
    }:
        monkeypatch.delenv(key, raising=False)

    assert module.SYNC_DRIVERNAME == expected
    assert module.build_sync_url().drivername == expected


@pytest.mark.parametrize(
    "env_value,allow_remote,expected_driver",
    [
        ("postgresql://prod.example.com/mongars", False, "sqlite"),
        ("postgresql://prod.example.com/mongars", True, "postgresql+asyncpg"),
        ("not-a-valid-url", False, "sqlite"),
    ],
)
def test_resolve_database_url_enforces_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    allow_remote: bool,
    expected_driver: str,
) -> None:
    monkeypatch.setenv("DATABASE_URL", env_value)
    if allow_remote:
        monkeypatch.setenv("MONGARS_ALLOW_REMOTE_DATABASE_BOOTSTRAP", "1")
    else:
        monkeypatch.delenv("MONGARS_ALLOW_REMOTE_DATABASE_BOOTSTRAP", raising=False)

    default_url = make_url("sqlite:///./fallback.db")
    resolved = init_db._resolve_database_url(env_value, default_url=default_url)

    if expected_driver == "sqlite":
        assert resolved.drivername.startswith("sqlite")
    else:
        assert resolved.drivername == expected_driver


def test_resolve_database_url_rejects_remote_without_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://remote.example.com/live")
    monkeypatch.delenv("MONGARS_ALLOW_REMOTE_DATABASE_BOOTSTRAP", raising=False)

    default_url = make_url("postgresql://localhost/test")
    resolved = init_db._resolve_database_url(
        os.environ.get("DATABASE_URL"), default_url=default_url
    )

    assert resolved.drivername.startswith("sqlite") or resolved.host in {
        "localhost",
        "127.0.0.1",
        "::1",
        None,
        "",
    }


def test_build_sync_url_honours_password_override(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mongars:changeme@postgres:5432/mongars_db",
    )
    monkeypatch.setenv("DB_PASSWORD", "override-secret")
    module = _load_init_db_script(monkeypatch)

    url = module.build_sync_url()

    assert url.password == "override-secret"


def test_build_sync_url_password_override_suppresses_logging(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mongars:changeme@postgres:5432/mongars_db",
    )
    monkeypatch.setenv("DB_PASSWORD", "override-secret")
    module = _load_init_db_script(monkeypatch)

    module.build_sync_url()

    assert "database password" not in caplog.text
    assert "password override" not in caplog.text


def test_build_sync_url_invalid_port_logging(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mongars:changeme@postgres:5432/mongars_db",
    )
    monkeypatch.setenv("DB_PORT", "6543bad")
    module = _load_init_db_script(monkeypatch)

    url = module.build_sync_url()

    assert "Invalid database port override provided; ignoring." in caplog.text
    assert "6543bad" not in caplog.text
    assert url.port == 5432
