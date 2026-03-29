"""Tests for the lightweight database bootstrap helpers."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest

os.environ["DEBUG"] = "true"
os.environ["DATABASE_URL"] = (
    "postgresql+asyncpg://postgres:postgres@localhost:5432/mongars_db"
)
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL
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
            "passlib",
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


def _prepare_bootstrap_db(tmp_path: Path) -> URL:
    db_path = tmp_path / "bootstrap.db"
    url = make_url(f"sqlite:///{db_path}")
    engine = create_engine(url.render_as_string(hide_password=False))
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE user_accounts (
                        username TEXT PRIMARY KEY,
                        password_hash TEXT NOT NULL,
                        is_admin BOOLEAN NOT NULL
                    )
                    """
                )
            )
    finally:
        engine.dispose()
    return url


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


def test_bootstrap_admin_credentials_require_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_init_db_script(monkeypatch)

    for key in (
        "ENABLE_ADMIN_BOOTSTRAP",
        "BOOTSTRAP_ADMIN_USERNAME",
        "BOOTSTRAP_ADMIN_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)

    assert module._bootstrap_admin_credentials() is None

    monkeypatch.setenv("ENABLE_ADMIN_BOOTSTRAP", "true")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", " ")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "secret")

    assert module._bootstrap_admin_credentials() is None

    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", " admin ")

    assert module._bootstrap_admin_credentials() == ("admin", "secret")


def test_ensure_bootstrap_admin_inserts_first_admin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_init_db_script(monkeypatch)
    url = _prepare_bootstrap_db(tmp_path)
    monkeypatch.setenv("ENABLE_ADMIN_BOOTSTRAP", "true")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "admin")

    created = module.ensure_bootstrap_admin(url)

    assert created is True

    engine = create_engine(url.render_as_string(hide_password=False))
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT username, password_hash, is_admin FROM user_accounts"
                )
            ).mappings().one()
    finally:
        engine.dispose()

    assert row["username"] == "admin"
    assert bool(row["is_admin"]) is True
    assert row["password_hash"] != "admin"
    assert module._BOOTSTRAP_PASSWORD_CONTEXT.verify("admin", row["password_hash"])


def test_ensure_bootstrap_admin_skips_when_admin_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_init_db_script(monkeypatch)
    url = _prepare_bootstrap_db(tmp_path)
    engine = create_engine(url.render_as_string(hide_password=False))
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO user_accounts (username, password_hash, is_admin) "
                    "VALUES (:username, :password_hash, :is_admin)"
                ),
                {
                    "username": "existing-admin",
                    "password_hash": "existing-hash",
                    "is_admin": True,
                },
            )
    finally:
        engine.dispose()

    monkeypatch.setenv("ENABLE_ADMIN_BOOTSTRAP", "true")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "admin")

    created = module.ensure_bootstrap_admin(url)

    assert created is False

    engine = create_engine(url.render_as_string(hide_password=False))
    try:
        with engine.connect() as conn:
            usernames = conn.execute(
                text("SELECT username FROM user_accounts ORDER BY username")
            ).scalars().all()
    finally:
        engine.dispose()

    assert usernames == ["existing-admin"]
