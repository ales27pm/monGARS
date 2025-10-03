"""Tests for the lightweight database bootstrap helpers."""

from __future__ import annotations

import os

import pytest
from sqlalchemy.engine import make_url

from monGARS import init_db


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
