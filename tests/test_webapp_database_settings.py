"""Tests for Django database configuration helpers."""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_settings_module():
    module = importlib.import_module("webapp.webapp.settings")
    return importlib.reload(module)


def test_database_url_respects_env_overrides(monkeypatch):
    monkeypatch.setenv("DJANGO_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://legacy:oldpw@legacy-host:5432/legacy_db"
    )
    monkeypatch.setenv("DB_USER", "mongars")
    monkeypatch.setenv("DB_PASSWORD", "override-secret")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("DB_NAME", "mongars_db")

    settings = _reload_settings_module()
    config = settings._database_config_from_url(os.environ["DATABASE_URL"])

    assert config["USER"] == "mongars"
    assert config["PASSWORD"] == "override-secret"
    assert config["HOST"] == "postgres"
    assert config["PORT"] == "6543"
    assert config["NAME"] == "mongars_db"


def test_postgres_env_vars_override_all(monkeypatch):
    monkeypatch.setenv("DJANGO_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://legacy:oldpw@legacy-host:5432/legacy_db"
    )
    monkeypatch.setenv("DB_USER", "mongars")
    monkeypatch.setenv("DB_PASSWORD", "override-secret")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("DB_NAME", "mongars_db")
    monkeypatch.setenv("POSTGRES_USER", "pg_superuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg_supersecret")
    monkeypatch.setenv("POSTGRES_HOST", "pg-host")
    monkeypatch.setenv("POSTGRES_PORT", "9999")
    monkeypatch.setenv("POSTGRES_DB", "pg_db")

    settings = _reload_settings_module()
    config = settings._database_config_from_url(os.environ["DATABASE_URL"])

    assert config["USER"] == "pg_superuser"
    assert config["PASSWORD"] == "pg_supersecret"
    assert config["HOST"] == "pg-host"
    assert config["PORT"] == "9999"
    assert config["NAME"] == "pg_db"


def test_blank_env_values_do_not_override(monkeypatch):
    monkeypatch.setenv("DJANGO_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://legacy:oldpw@legacy-host:5432/legacy_db"
    )
    monkeypatch.setenv("DB_HOST", "")
    monkeypatch.setenv("DB_PASSWORD", "  ")

    settings = _reload_settings_module()
    config = settings._database_config_from_url(os.environ["DATABASE_URL"])

    assert config["HOST"] == "legacy-host"
    assert config["PASSWORD"] == "oldpw"


def test_database_url_uses_parsed_values_when_no_env(monkeypatch):
    monkeypatch.setenv("DJANGO_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://legacy:oldpw@legacy-host:6543/legacy_db"
    )
    for key in (
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = _reload_settings_module()
    config = settings._database_config_from_url(os.environ["DATABASE_URL"])

    assert config["NAME"] == "legacy_db"
    assert config["USER"] == "legacy"
    assert config["PASSWORD"] == "oldpw"
    assert config["HOST"] == "legacy-host"
    assert config["PORT"] == "6543"


def test_build_database_settings_fails_without_password_in_production(monkeypatch):
    monkeypatch.setenv("DJANGO_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("DJANGO_DEBUG", "false")
    monkeypatch.setenv("DB_ENGINE", "postgresql")
    monkeypatch.setenv("DB_NAME", "mongars_db")
    monkeypatch.setenv("DB_USER", "mongars")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DJANGO_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="missing required credentials"):
        _reload_settings_module()


def test_build_database_settings_allows_explicit_valid_env_in_production(monkeypatch):
    monkeypatch.setenv("DJANGO_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("DJANGO_DEBUG", "false")
    monkeypatch.setenv("DB_ENGINE", "postgresql")
    monkeypatch.setenv("DB_NAME", "mongars_db")
    monkeypatch.setenv("DB_USER", "mongars")
    monkeypatch.setenv("DB_PASSWORD", "prod-secret")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DJANGO_DATABASE_URL", raising=False)

    settings = _reload_settings_module()
    config = settings._build_database_settings()

    assert config["ENGINE"] == "django.db.backends.postgresql"
    assert config["NAME"] == "mongars_db"
    assert config["USER"] == "mongars"
    assert config["PASSWORD"] == "prod-secret"
    assert config["HOST"] == "postgres"
    assert config["PORT"] == "5432"
