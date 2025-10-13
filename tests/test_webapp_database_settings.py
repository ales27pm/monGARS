"""Tests for Django database configuration helpers."""

from __future__ import annotations

import importlib
import os


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
