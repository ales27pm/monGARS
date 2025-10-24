import os
from typing import Any

import pytest

from webapp.webapp import settings


@pytest.fixture(autouse=True)
def reset_env():
    preserved = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(preserved)


def _base_expected_options() -> dict[str, Any]:
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "mongars_db",
        "USER": "mongars",
        "PASSWORD": "changeme",
        "HOST": "postgres",
        "PORT": "5432",
    }


def test_default_postgres_settings_includes_optional_options(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DATABASE_SSLMODE", "require")
    monkeypatch.setenv("DATABASE_TARGET_SESSION_ATTRS", "read-write")
    monkeypatch.setenv("DATABASE_OPTIONS_JSON", '{"application_name": "monGARS"}')

    config = settings._default_postgres_settings()

    expected = _base_expected_options()
    assert config.items() >= expected.items()
    assert config["OPTIONS"] == {
        "sslmode": "require",
        "target_session_attrs": "read-write",
        "application_name": "monGARS",
    }


def test_default_postgres_settings_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DATABASE_OPTIONS_JSON", "not-json")

    with pytest.raises(RuntimeError):
        settings._default_postgres_settings()
