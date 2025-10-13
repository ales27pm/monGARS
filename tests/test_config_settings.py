import os

import pytest

from monGARS import config


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("OTEL_DEBUG", raising=False)
    monkeypatch.delenv("EVENTBUS_USE_REDIS", raising=False)
    yield
    config.get_settings.cache_clear()


def test_get_settings_generates_secret_for_debug(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    settings = config.get_settings()
    assert settings.debug is True
    assert settings.SECRET_KEY is not None
    assert len(settings.SECRET_KEY) >= 32


def test_secret_key_is_random_between_calls(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    first = config.get_settings()
    config.get_settings.cache_clear()
    second = config.get_settings()
    assert first.SECRET_KEY != second.SECRET_KEY
    assert len(first.SECRET_KEY) >= 32
    assert len(second.SECRET_KEY) >= 32


def test_get_settings_bootstraps_secret_in_production(monkeypatch, tmp_path, caplog):
    config.get_settings.cache_clear()
    env_file = tmp_path / ".env"
    env_file.write_text("DEBUG=false\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with caplog.at_level("INFO"):
        settings = config.get_settings()

    assert settings.SECRET_KEY
    assert len(settings.SECRET_KEY) >= 32
    assert "SECRET_KEY missing while DEBUG is disabled" in caplog.text
    assert "Persisted generated SECRET_KEY" in caplog.text
    assert "SECRET_KEY=" in env_file.read_text()

    persisted = settings.SECRET_KEY
    config.get_settings.cache_clear()
    caplog.clear()
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with caplog.at_level("INFO"):
        settings_again = config.get_settings()

    assert settings_again.SECRET_KEY == persisted
    assert settings_again._secret_key_origin == "provided"
    assert "SECRET_KEY missing while DEBUG is disabled" not in caplog.text


def test_get_settings_warns_when_env_file_read_only(monkeypatch, tmp_path, caplog):
    env_file = tmp_path / ".env"
    env_file.write_text("DEBUG=false\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    def _fail_set_key(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(config, "set_key", _fail_set_key)

    with caplog.at_level("WARNING"):
        settings = config.get_settings()

    assert settings.SECRET_KEY
    assert os.environ["SECRET_KEY"] == settings.SECRET_KEY
    assert "Unable to persist SECRET_KEY" in caplog.text
    assert "SECRET_KEY=" not in env_file.read_text(encoding="utf-8")


def test_settings_defers_secret_when_vault_configured(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)

    settings = config.Settings(
        debug=False,
        JWT_ALGORITHM="HS256",
        VAULT_URL="https://vault.example",  # noqa: S106 - test fixture value
        VAULT_TOKEN="unit-test-token",  # noqa: S106 - test fixture value
    )

    assert settings.SECRET_KEY is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vault_secrets",
    [
        {"SECRET_KEY": "vault-secret"},
        {"SECRET_KEY": "vault-secret", "API_KEY": "vault-api-key"},
        {
            "SECRET_KEY": "another-secret",
            "API_KEY": "another-api-key",
            "DB_PASSWORD": "db-pass",
        },
    ],
)
async def test_get_settings_fetches_vault_secret_with_active_loop(
    monkeypatch, vault_secrets
):
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("VAULT_URL", "https://vault.example")
    monkeypatch.setenv("VAULT_TOKEN", "unit-test-token")

    captured: dict[str, config.Settings] = {}

    def _fake_fetch(settings: config.Settings) -> dict[str, str]:
        captured["settings"] = settings
        return vault_secrets

    monkeypatch.setattr(config, "fetch_secrets_from_vault", _fake_fetch)

    settings = config.get_settings()

    assert captured
    for key, value in vault_secrets.items():
        assert getattr(settings, key) == value


@pytest.mark.parametrize("value", ["True", "true", "1"])
def test_debug_env_parsing_variants(monkeypatch, value):
    monkeypatch.setenv("DEBUG", value)
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    settings = config.get_settings()
    assert settings.debug is True


@pytest.mark.parametrize("value", ["False", "false", "0"])
def test_debug_env_false_variants(monkeypatch, value):
    monkeypatch.setenv("DEBUG", value)
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    settings = config.get_settings()
    assert settings.debug is False


@pytest.mark.parametrize("value", ["True", "true", "1"])
def test_otel_debug_env_parsing(monkeypatch, value):
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("OTEL_DEBUG", value)
    settings = config.get_settings()
    assert settings.otel_debug is True


@pytest.mark.parametrize("value", ["True", "true", "1"])
def test_eventbus_use_redis_env_parsing_true(monkeypatch, value):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("EVENTBUS_USE_REDIS", value)
    settings = config.get_settings()
    assert settings.EVENTBUS_USE_REDIS is True


@pytest.mark.parametrize("value", ["False", "false", "0"])
def test_eventbus_use_redis_env_parsing_false(monkeypatch, value):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("EVENTBUS_USE_REDIS", value)
    settings = config.get_settings()
    assert settings.EVENTBUS_USE_REDIS is False


def test_eventbus_use_redis_default_false(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    monkeypatch.delenv("EVENTBUS_USE_REDIS", raising=False)
    settings = config.get_settings()
    assert settings.EVENTBUS_USE_REDIS is False


def test_settings_rejects_private_keys_when_hs256_locked(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")

    with pytest.raises(ValueError, match="not supported with symmetric JWT algorithms"):
        config.Settings(
            JWT_ALGORITHM="HS256",
            JWT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nfoo\n-----END PRIVATE KEY-----",
            JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\nbar\n-----END PUBLIC KEY-----",
        )


def test_settings_reject_non_hs256_algorithm(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")

    with pytest.raises(
        ValueError, match="Asymmetric JWT algorithms require both JWT_PRIVATE_KEY"
    ):
        config.Settings(JWT_ALGORITHM="RS256")


def test_validate_jwt_configuration_allows_hs256_with_secret(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    settings = config.Settings(JWT_ALGORITHM="HS256")

    # Should not raise
    config.validate_jwt_configuration(settings)


def test_validate_jwt_configuration_requires_secret_key(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)

    settings = config.Settings(JWT_ALGORITHM="HS256", SECRET_KEY="unit-test-secret")
    settings_without_secret = settings.model_copy(update={"SECRET_KEY": None})

    with pytest.raises(ValueError, match="require SECRET_KEY"):
        config.validate_jwt_configuration(settings_without_secret)


def test_database_url_password_override(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mongars:changeme@postgres:5432/mongars_db",
    )
    monkeypatch.setenv("DB_PASSWORD", "override-secret")

    settings = config.get_settings()

    assert settings.database_url.password == "override-secret"


def test_database_url_component_overrides(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://legacy:changeme@old-host:5432/legacy_db",
    )
    monkeypatch.setenv("DB_USER", "mongars")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_NAME", "mongars_db")
    monkeypatch.setenv("DB_PORT", "6543")

    settings = config.get_settings()

    assert settings.database_url.username == "mongars"
    assert settings.database_url.host == "postgres"
    assert settings.database_url.port == 6543
    assert settings.database_url.path.lstrip("/") == "mongars_db"
