import pytest

from monGARS import config


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("OTEL_DEBUG", raising=False)
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


def test_get_settings_requires_secret_in_production(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    with pytest.raises(ValueError, match="SECRET_KEY must be provided in production"):
        config.get_settings()


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


def test_settings_rejects_private_keys_when_hs256_locked(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")

    with pytest.raises(ValueError, match="not supported when JWT_ALGORITHM is HS256"):
        config.Settings(
            JWT_ALGORITHM="HS256",
            JWT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nfoo\n-----END PRIVATE KEY-----",
            JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\nbar\n-----END PUBLIC KEY-----",
        )


def test_settings_reject_non_hs256_algorithm(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")

    with pytest.raises(ValueError, match="must be set to HS256"):
        config.Settings(JWT_ALGORITHM="RS256")


def test_validate_jwt_configuration_allows_hs256_with_secret(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    settings = config.Settings(JWT_ALGORITHM="HS256")

    # Should not raise
    config.validate_jwt_configuration(settings)


def test_validate_jwt_configuration_requires_secret_key(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)

    settings = config.Settings(JWT_ALGORITHM="HS256")

    with pytest.raises(ValueError, match="requires SECRET_KEY"):
        config.validate_jwt_configuration(settings)
