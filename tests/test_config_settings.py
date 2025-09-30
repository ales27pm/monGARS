import pytest

from monGARS import config


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    yield
    config.get_settings.cache_clear()


def test_get_settings_generates_secret_for_debug(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    settings = config.get_settings()
    assert settings.debug is True
    assert settings.SECRET_KEY is not None
    assert len(settings.SECRET_KEY) >= 32


def test_get_settings_requires_secret_in_production(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    with pytest.raises(ValueError):
        config.get_settings()
