from datetime import timedelta

import pytest

from monGARS.config import Settings
from monGARS.core.security import SecurityManager


@pytest.fixture()
def hs256_settings(monkeypatch) -> Settings:
    monkeypatch.delenv("SECRET_KEY", raising=False)
    return Settings(SECRET_KEY="unit-test-secret")


def test_security_manager_rejects_non_hs_algorithm(hs256_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires HS256"):
        SecurityManager(
            settings=hs256_settings,
            secret_key=hs256_settings.SECRET_KEY,
            algorithm="RS256",
        )


def test_security_manager_rejects_asymmetric_material(hs256_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Asymmetric JWT keys are not supported"):
        SecurityManager(
            settings=hs256_settings,
            private_key="-----BEGIN PRIVATE KEY-----\nfoo\n-----END PRIVATE KEY-----",
        )


def test_security_manager_generates_hs256_tokens(hs256_settings: Settings) -> None:
    manager = SecurityManager(settings=hs256_settings)

    token = manager.create_access_token(
        {"sub": "alice"}, expires_delta=timedelta(minutes=5)
    )
    payload = manager.verify_token(token)

    assert payload["sub"] == "alice"
    assert "exp" in payload
