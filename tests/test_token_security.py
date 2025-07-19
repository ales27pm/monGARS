import importlib
import os

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key")

from monGARS.core import security

importlib.reload(security)


def test_encrypt_and_decrypt_roundtrip():
    token = "mytoken"
    encrypted = security.encrypt_token(token)
    assert encrypted != token
    assert security.decrypt_token(encrypted) == token


def test_encrypt_and_decrypt_with_custom_key(monkeypatch):
    custom_key = "custom-secret-key-987654321098765432109876"
    monkeypatch.setenv("SECRET_KEY", custom_key)
    import monGARS.config as config

    config.get_settings.cache_clear()
    import importlib

    import monGARS.core.security as security_mod

    importlib.reload(security_mod)
    token = "customkeytoken"
    encrypted = security_mod.encrypt_token(token)
    assert encrypted != token
    assert security_mod.decrypt_token(encrypted) == token


def test_decrypt_with_wrong_key(monkeypatch):
    token = "wrongkeytoken"
    encrypted = security.encrypt_token(token)
    monkeypatch.setenv("SECRET_KEY", "another-secret-key-0000000000000000000000")
    import monGARS.config as config

    config.get_settings.cache_clear()
    import importlib

    import monGARS.core.security as security_mod

    importlib.reload(security_mod)
    with pytest.raises(ValueError):
        security_mod.decrypt_token(encrypted)


def test_encrypt_and_decrypt_empty_token():
    token = ""
    encrypted = security.encrypt_token(token)
    assert encrypted != token
    assert security.decrypt_token(encrypted) == token


def test_encrypt_and_decrypt_very_long_token():
    token = "a" * 10000
    encrypted = security.encrypt_token(token)
    assert encrypted != token
    assert security.decrypt_token(encrypted) == token


def test_decrypt_invalid_token():
    with pytest.raises(ValueError):
        security.decrypt_token("invalid")
