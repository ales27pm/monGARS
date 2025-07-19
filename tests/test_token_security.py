import importlib
import os

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-123456789012345678901234")

from monGARS.core import security

importlib.reload(security)


def test_encrypt_and_decrypt_roundtrip():
    token = "mytoken"
    encrypted = security.encrypt_token(token)
    assert encrypted != token
    assert security.decrypt_token(encrypted) == token


def test_decrypt_invalid_token():
    with pytest.raises(ValueError):
        security.decrypt_token("invalid")
