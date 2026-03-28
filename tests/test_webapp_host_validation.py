from __future__ import annotations

import os
import sys
from pathlib import Path

import django
import pytest
from django.core.exceptions import DisallowedHost
from django.http import HttpResponse
from django.test import RequestFactory, override_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
django.setup()

from webapp.host_validation import LocalNetworkHostValidationMiddleware  # noqa: E402


def _build_middleware() -> LocalNetworkHostValidationMiddleware:
    return LocalNetworkHostValidationMiddleware(lambda request: HttpResponse("ok"))


@override_settings(
    ALLOW_PRIVATE_NETWORK_HOSTS=True,
    HOST_VALIDATION_ALLOWLIST=("localhost", "127.0.0.1", "0.0.0.0"),
    ALLOWED_HOSTS=["localhost", "127.0.0.1", "0.0.0.0", "*"],
)
def test_private_lan_ip_is_accepted():
    request = RequestFactory().get("/chat/login/", HTTP_HOST="10.0.0.154:8001")

    response = _build_middleware()(request)

    assert response.status_code == 200


@override_settings(
    ALLOW_PRIVATE_NETWORK_HOSTS=True,
    HOST_VALIDATION_ALLOWLIST=("localhost", "127.0.0.1", "0.0.0.0"),
    ALLOWED_HOSTS=["localhost", "127.0.0.1", "0.0.0.0", "*"],
)
def test_public_ip_is_rejected():
    request = RequestFactory().get("/chat/login/", HTTP_HOST="198.51.100.25:8001")

    with pytest.raises(DisallowedHost, match="198.51.100.25:8001"):
        _build_middleware()(request)
