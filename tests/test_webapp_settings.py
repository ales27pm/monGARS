"""Regression tests for Django settings helpers."""

from __future__ import annotations

import importlib
import socket
import sys
from typing import Any


def test_debug_includes_private_addresses(monkeypatch):
    """Mobile devices should be able to target private debug IPs without 400s."""

    monkeypatch.setenv("DJANGO_SECRET_KEY", "dummy")
    monkeypatch.setenv("DJANGO_DEBUG", "true")
    monkeypatch.delenv("DJANGO_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("DJANGO_DEBUG_HOSTS", raising=False)
    monkeypatch.delenv("WEBAPP_HOST", raising=False)
    monkeypatch.delenv("HOST", raising=False)

    monkeypatch.setattr("socket.gethostname", lambda: "mongars-dev")
    monkeypatch.setattr("socket.getfqdn", lambda: "mongars-dev.local")
    monkeypatch.setattr(
        "socket.gethostbyname_ex",
        lambda _hostname: ("mongars-dev", ["alias"], ["192.168.1.50", "203.0.113.9"]),
    )

    def fake_getaddrinfo(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        return [
            (socket.AF_INET, None, None, "", ("10.0.5.77", 0)),
            (socket.AF_INET, None, None, "", ("0.0.0.0", 0)),
            (socket.AF_INET6, None, None, "", ("fd00::1", 0, 0, 0)),
            (socket.AF_INET6, None, None, "", ("2001:4860::1", 0, 0, 0)),
        ]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)

    settings = _reload_settings()

    allowed = set(settings.ALLOWED_HOSTS)
    assert {
        "mongars-dev",
        "mongars-dev.local",
        "192.168.1.50",
        "10.0.5.77",
        "fd00::1",
    } <= allowed
    assert "203.0.113.9" not in allowed
    assert "0.0.0.0" in allowed
    assert "2001:4860::1" not in allowed


def test_debug_env_hosts_are_preserved(monkeypatch):
    """Explicit debug hosts should supplement discovered addresses."""

    monkeypatch.setenv("DJANGO_SECRET_KEY", "dummy")
    monkeypatch.setenv("DJANGO_DEBUG", "true")
    monkeypatch.setenv("DJANGO_DEBUG_HOSTS", "dev.box,192.168.99.88")
    monkeypatch.delenv("DJANGO_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("WEBAPP_HOST", raising=False)
    monkeypatch.delenv("HOST", raising=False)

    monkeypatch.setattr("socket.gethostname", lambda: "broken", raising=False)

    def erroring(
        *_args: Any, **_kwargs: Any
    ):  # pragma: no cover - verifying resilience
        raise OSError("network lookup disabled in test")

    monkeypatch.setattr("socket.getfqdn", erroring)
    monkeypatch.setattr("socket.gethostbyname_ex", erroring)
    monkeypatch.setattr("socket.getaddrinfo", erroring)

    settings = _reload_settings()

    assert "dev.box" in settings.ALLOWED_HOSTS
    assert "192.168.99.88" in settings.ALLOWED_HOSTS


def test_compose_defaults_include_container_host(monkeypatch):
    """Docker Compose environments should not require manual host overrides."""

    monkeypatch.setenv("DJANGO_SECRET_KEY", "dummy")
    monkeypatch.delenv("DJANGO_DEBUG", raising=False)
    monkeypatch.delenv("DJANGO_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("DJANGO_DEBUG_HOSTS", raising=False)
    monkeypatch.setenv("WEBAPP_HOST", "compose.webapp")
    monkeypatch.setenv("HOST", "compose.webapp")

    settings = _reload_settings()

    assert "0.0.0.0" in settings.ALLOWED_HOSTS
    assert settings.ALLOWED_HOSTS.count("compose.webapp") == 1
    assert "compose.webapp" in settings.ALLOWED_HOSTS


def _reload_settings():
    sys.modules.pop("webapp.webapp.settings", None)
    return importlib.import_module("webapp.webapp.settings")
