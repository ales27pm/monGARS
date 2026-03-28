"""Regression tests for Django settings helpers."""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))


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


def test_local_only_allowlist_enables_private_network_hosts(monkeypatch):
    """Loopback-only deployments should accept LAN IP host headers."""

    monkeypatch.setenv("DJANGO_SECRET_KEY", "dummy")
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
    monkeypatch.setenv("DJANGO_DEBUG", "false")
    monkeypatch.delenv("DJANGO_DEBUG_HOSTS", raising=False)
    monkeypatch.delenv("WEBAPP_HOST", raising=False)
    monkeypatch.delenv("HOST", raising=False)

    settings = _reload_settings()

    assert settings.ALLOW_PRIVATE_NETWORK_HOSTS is True
    assert "*" in settings.ALLOWED_HOSTS
    assert {"localhost", "127.0.0.1", "0.0.0.0"} <= set(
        settings.HOST_VALIDATION_ALLOWLIST
    )


def test_public_allowlist_keeps_strict_host_validation(monkeypatch):
    """Public deployments should not widen ALLOWED_HOSTS automatically."""

    monkeypatch.setenv("DJANGO_SECRET_KEY", "dummy")
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "app.example.com")
    monkeypatch.setenv("DJANGO_DEBUG", "false")
    monkeypatch.delenv("DJANGO_DEBUG_HOSTS", raising=False)
    monkeypatch.delenv("WEBAPP_HOST", raising=False)
    monkeypatch.delenv("HOST", raising=False)

    settings = _reload_settings()

    assert settings.ALLOW_PRIVATE_NETWORK_HOSTS is False
    assert "*" not in settings.ALLOWED_HOSTS
    assert settings.HOST_VALIDATION_ALLOWLIST == ("app.example.com", "0.0.0.0")


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
    sys.modules.pop("webapp.settings", None)
    sys.modules.pop("webapp.webapp.settings", None)
    return importlib.import_module("webapp.settings")
