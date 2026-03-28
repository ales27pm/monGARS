"""Helpers for validating local browser origins for WebSocket and CORS flows."""

from __future__ import annotations

from ipaddress import ip_address, ip_network
from typing import Any, Iterable
from urllib.parse import urlsplit

_PRIVATE_IPV4_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("169.254.0.0/16"),
    ip_network("127.0.0.0/8"),
    ip_network("100.64.0.0/10"),
)
_PRIVATE_IPV6_NETWORKS = (
    ip_network("fc00::/7"),
    ip_network("fe80::/10"),
    ip_network("::1/128"),
)
_LOCAL_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
    r"10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}|"
    r"169\.254(?:\.\d{1,3}){2}|"
    r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2}|"
    r"\[(?:::1|f[c-d][0-9a-fA-F:]+|fe[89abAB][0-9a-fA-F:]+)\]"
    r")(?::\d{1,5})?$"
)


def normalise_origin(origin: Any) -> str:
    return str(origin).strip().rstrip("/") if origin is not None else ""


def _origin_host(origin: str) -> str:
    try:
        return (urlsplit(origin).hostname or "").strip().lower()
    except ValueError:
        return ""


def _is_private_ip(host: str) -> bool:
    try:
        address = ip_address(host)
    except ValueError:
        return False

    if address.version == 4:
        return any(address in network for network in _PRIVATE_IPV4_NETWORKS)

    return any(address in network for network in _PRIVATE_IPV6_NETWORKS)


def is_local_origin(origin: str) -> bool:
    host = _origin_host(origin)
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "api", "webapp", "nginx"}:
        return True
    return _is_private_ip(host)


def default_allow_private_network_origins(origins: Iterable[Any]) -> bool:
    candidates = [
        normalise_origin(origin)
        for origin in origins
        if normalise_origin(origin) and normalise_origin(origin) != "*"
    ]
    if not candidates:
        return False
    return all(is_local_origin(origin) for origin in candidates)


def is_allowed_ws_origin(origin: str, allowed_origins: Iterable[Any]) -> bool:
    normalised = normalise_origin(origin)
    allowed = {
        normalise_origin(candidate)
        for candidate in allowed_origins
        if normalise_origin(candidate)
    }
    if not normalised:
        return False
    if "*" in allowed or normalised in allowed:
        return True
    return default_allow_private_network_origins(allowed) and is_local_origin(
        normalised
    )


def private_network_origin_regex(allowed_origins: Iterable[Any]) -> str | None:
    if default_allow_private_network_origins(allowed_origins):
        return _LOCAL_ORIGIN_REGEX
    return None
