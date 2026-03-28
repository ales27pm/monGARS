from __future__ import annotations

from ipaddress import ip_address, ip_network
from typing import Iterable

from django.conf import settings
from django.core.exceptions import DisallowedHost
from django.http.request import split_domain_port, validate_host

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


def is_private_ip(value: str) -> bool:
    try:
        address = ip_address(value)
    except ValueError:
        return False

    if address.version == 4:
        return any(address in network for network in _PRIVATE_IPV4_NETWORKS)

    return any(address in network for network in _PRIVATE_IPV6_NETWORKS)


def is_local_host(host: str) -> bool:
    trimmed = host.strip().strip("[]")
    if not trimmed:
        return False
    if trimmed in {"localhost", "0.0.0.0", "webapp", "api", "nginx"}:
        return True
    return is_private_ip(trimmed)


def default_allow_private_network_hosts(hosts: Iterable[str], *, debug: bool) -> bool:
    if debug:
        return True
    candidates = [host for host in hosts if host and host != "*"]
    if not candidates:
        return False
    return all(is_local_host(host) for host in candidates)


class LocalNetworkHostValidationMiddleware:
    """Allow RFC1918/ULA host headers only for local-only deployments."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not getattr(settings, "ALLOW_PRIVATE_NETWORK_HOSTS", False):
            return self.get_response(request)

        allowlist = tuple(getattr(settings, "HOST_VALIDATION_ALLOWLIST", ()))
        if "*" in allowlist:
            return self.get_response(request)

        raw_host = request.get_host()
        domain, _port = split_domain_port(raw_host)
        host = (domain or raw_host).strip().rstrip(".").lower()
        if validate_host(host, allowlist) or is_local_host(host):
            return self.get_response(request)

        raise DisallowedHost(
            f"Invalid HTTP_HOST header: {raw_host!r}. Add it to DJANGO_ALLOWED_HOSTS."
        )
