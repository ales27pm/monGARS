import os
import socket
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from dotenv import load_dotenv

load_dotenv()


def _iter_debug_hosts() -> Iterable[str]:
    """Yield hostnames and addresses safe to trust while debugging locally."""

    explicit_hosts = _parse_debug_host_env()
    for host in explicit_hosts:
        yield host

    hostname = _safe_socket_call(socket.gethostname)
    if hostname:
        yield hostname

    fqdn = _safe_socket_call(socket.getfqdn)
    if fqdn and fqdn != hostname:
        yield fqdn

    for address in _private_interface_addresses(hostname):
        yield address


def _parse_debug_host_env() -> list[str]:
    """Return hosts declared via ``DJANGO_DEBUG_HOSTS`` in import order."""

    raw_hosts = os.environ.get("DJANGO_DEBUG_HOSTS", "")
    parsed: list[str] = []
    for candidate in raw_hosts.split(","):
        trimmed = candidate.strip()
        if trimmed:
            parsed.append(trimmed)
    return parsed


def _private_interface_addresses(hostname: str | None) -> Iterable[str]:
    """Return private interface addresses discovered on the current machine."""

    discovered: set[str] = set()

    if hostname:
        discovered.update(_resolve_private_ips(hostname))

    try:
        import netifaces
        for iface in netifaces.interfaces():
            for family, addresses in netifaces.ifaddresses(iface).items():
                if family not in (netifaces.AF_INET, netifaces.AF_INET6):
                    continue
                for addr_info in addresses:
                    host = addr_info.get("addr")
                    if host and _is_private_ip(host):
                        discovered.add(host)
    except ImportError:
        # Fallback for when netifaces is not installed
        for info in (
            _safe_socket_call(socket.getaddrinfo, None, 0, proto=socket.IPPROTO_TCP) or []
        ):
            host = info[4][0]
            if _is_private_ip(host):
                discovered.add(host)

    return sorted(discovered)


def _resolve_private_ips(hostname: str) -> Iterable[str]:
    """Resolve ``hostname`` to the private IPs bound locally."""

    resolved: list[str] = []
    host_info = _safe_socket_call(socket.gethostbyname_ex, hostname)
    if host_info:
        _, _, addresses = host_info
        for address in addresses:
            if _is_private_ip(address):
                resolved.append(address)
    return resolved


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


def _is_private_ip(value: str) -> bool:
    try:
        address = ip_address(value)
    except ValueError:
        return False

    return address.is_private


T = TypeVar("T")


def _safe_socket_call(func: Callable[..., T], *args, **kwargs) -> T | None:
    try:
        return func(*args, **kwargs)
    except OSError:
        return None


BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY environment variable required")
DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() in ("true", "1")
_default_allowed_hosts = "localhost,127.0.0.1,[::1]"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", _default_allowed_hosts).split(
        ","
    )
    if host.strip()
]

if DEBUG:
    for debug_host in _iter_debug_hosts():
        if debug_host not in ALLOWED_HOSTS:
            ALLOWED_HOSTS.append(debug_host)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "chat",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "webapp.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "webapp.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True

LOGGING = {
    "version": 1,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}
