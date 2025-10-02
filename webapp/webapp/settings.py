import os
import socket
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar
from urllib.parse import parse_qsl, unquote, urlparse

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
            _safe_socket_call(socket.getaddrinfo, None, 0, proto=socket.IPPROTO_TCP)
            or []
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

    if address.version == 4:
        return any(address in network for network in _PRIVATE_IPV4_NETWORKS)

    return any(address in network for network in _PRIVATE_IPV6_NETWORKS)


T = TypeVar("T")


def _safe_socket_call(func: Callable[..., T], *args, **kwargs) -> T | None:
    try:
        return func(*args, **kwargs)
    except OSError:
        return None


def _dedupe_hosts(hosts: Iterable[str]) -> list[str]:
    """Return hosts with whitespace stripped and duplicates removed."""

    seen: set[str] = set()
    deduped: list[str] = []
    for host in hosts:
        trimmed = host.strip()
        if trimmed and trimmed not in seen:
            deduped.append(trimmed)
            seen.add(trimmed)
    return deduped


def _parse_host_csv(raw_hosts: str | None) -> list[str]:
    """Parse comma separated hosts from environment variables."""

    if not raw_hosts:
        return []
    return _dedupe_hosts(raw_hosts.split(","))


def _default_allowed_hosts_list() -> list[str]:
    """Compose the baseline ALLOWED_HOSTS for container and local setups."""

    base_hosts = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]
    compose_hosts = []
    compose_hosts.extend(_parse_host_csv(os.environ.get("WEBAPP_HOST")))
    compose_hosts.extend(_parse_host_csv(os.environ.get("HOST")))
    return _dedupe_hosts([*base_hosts, *compose_hosts])


BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY environment variable required")
DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() in ("true", "1")
_DEFAULT_ALLOWED_HOSTS = _default_allowed_hosts_list()
_default_allowed_hosts = ",".join(_DEFAULT_ALLOWED_HOSTS)
_explicit_allowed_hosts = os.environ.get("DJANGO_ALLOWED_HOSTS")

if _explicit_allowed_hosts:
    ALLOWED_HOSTS = _parse_host_csv(_explicit_allowed_hosts)
else:
    ALLOWED_HOSTS = list(_DEFAULT_ALLOWED_HOSTS)

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


def _sqlite_database_settings(path: str | None = None) -> dict[str, Any]:
    """Return a SQLite configuration for explicit local development."""

    sqlite_path = path or os.environ.get("DJANGO_SQLITE_PATH")
    if not sqlite_path:
        sqlite_path = str(BASE_DIR / "db.sqlite3")
    return {"ENGINE": "django.db.backends.sqlite3", "NAME": sqlite_path}


def _database_conn_max_age(engine: str) -> int:
    """Return the connection persistence configured for ``engine``."""

    env_value = os.environ.get("DJANGO_DB_CONN_MAX_AGE")
    if env_value is not None:
        try:
            return int(env_value)
        except ValueError as exc:
            raise RuntimeError("DJANGO_DB_CONN_MAX_AGE must be an integer") from exc

    return 0 if engine == "django.db.backends.sqlite3" else 60


def _database_config_from_url(url: str) -> dict[str, Any]:
    """Build a Django DATABASES configuration from a RFC-1738 style URL."""

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if "+" in scheme:
        scheme = scheme.split("+", 1)[0]

    if scheme in {"postgres", "postgresql"}:
        engine = "django.db.backends.postgresql"
    elif scheme in {"mysql", "mariadb"}:
        engine = "django.db.backends.mysql"
    elif scheme in {"sqlite", "sqlite3"}:
        raw_path = f"{parsed.netloc or ''}{parsed.path or ''}"
        if raw_path.startswith("//"):
            normalized_path = "/" + unquote(raw_path.lstrip("/"))
        else:
            normalized_path = unquote(raw_path.lstrip("/"))
        sqlite_name = normalized_path or ":memory:"
        return _sqlite_database_settings(sqlite_name)
    else:
        raise RuntimeError(f"Unsupported database scheme: {parsed.scheme!r}")

    name = unquote(parsed.path.lstrip("/")) or os.environ.get("DB_NAME") or ""
    host = parsed.hostname or os.environ.get("DB_HOST", "")
    port = str(parsed.port or os.environ.get("DB_PORT", ""))
    user = (
        unquote(parsed.username) if parsed.username else os.environ.get("DB_USER", "")
    )
    password = (
        unquote(parsed.password)
        if parsed.password
        else os.environ.get("DB_PASSWORD", "")
    )
    options = {
        key: value for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    }

    config: dict[str, Any] = {
        "ENGINE": engine,
        "NAME": name,
        "USER": user,
        "PASSWORD": password,
        "HOST": host,
        "PORT": port,
    }

    conn_max_age = _database_conn_max_age(engine)
    if conn_max_age:
        config["CONN_MAX_AGE"] = conn_max_age

    if options:
        config["OPTIONS"] = options

    return config


def _database_config_from_discrete_env() -> dict[str, Any] | None:
    """Derive database settings from ``DB_*`` environment variables."""

    env_values = {key: os.environ.get(key) for key in ("DB_NAME", "DB_USER", "DB_HOST")}
    engine_hint = os.environ.get("DB_ENGINE")
    if not engine_hint and not any(env_values.values()):
        return None

    normalized_engine = (engine_hint or "postgresql").lower()
    if normalized_engine in {"sqlite", "sqlite3"}:
        sqlite_name = os.environ.get("DB_NAME") or os.environ.get("DB_PATH")
        return _sqlite_database_settings(sqlite_name)
    if normalized_engine in {"postgres", "postgresql", "psql"}:
        engine = "django.db.backends.postgresql"
    elif normalized_engine in {"mysql", "mariadb"}:
        engine = "django.db.backends.mysql"
    else:
        engine = engine_hint or "django.db.backends.postgresql"

    name = os.environ.get("DB_NAME") or os.environ.get("POSTGRES_DB", "mongars_db")
    user = os.environ.get("DB_USER") or os.environ.get("POSTGRES_USER", "mongars")
    password = os.environ.get("DB_PASSWORD") or os.environ.get(
        "POSTGRES_PASSWORD", "changeme"
    )
    host = os.environ.get("DB_HOST") or "postgres"
    port = str(os.environ.get("DB_PORT") or os.environ.get("POSTGRES_PORT") or "5432")

    config: dict[str, Any] = {
        "ENGINE": engine,
        "NAME": name,
        "USER": user,
        "PASSWORD": password,
        "HOST": host,
        "PORT": port,
    }

    conn_max_age = _database_conn_max_age(engine)
    if conn_max_age:
        config["CONN_MAX_AGE"] = conn_max_age

    return config


def _default_postgres_settings() -> dict[str, Any]:
    """Return the production-ready Postgres configuration."""

    config: dict[str, Any] = {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME")
        or os.environ.get("POSTGRES_DB")
        or "mongars_db",
        "USER": os.environ.get("DB_USER")
        or os.environ.get("POSTGRES_USER")
        or "mongars",
        "PASSWORD": os.environ.get("DB_PASSWORD")
        or os.environ.get("POSTGRES_PASSWORD")
        or "changeme",
        "HOST": os.environ.get("DB_HOST") or "postgres",
        "PORT": str(
            os.environ.get("DB_PORT") or os.environ.get("POSTGRES_PORT") or "5432"
        ),
    }

    conn_max_age = _database_conn_max_age(config["ENGINE"])
    if conn_max_age:
        config["CONN_MAX_AGE"] = conn_max_age

    return config


def _build_database_settings() -> dict[str, Any]:
    """Compose the DATABASES['default'] configuration from the environment."""

    database_url = os.environ.get("DJANGO_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )
    if database_url:
        return _database_config_from_url(database_url)

    discrete = _database_config_from_discrete_env()
    if discrete:
        return discrete

    sqlite_requested = os.environ.get("DJANGO_USE_SQLITE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if sqlite_requested or os.environ.get("DJANGO_SQLITE_PATH"):
        return _sqlite_database_settings()

    return _default_postgres_settings()


DATABASES = {"default": _build_database_settings()}

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
