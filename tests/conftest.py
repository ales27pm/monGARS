"""Shared pytest configuration and fixtures."""

import asyncio
import os
import warnings

import pytest

try:
    import pytest_asyncio
except ImportError:  # pragma: no cover - dependency guard for lightweight test runs
    class _PytestAsyncioFallback:
        """Provide a graceful skip when pytest-asyncio is not installed."""

        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                @pytest.fixture(*args, **kwargs)
                def wrapper(*fargs, **fkwargs):
                    pytest.skip(
                        "pytest-asyncio is required for async fixtures; install monGARS[test] "
                        "or add pytest-asyncio to your environment to run this test.",
                        allow_module_level=False,
                    )

                return wrapper

            return decorator

    pytest_asyncio = _PytestAsyncioFallback()

# Ensure the lightweight sqlite backend is used for tests to avoid external
# database dependencies while pgvector-backed code paths remain exercised in
# unit tests through mocks. ``setdefault`` is intentionally avoided so the
# test suite always overrides CI-provided connection strings.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./mongars_test.db"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["DJANGO_SECRET_KEY"] = "test-django-secret-key"
os.environ["JWT_ALGORITHM"] = "HS256"


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for tests that expect the legacy fixture."""

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"^awq(\.|$)",
)


@pytest_asyncio.fixture
async def ensure_test_users() -> None:
    """Provision standard test users expected by API contract suites."""

    from monGARS.api.dependencies import get_persistence_repository
    from monGARS.core.security import SecurityManager

    repo = get_persistence_repository()
    sec_manager = SecurityManager()
    defaults = (
        ("u1", "x", True),
        ("u2", "y", False),
    )
    for username, password, is_admin in defaults:
        try:
            await repo.create_user_atomic(
                username,
                sec_manager.get_password_hash(password),
                is_admin=is_admin,
            )
        except ValueError:
            continue
    yield
