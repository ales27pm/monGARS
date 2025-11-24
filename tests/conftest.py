"""Shared pytest configuration and fixtures."""

import asyncio
import os
import warnings

import pytest
import pytest_asyncio

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
