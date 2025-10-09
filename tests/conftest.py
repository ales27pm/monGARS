"""Shared pytest configuration and fixtures."""

import os
import warnings

# Ensure the lightweight sqlite backend is used for tests to avoid external
# database dependencies while pgvector-backed code paths remain exercised in
# unit tests through mocks. ``setdefault`` is intentionally avoided so the
# test suite always overrides CI-provided connection strings.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./mongars_test.db"
os.environ["SECRET_KEY"] = "test-secret-key"

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"^awq(\.|$)",
)

import pytest_asyncio

from monGARS.api.dependencies import get_persistence_repository
from monGARS.core.security import SecurityManager


@pytest_asyncio.fixture
async def ensure_test_users() -> None:
    """Provision standard test users expected by API contract suites."""

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
