"""Shared pytest configuration and fixtures."""

import os

# Ensure the lightweight sqlite backend is used for tests to avoid external
# database dependencies while pgvector-backed code paths remain exercised in
# unit tests through mocks. ``setdefault`` is intentionally avoided so the
# test suite always overrides CI-provided connection strings.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./mongars_test.db"
os.environ.setdefault("SECRET_KEY", "test-secret-key")
