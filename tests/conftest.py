"""Shared pytest configuration and fixtures."""

import os

# Ensure the lightweight sqlite backend is used for tests to avoid external
# database dependencies while pgvector-backed code paths remain exercised in
# unit tests through mocks.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./mongars_test.db")
