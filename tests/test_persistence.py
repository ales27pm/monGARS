from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.exc import OperationalError

from monGARS.core.persistence import PersistenceRepository


class _DummySession:
    async def __aenter__(self) -> "_DummySession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def in_transaction(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_persistence_retries_and_surfaces_connection_failure():
    attempts: list[int] = []

    @asynccontextmanager
    async def failing_session_factory():
        yield _DummySession()

    repo = PersistenceRepository(session_factory=failing_session_factory)

    async def failing_operation(_session):
        attempts.append(1)
        raise OperationalError("SELECT 1", {}, RuntimeError("db down"))

    with pytest.raises(OperationalError):
        await repo._execute_with_retry(
            failing_operation, operation_name="failing_operation"
        )

    assert len(attempts) == 3
