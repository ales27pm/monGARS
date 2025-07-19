import asyncio
import os

import pytest

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from monGARS.core.llm_integration import CircuitBreaker


@pytest.mark.asyncio
async def test_circuit_breaker_trips_and_recovers():
    cb = CircuitBreaker(fail_max=2, reset_timeout=1)

    async def fail():
        raise RuntimeError("boom")

    async def succeed():
        return 42

    with pytest.raises(RuntimeError):
        await cb.call(fail)
    with pytest.raises(RuntimeError):
        await cb.call(fail)
    with pytest.raises(Exception):
        await cb.call(succeed)

    await asyncio.sleep(1.1)
    result = await cb.call(succeed)
    assert result == 42
