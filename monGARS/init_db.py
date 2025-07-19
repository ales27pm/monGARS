from __future__ import annotations

from contextlib import asynccontextmanager


class ConversationHistory:
    pass


class Interaction:
    pass


class UserPreferences:
    pass


class UserPersonality:
    pass


@asynccontextmanager
async def async_session_factory():
    class DummySession:
        async def execute(self, *a, **k):
            return []

        async def scalar_one_or_none(self):
            return None

        async def commit(self):
            pass

        async def merge(self, *a, **k):
            pass

    yield DummySession()
