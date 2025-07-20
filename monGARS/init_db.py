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


class DummySession:
    def __getattr__(self, name):
        async def _stub(*_, **__):
            if name == "execute":
                return []
            return None

        return _stub


@asynccontextmanager
async def async_session_factory():
    yield DummySession()
