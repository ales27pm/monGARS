import importlib
import os

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-123456789012345678901234")

from monGARS.core import security, social

importlib.reload(social)
importlib.reload(security)


class FakeResponse:
    def __init__(self, status: int = 201) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class FakeSession:
    def post(self, *args, **kwargs):
        return FakeResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.asyncio
async def test_post_to_twitter(monkeypatch):
    monkeypatch.setattr(social.aiohttp, "ClientSession", lambda: FakeSession())
    mgr = social.SocialMediaManager()
    token = security.encrypt_token("secret-token")

    assert await mgr.post_to_twitter("hi", token)
