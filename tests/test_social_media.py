import asyncio
import importlib
import os

import aiohttp
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key")

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


class UnauthorizedSession(FakeSession):
    def post(self, *args, **kwargs):
        return FakeResponse(status=401)


class TimeoutSession:
    def post(self, *args, **kwargs):
        raise asyncio.TimeoutError

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class ErrorSession:
    def post(self, *args, **kwargs):
        raise aiohttp.ClientError("boom")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.asyncio
async def test_post_to_twitter_auth_failure(monkeypatch):
    monkeypatch.setattr(social.aiohttp, "ClientSession", lambda: UnauthorizedSession())
    mgr = social.SocialMediaManager()
    token = security.encrypt_token("secret-token")

    assert not await mgr.post_to_twitter("fail", token)


@pytest.mark.asyncio
async def test_post_to_twitter_timeout(monkeypatch):
    monkeypatch.setattr(social.aiohttp, "ClientSession", lambda: TimeoutSession())
    mgr = social.SocialMediaManager()
    token = security.encrypt_token("secret-token")

    assert not await mgr.post_to_twitter("timeout", token)


@pytest.mark.asyncio
async def test_post_to_twitter_network_error(monkeypatch):
    monkeypatch.setattr(social.aiohttp, "ClientSession", lambda: ErrorSession())
    mgr = social.SocialMediaManager()
    token = security.encrypt_token("secret-token")

    assert not await mgr.post_to_twitter("boom", token)


@pytest.mark.asyncio
async def test_post_to_twitter_bad_token(monkeypatch):
    monkeypatch.setattr(social.aiohttp, "ClientSession", lambda: FakeSession())
    mgr = social.SocialMediaManager()

    assert not await mgr.post_to_twitter("hi", "not-encrypted")
