import os

import httpx
import pytest
import pytest_asyncio

from monGARS.api.authentication import get_current_user
from monGARS.api.dependencies import get_peer_communicator
from monGARS.api.web_api import app
from monGARS.core.peer import PeerCommunicator


@pytest.fixture(autouse=True)
def secret_key_env(monkeypatch):
    original = os.environ.get("SECRET_KEY")
    monkeypatch.setenv("SECRET_KEY", "test-peer")
    yield
    if original is not None:
        monkeypatch.setenv("SECRET_KEY", original)
    else:
        monkeypatch.delenv("SECRET_KEY", raising=False)


@pytest_asyncio.fixture
async def client(secret_key_env):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as async_client:
        app.dependency_overrides[get_current_user] = lambda: {"sub": "u1"}
        comm = get_peer_communicator()
        comm.peers = ["http://test/api/v1/peer/message"]
        comm._client = async_client
        yield async_client
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_peer_message_roundtrip(client, monkeypatch):
    captured = {}

    original = PeerCommunicator.decode

    def capture(payload: str):
        data = original(payload)
        captured["data"] = data
        return data

    monkeypatch.setattr(PeerCommunicator, "decode", staticmethod(capture))
    communicator = PeerCommunicator(["http://test/api/v1/peer/message"], client)
    message = {"hello": "world"}
    results = await communicator.send(message)
    assert results == [True]
    assert captured["data"] == message


@pytest.mark.asyncio
async def test_peer_invalid_url(monkeypatch):
    async def failing_post(*args, **kwargs):
        raise httpx.RequestError("fail")

    client = httpx.AsyncClient()
    monkeypatch.setattr(client, "post", failing_post)
    communicator = PeerCommunicator(["http://bad"], client=client)
    results = await communicator.send({"x": "y"})
    assert results == [False]
    await client.aclose()


@pytest.mark.asyncio
async def test_peer_non_200_response(monkeypatch):
    class MockResp:
        status_code = 500

        async def aclose(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResp()

    client = httpx.AsyncClient()
    monkeypatch.setattr(client, "post", mock_post)
    communicator = PeerCommunicator(["http://test"], client=client)
    results = await communicator.send({"x": "y"})
    assert results == [False]
    await client.aclose()


@pytest.mark.asyncio
async def test_peer_empty_message(client):
    communicator = PeerCommunicator(["http://test/api/v1/peer/message"], client)
    results = await communicator.send({})
    assert results == [False]


@pytest.mark.asyncio
async def test_peer_missing_message(client):
    communicator = PeerCommunicator(["http://test/api/v1/peer/message"], client)
    results = await communicator.send(None)
    assert results == [False]
