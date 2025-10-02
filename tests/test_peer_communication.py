import os
from datetime import datetime, timezone

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
        app.dependency_overrides[get_current_user] = lambda: {
            "sub": "u1",
            "admin": True,
        }
        comm = get_peer_communicator()
        comm.peers = set()
        comm._client = async_client
        if hasattr(comm, "_telemetry_cache"):
            comm._telemetry_cache.clear()
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


@pytest.mark.asyncio
async def test_peer_register_and_list(client):
    resp = await client.post(
        "/api/v1/peer/register",
        json={"url": "http://test/api/v1/peer/message"},
        headers={"Authorization": "Bearer token"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "registered"
    assert resp.json()["count"] == 1

    again = await client.post(
        "/api/v1/peer/register",
        json={"url": "http://test/api/v1/peer/message"},
        headers={"Authorization": "Bearer token"},
    )
    assert again.status_code == 200
    assert again.json()["status"] == "already registered"
    assert again.json()["count"] == 1

    list_resp = await client.get(
        "/api/v1/peer/list", headers={"Authorization": "Bearer token"}
    )
    assert list_resp.status_code == 200
    assert list_resp.json() == ["http://test/api/v1/peer/message"]

    unreg = await client.post(
        "/api/v1/peer/unregister",
        json={"url": "http://test/api/v1/peer/message"},
        headers={"Authorization": "Bearer token"},
    )
    assert unreg.status_code == 200
    assert unreg.json()["status"] == "unregistered"
    assert unreg.json()["count"] == 0


@pytest.mark.asyncio
async def test_peer_load_endpoint(client):
    communicator = get_peer_communicator()
    original_provider = communicator._load_provider

    async def provider():
        return {
            "scheduler_id": "sched-test",
            "queue_depth": 3,
            "active_workers": 1,
            "concurrency": 2,
            "load_factor": 2.0,
        }

    communicator.register_load_provider(provider)
    try:
        resp = await client.get(
            "/api/v1/peer/load",
            headers={"Authorization": "Bearer token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scheduler_id"] == "sched-test"
        assert data["queue_depth"] == 3
        assert data["load_factor"] == pytest.approx(2.0)
    finally:
        communicator._load_provider = original_provider


@pytest.mark.asyncio
async def test_peer_telemetry_roundtrip(client):
    communicator = get_peer_communicator()
    communicator.ingest_remote_telemetry(
        "http://seed/api/v1/peer/message",
        {
            "scheduler_id": "seed",
            "queue_depth": 1,
            "active_workers": 1,
            "concurrency": 1,
            "load_factor": 0.5,
            "worker_uptime_seconds": 2.0,
            "tasks_processed": 4,
            "tasks_failed": 1,
            "task_failure_rate": 0.25,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "http://seed/api/v1/peer/message",
        },
    )

    payload = {
        "scheduler_id": "remote-1",
        "queue_depth": 2,
        "active_workers": 1,
        "concurrency": 2,
        "load_factor": 0.4,
        "worker_uptime_seconds": 5.0,
        "tasks_processed": 10,
        "tasks_failed": 1,
        "task_failure_rate": 0.1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": "http://remote/api/v1/peer/message",
    }

    resp = await client.post(
        "/api/v1/peer/telemetry",
        json=payload,
        headers={"Authorization": "Bearer token"},
    )
    assert resp.status_code == 202

    telemetry = communicator.get_peer_telemetry()
    assert any(item["scheduler_id"] == "remote-1" for item in telemetry)

    resp_get = await client.get(
        "/api/v1/peer/telemetry",
        headers={"Authorization": "Bearer token"},
    )
    assert resp_get.status_code == 200
    data = resp_get.json()
    assert "telemetry" in data
    identifiers = {entry["scheduler_id"] for entry in data["telemetry"]}
    assert "remote-1" in identifiers
