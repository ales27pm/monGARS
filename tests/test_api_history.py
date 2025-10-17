import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app

pytestmark = pytest.mark.usefixtures("ensure_test_users")


@pytest.fixture
def client() -> TestClient:
    """Return a test client with isolated hippocampus state."""
    hippocampus._memory.clear()
    hippocampus._locks.clear()
    with TestClient(app) as client:
        yield client
    hippocampus._memory.clear()
    hippocampus._locks.clear()


@pytest.mark.asyncio
async def test_history_endpoint_returns_records(client: TestClient):
    await hippocampus.store("u1", "q1", "r1")
    await hippocampus.store("u1", "q2", "r2")
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1", "limit": 2},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [item["query"] for item in data] == ["q2", "q1"]


@pytest.mark.asyncio
async def test_history_non_positive_limit_returns_422(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1", "limit": -1},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_history_zero_limit_returns_422(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1", "limit": 0},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_history_empty_user_id_returns_422(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_history_whitespace_user_id_returns_422(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "   "},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_history_limit_capped(client: TestClient):
    for i in range(hippocampus.MAX_HISTORY + 5):
        await hippocampus.store("u1", f"q{i}", f"r{i}")
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1", "limit": hippocampus.MAX_HISTORY + 10},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert len(resp.json()) == hippocampus.MAX_HISTORY


@pytest.mark.asyncio
async def test_history_forbidden_other_user(client: TestClient):
    await hippocampus.store("u1", "q1", "r1")
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u2"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_history_no_token_returns_401(client: TestClient):
    resp = client.get("/api/v1/conversation/history", params={"user_id": "u1"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_history_invalid_token_returns_401(client: TestClient):
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1"},
        headers={"Authorization": "Bearer bad"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_history_no_records_returns_empty_list(client: TestClient):
    token = client.post("/token", data={"username": "u2", "password": "y"}).json()[
        "access_token"
    ]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u2"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json() == []
