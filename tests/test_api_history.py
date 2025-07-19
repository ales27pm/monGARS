import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app


@pytest.mark.asyncio
async def test_history_endpoint_returns_records():
    await hippocampus.store("u1", "q1", "r1")
    client = TestClient(app)
    token_resp = client.post("/token", data={"username": "u1", "password": "x"})
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]
    resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["query"] == "q1"
