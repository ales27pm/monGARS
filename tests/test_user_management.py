import asyncio
import os

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import get_peer_communicator, get_persistence_repository
from monGARS.api.web_api import app, sec_manager
from monGARS.init_db import reset_database


@pytest.fixture
def client():
    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    repo = get_persistence_repository()

    async def setup() -> None:
        await reset_database()
        await repo.create_user(
            "admin", sec_manager.get_password_hash("secret"), is_admin=True
        )
        await repo.create_user(
            "user", sec_manager.get_password_hash("passphrase"), is_admin=False
        )

    loop.run_until_complete(setup())
    client = TestClient(app)

    try:
        yield client
    finally:
        client.close()

        async def teardown() -> None:
            await reset_database()

        loop.run_until_complete(teardown())
        loop.close()
        if previous_loop is not None and not previous_loop.is_closed():
            asyncio.set_event_loop(previous_loop)
        else:
            asyncio.set_event_loop(None)


def get_token(client, username, password):
    response = client.post("/token", data={"username": username, "password": password})
    data = response.json()
    if "access_token" in data:
        return data["access_token"]
    print(f"Failed to get access_token for {username}: {data}")
    return None


def test_register_and_login(client):
    resp = client.post(
        "/api/v1/user/register", json={"username": "newuser", "password": "newpassword"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "registered"

    token = get_token(client, "newuser", "newpassword")
    assert token


def test_register_existing_username(client):
    resp1 = client.post(
        "/api/v1/user/register", json={"username": "dup", "password": "password123"}
    )
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "registered"
    resp2 = client.post(
        "/api/v1/user/register", json={"username": "dup", "password": "password123"}
    )
    assert resp2.status_code in (400, 409)


def test_register_invalid_username(client):
    resp = client.post(
        "/api/v1/user/register", json={"username": "", "password": "password123"}
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/v1/user/register",
        json={"username": "invalid user!", "password": "password123"},
    )
    assert resp.status_code == 422


def test_register_short_password(client):
    resp = client.post(
        "/api/v1/user/register", json={"username": "shortpwuser", "password": "pw"}
    )
    assert resp.status_code == 422


def test_login_incorrect_credentials(client):
    client.post(
        "/api/v1/user/register",
        json={"username": "loginuser", "password": "correctpassword"},
    )
    resp = client.post(
        "/token", data={"username": "loginuser", "password": "wrongpassword"}
    )
    assert resp.status_code == 401
    resp = client.post("/token", data={"username": "nosuchuser", "password": "any"})
    assert resp.status_code == 401


def test_peer_endpoints_require_admin(client, monkeypatch):
    peer_comm = get_peer_communicator()
    peer_comm.peers = set()

    user_token = get_token(client, "user", "passphrase")
    admin_token = get_token(client, "admin", "secret")
    invalid_token = "invalid.token.value"

    # /peer/register
    resp = client.post("/api/v1/peer/register", json={"url": "http://x"})
    assert resp.status_code == 401

    resp = client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {invalid_token}"},
    )
    assert resp.status_code in (401, 403)

    resp = client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403

    resp = client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200

    # /peer/unregister
    resp = client.post("/api/v1/peer/unregister", json={"url": "http://x"})
    assert resp.status_code == 401

    resp = client.post(
        "/api/v1/peer/unregister",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {invalid_token}"},
    )
    assert resp.status_code in (401, 403)

    resp = client.post(
        "/api/v1/peer/unregister",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403

    resp = client.post(
        "/api/v1/peer/unregister",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code in (200, 404)

    # /peer/list
    resp = client.get("/api/v1/peer/list")
    assert resp.status_code == 401

    resp = client.get(
        "/api/v1/peer/list",
        headers={"Authorization": f"Bearer {invalid_token}"},
    )
    assert resp.status_code in (401, 403)

    resp = client.get(
        "/api/v1/peer/list",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403

    resp = client.get(
        "/api/v1/peer/list",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
