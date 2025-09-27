import os

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import get_peer_communicator
from monGARS.api.web_api import admin_users, app, sec_manager, users_db


@pytest.fixture
def client():
    original_users = users_db.copy()
    original_admins = set(admin_users)
    users_db.clear()
    users_db.update(
        {
            "admin": sec_manager.get_password_hash("secret"),
            "user": sec_manager.get_password_hash("passphrase"),
        }
    )
    admin_users.clear()
    admin_users.add("admin")
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        users_db.clear()
        users_db.update(original_users)
        admin_users.clear()
        admin_users.update(original_admins)


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
