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
    yield client
    users_db.clear()
    users_db.update(original_users)
    admin_users.clear()
    admin_users.update(original_admins)


def get_token(client, username, password):
    return client.post(
        "/token", data={"username": username, "password": password}
    ).json()["access_token"]


def test_register_and_login(client):
    resp = client.post(
        "/api/v1/user/register", json={"username": "newuser", "password": "newpassword"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "registered"

    token = get_token(client, "newuser", "newpassword")
    assert token


def test_peer_endpoints_require_admin(client, monkeypatch):
    peer_comm = get_peer_communicator()
    peer_comm.peers = set()

    user_token = get_token(client, "user", "passphrase")
    resp = client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403

    admin_token = get_token(client, "admin", "secret")
    resp = client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
