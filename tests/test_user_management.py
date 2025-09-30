"""End-to-end tests for user registration and authentication flows."""

import pytest
import pytest_asyncio
from fastapi import status
from httpx import ASGITransport, AsyncClient

from monGARS.api.dependencies import get_peer_communicator, get_persistence_repository
from monGARS.api.web_api import app, sec_manager
from monGARS.init_db import reset_database


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    await reset_database()
    repo = get_persistence_repository()
    await repo.create_user(
        "admin", sec_manager.get_password_hash("secret"), is_admin=True
    )
    await repo.create_user(
        "user", sec_manager.get_password_hash("passphrase"), is_admin=False
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client

    await reset_database()


async def get_token(client: AsyncClient, username: str, password: str) -> str:
    response = await client.post(
        "/token", data={"username": username, "password": password}
    )
    assert response.status_code == 200, response.json()
    data = response.json()
    assert "access_token" in data
    return data["access_token"]


@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/user/register",
        json={"username": "newuser", "password": "newpassword"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "registered"

    token = await get_token(client, "newuser", "newpassword")
    payload = sec_manager.verify_token(token)
    assert payload["sub"] == "newuser"
    assert payload["admin"] is False


@pytest.mark.asyncio
async def test_register_existing_username_returns_conflict(
    client: AsyncClient,
) -> None:
    resp_first = await client.post(
        "/api/v1/user/register",
        json={"username": "dup", "password": "password123"},
    )
    assert resp_first.status_code == 200
    resp_second = await client.post(
        "/api/v1/user/register",
        json={"username": "dup", "password": "password123"},
    )
    assert resp_second.status_code == status.HTTP_409_CONFLICT
    assert resp_second.json()["detail"] == "Username already exists"


@pytest.mark.asyncio
async def test_register_invalid_username(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/user/register", json={"username": "", "password": "password123"}
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    resp = await client.post(
        "/api/v1/user/register",
        json={"username": "invalid user!", "password": "password123"},
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_register_short_password(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/user/register",
        json={"username": "shortpwuser", "password": "pw"},
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_login_incorrect_credentials(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/user/register",
        json={"username": "loginuser", "password": "correctpassword"},
    )
    resp = await client.post(
        "/token", data={"username": "loginuser", "password": "wrongpassword"}
    )
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    resp = await client.post(
        "/token", data={"username": "nosuchuser", "password": "any"}
    )
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_admin_claim_reflects_admin_status(client: AsyncClient) -> None:
    admin_token = await get_token(client, "admin", "secret")
    admin_payload = sec_manager.verify_token(admin_token)
    assert admin_payload["admin"] is True

    user_token = await get_token(client, "user", "passphrase")
    user_payload = sec_manager.verify_token(user_token)
    assert user_payload["admin"] is False


@pytest.mark.asyncio
async def test_peer_endpoints_require_admin(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer_comm = get_peer_communicator()
    peer_comm.peers = set()

    user_token = await get_token(client, "user", "passphrase")
    admin_token = await get_token(client, "admin", "secret")
    invalid_token = "invalid.token.value"

    resp = await client.post("/api/v1/peer/register", json={"url": "http://x"})
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED

    resp = await client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {invalid_token}"},
    )
    assert resp.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    resp = await client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == status.HTTP_403_FORBIDDEN

    resp = await client.post(
        "/api/v1/peer/register",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == status.HTTP_200_OK

    resp = await client.post("/api/v1/peer/unregister", json={"url": "http://x"})
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED

    resp = await client.post(
        "/api/v1/peer/unregister",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {invalid_token}"},
    )
    assert resp.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    resp = await client.post(
        "/api/v1/peer/unregister",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == status.HTTP_403_FORBIDDEN

    resp = await client.post(
        "/api/v1/peer/unregister",
        json={"url": "http://x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code in (status.HTTP_200_OK, status.HTTP_404_NOT_FOUND)

    resp = await client.get("/api/v1/peer/list")
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED

    resp = await client.get(
        "/api/v1/peer/list",
        headers={"Authorization": f"Bearer {invalid_token}"},
    )
    assert resp.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    resp = await client.get(
        "/api/v1/peer/list",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == status.HTTP_403_FORBIDDEN

    resp = await client.get(
        "/api/v1/peer/list",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == status.HTTP_200_OK
