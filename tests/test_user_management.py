"""End-to-end tests for user registration and authentication flows."""

import asyncio

import pytest
import pytest_asyncio
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from monGARS.api.authentication import authenticate_user, ensure_bootstrap_users
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
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as async_client:
            yield async_client
    await reset_database()


@pytest_asyncio.fixture
async def empty_client() -> AsyncClient:
    await reset_database()
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as async_client:
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
    data = resp.json()
    assert data["status"] == "registered"
    assert data["is_admin"] is False

    token = await get_token(client, "newuser", "newpassword")
    payload = sec_manager.verify_token(token)
    assert payload["sub"] == "newuser"
    assert payload["admin"] is False


@pytest.mark.asyncio
async def test_register_admin_when_none_exists(
    empty_client: AsyncClient,
) -> None:
    resp = await empty_client.post(
        "/api/v1/user/register/admin",
        json={"username": "founder", "password": "founderpw"},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body == {"status": "registered", "is_admin": True}

    token = await get_token(empty_client, "founder", "founderpw")
    payload = sec_manager.verify_token(token)
    assert payload["sub"] == "founder"
    assert payload["admin"] is True

    resp_conflict = await empty_client.post(
        "/api/v1/user/register/admin",
        json={"username": "second", "password": "secondpw"},
    )
    assert resp_conflict.status_code == status.HTTP_403_FORBIDDEN
    assert resp_conflict.json()["detail"] == "Admin already exists"

    resp_user = await empty_client.post(
        "/api/v1/user/register",
        json={"username": "member", "password": "memberpw"},
    )
    assert resp_user.status_code == status.HTTP_200_OK
    member_body = resp_user.json()
    assert member_body["status"] == "registered"
    assert member_body["is_admin"] is False


@pytest.mark.asyncio
async def test_user_list_requires_admin_and_returns_usernames(
    client: AsyncClient,
) -> None:
    first_registration = await client.post(
        "/api/v1/user/register",
        json={"username": "firstuser", "password": "firstpassword"},
    )
    assert first_registration.status_code == status.HTTP_200_OK

    second_registration = await client.post(
        "/api/v1/user/register",
        json={"username": "seconduser", "password": "secondpassword"},
    )
    assert second_registration.status_code == status.HTTP_200_OK

    admin_token = await get_token(client, "admin", "secret")
    user_token = await get_token(client, "user", "passphrase")

    unauthenticated = await client.get("/api/v1/user/list")
    assert unauthenticated.status_code == status.HTTP_401_UNAUTHORIZED

    forbidden = await client.get(
        "/api/v1/user/list",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert forbidden.status_code == status.HTTP_403_FORBIDDEN

    response = await client.get(
        "/api/v1/user/list",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["users"]
    assert {"firstuser", "seconduser"}.issubset(set(body["users"]))


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
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    resp = await client.post(
        "/api/v1/user/register",
        json={"username": "invalid user!", "password": "password123"},
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.asyncio
async def test_register_short_password(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/user/register",
        json={"username": "shortpwuser", "password": "pw"},
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


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
async def test_change_password_success(client: AsyncClient) -> None:
    token = await get_token(client, "user", "passphrase")

    response = await client.post(
        "/api/v1/user/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "old_password": "passphrase",
            "new_password": "newpassphrase",
        },
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "changed"}

    old_login = await client.post(
        "/token", data={"username": "user", "password": "passphrase"}
    )
    assert old_login.status_code == status.HTTP_401_UNAUTHORIZED

    new_login = await client.post(
        "/token", data={"username": "user", "password": "newpassphrase"}
    )
    assert new_login.status_code == status.HTTP_200_OK
    new_payload = sec_manager.verify_token(new_login.json()["access_token"])
    assert new_payload["sub"] == "user"


@pytest.mark.asyncio
async def test_change_password_wrong_old_password(client: AsyncClient) -> None:
    token = await get_token(client, "user", "passphrase")

    response = await client.post(
        "/api/v1/user/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "old_password": "incorrectpw",
            "new_password": "replacement",
        },
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == "Incorrect password"

    still_valid = await client.post(
        "/token", data={"username": "user", "password": "passphrase"}
    )
    assert still_valid.status_code == status.HTTP_200_OK


def test_login_invalid_credentials_returns_401_detail() -> None:
    asyncio.run(reset_database())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/token",
                data={"username": "wronguser", "password": "wrongpass"},
            )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json() == {"detail": "Invalid credentials"}
    finally:
        asyncio.run(reset_database())


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


@pytest.mark.asyncio
async def test_authenticate_user_requires_persisted_account() -> None:
    await reset_database()
    repo = get_persistence_repository()

    with pytest.raises(HTTPException) as exc_info:
        await authenticate_user(repo, "ghost", "pw", sec_manager)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    await reset_database()


@pytest.mark.asyncio
async def test_authenticate_user_incorrect_password_fails() -> None:
    await reset_database()
    repo = get_persistence_repository()

    username = "testuser"
    password_hash = sec_manager.get_password_hash("correctpassword")
    await repo.create_user(username, password_hash)

    with pytest.raises(HTTPException) as exc_info:
        await authenticate_user(repo, username, "wrongpassword", sec_manager)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    await reset_database()


@pytest.mark.asyncio
async def test_bootstrap_users_create_demo_accounts() -> None:
    await reset_database()
    repo = get_persistence_repository()
    demo_defaults = {
        "demo": {
            "password_hash": sec_manager.get_password_hash("demo-pass"),
            "is_admin": True,
        }
    }
    await ensure_bootstrap_users(repo, demo_defaults)
    user = await repo.get_user_by_username("demo")
    assert user is not None
    assert user.is_admin is True

    await ensure_bootstrap_users(repo, demo_defaults)
    second = await repo.get_user_by_username("demo")
    assert second is not None
    await reset_database()


@pytest.mark.asyncio
async def test_startup_does_not_create_demo_accounts(client: AsyncClient) -> None:
    ready = await client.get("/ready")
    assert ready.status_code == status.HTTP_200_OK
    repo = get_persistence_repository()
    assert await repo.get_user_by_username("u1") is None
