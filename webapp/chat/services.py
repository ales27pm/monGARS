import logging
import os

import httpx

logger = logging.getLogger(__name__)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


async def fetch_history(user_id: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{FASTAPI_URL}/api/v1/conversation/history",
                params={"user_id": user_id},
                headers=headers,
            )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Erreur: {resp.status_code}"}
    except Exception as e:  # pragma: no cover - network failure
        logger.error("fetch_history failed: %s", e)
        return {"error": f"Impossible de se connecter: {e}"}


async def authenticate_user(username: str, password: str) -> str | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{FASTAPI_URL}/token",
                data={"username": username, "password": password},
            )
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except Exception as e:  # pragma: no cover - network failure
        logger.error("authenticate_user failed: %s", e)
    return None
