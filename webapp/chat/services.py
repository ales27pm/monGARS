from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

import httpx

logger = logging.getLogger(__name__)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


class ChatResult(TypedDict, total=False):
    """Result returned by :func:`post_chat_message`."""

    response: str
    confidence: float
    processing_time: float
    error: str


async def fetch_history(user_id: str, token: str) -> Any:
    """Return recent conversation history for ``user_id``.

    The FastAPI endpoint responds with a JSON list sorted from newest to oldest.
    When the request fails, a mapping containing an ``error`` key is returned so
    callers can surface a user-friendly message.
    """

    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{FASTAPI_URL}/api/v1/conversation/history",
                params={"user_id": user_id},
                headers=headers,
            )
    except httpx.HTTPError as exc:  # pragma: no cover - network failure
        logger.error("fetch_history failed", exc_info=exc)
        return {"error": f"Impossible de se connecter: {exc}"}

    if resp.status_code == 200:
        try:
            return resp.json()
        except ValueError as exc:
            logger.error("fetch_history invalid JSON", exc_info=exc)
            return {"error": "Réponse d'historique invalide"}
    detail = _extract_error(resp)
    return {"error": detail or f"Erreur: {resp.status_code}"}


async def post_chat_message(user_id: str, token: str, message: str) -> ChatResult:
    """Send ``message`` to the FastAPI chat endpoint and return the response."""

    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": message}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{FASTAPI_URL}/api/v1/conversation/chat",
                json=payload,
                headers=headers,
            )
    except httpx.HTTPError as exc:  # pragma: no cover - network failure
        logger.error("post_chat_message failed", exc_info=exc)
        return {"error": f"Impossible d'envoyer le message: {exc}"}

    if resp.status_code != 200:
        detail = _extract_error(resp)
        return {"error": detail or f"Erreur: {resp.status_code}"}

    try:
        body: dict[str, Any] = resp.json()
    except ValueError as exc:
        logger.error("post_chat_message invalid JSON", exc_info=exc)
        return {"error": "Réponse inattendue du service de chat"}

    result: ChatResult = {
        "response": str(body.get("response", "")),
    }
    if "confidence" in body:
        try:
            result["confidence"] = float(body["confidence"])
        except (TypeError, ValueError):
            logger.debug("Confidence value not convertible", exc_info=True)
    if "processing_time" in body:
        try:
            result["processing_time"] = float(body["processing_time"])
        except (TypeError, ValueError):
            logger.debug("Processing time value not convertible", exc_info=True)
    return result


async def authenticate_user(username: str, password: str) -> str | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{FASTAPI_URL}/token",
                data={"username": username, "password": password},
            )
    except httpx.HTTPError as exc:  # pragma: no cover - network failure
        logger.error("authenticate_user failed", exc_info=exc)
        return None

    if resp.status_code == 200:
        try:
            return resp.json().get("access_token")
        except ValueError as exc:
            logger.error("authenticate_user invalid JSON", exc_info=exc)
            return None
    return None


def _extract_error(resp: httpx.Response) -> str | None:
    """Return a human-friendly error message from an HTTP response."""

    try:
        payload = resp.json()
    except ValueError:
        return None
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if isinstance(detail, str):
            return detail
    return None
