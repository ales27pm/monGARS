"""WebSocket ticket issuance and verification helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from monGARS.api.authentication import get_current_user
from monGARS.api.ticket_signer import BadSignature, SignatureExpired, TicketSigner
from monGARS.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth/ws", tags=["ws-ticket"])


class WSTicketResponse(BaseModel):
    ticket: str
    ttl: int


def _ticket_signer() -> TicketSigner:
    return TicketSigner(get_settings().SECRET_KEY)


def _allowed_ws_origins() -> set[str]:
    settings = get_settings()
    return {str(origin).rstrip("/") for origin in settings.WS_ALLOWED_ORIGINS}


def _cors_headers_for_origin(origin: str) -> dict[str, str]:
    return {
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Origin": origin,
        "Vary": "Origin",
    }


@router.options("/ticket", include_in_schema=False)
async def issue_ws_ticket_options(request: Request) -> Response:
    origin = (request.headers.get("origin") or "").rstrip("/")
    allowed = _allowed_ws_origins()
    if allowed and origin not in allowed:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    requested_headers = request.headers.get("access-control-request-headers")
    headers = _cors_headers_for_origin(origin)
    headers["Access-Control-Allow-Headers"] = (
        requested_headers or headers["Access-Control-Allow-Headers"]
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=headers)


@router.post("/ticket", response_model=WSTicketResponse)
async def issue_ws_ticket(
    request: Request,
    response: Response,
    current: Mapping[str, Any] = Depends(get_current_user),
) -> WSTicketResponse:
    origin = (request.headers.get("origin") or "").rstrip("/")
    allowed = _allowed_ws_origins()
    if origin and (not allowed or origin in allowed):
        for key, value in _cors_headers_for_origin(origin).items():
            response.headers[key] = value
    uid = current.get("sub")
    if not isinstance(uid, str) or not uid.strip():
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload: 'sub' must be a non-empty string",
        )
    signer = _ticket_signer()
    token = signer.sign(uid.encode("utf-8"))
    return WSTicketResponse(ticket=token, ttl=get_settings().WS_TICKET_TTL_SECONDS)


def verify_ws_ticket(token: str) -> str:
    signer = _ticket_signer()
    try:
        return signer.unsign(
            token, max_age=get_settings().WS_TICKET_TTL_SECONDS
        ).decode("utf-8")
    except SignatureExpired as exc:  # pragma: no cover - simple pass-through
        logger.warning(
            "Expired WebSocket ticket verification attempt",
            extra={"reason": "expired"},
        )
        raise HTTPException(status_code=401, detail="Ticket expired") from exc
    except BadSignature as exc:  # pragma: no cover - simple pass-through
        logger.warning(
            "Invalid WebSocket ticket verification attempt",
            extra={"reason": "invalid_signature"},
        )
        raise HTTPException(status_code=401, detail="Invalid ticket") from exc
