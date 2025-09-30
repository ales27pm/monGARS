"""WebSocket ticket issuance and verification helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from monGARS.api.authentication import get_current_user
from monGARS.api.ticket_signer import BadSignature, SignatureExpired, TicketSigner
from monGARS.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
router = APIRouter(prefix="/api/v1/auth/ws", tags=["ws-ticket"])


class WSTicketResponse(BaseModel):
    ticket: str
    ttl: int


def _ticket_signer() -> TicketSigner:
    return TicketSigner(settings.SECRET_KEY)


@router.post("/ticket", response_model=WSTicketResponse)
async def issue_ws_ticket(
    current: Mapping[str, Any] = Depends(get_current_user),
) -> WSTicketResponse:
    uid = current.get("sub")
    if not isinstance(uid, str) or not uid.strip():
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload: 'sub' must be a non-empty string",
        )
    signer = _ticket_signer()
    token = signer.sign(uid.encode("utf-8"))
    return WSTicketResponse(ticket=token, ttl=settings.WS_TICKET_TTL_SECONDS)


def verify_ws_ticket(token: str) -> str:
    signer = _ticket_signer()
    try:
        return signer.unsign(token, max_age=settings.WS_TICKET_TTL_SECONDS).decode(
            "utf-8"
        )
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
