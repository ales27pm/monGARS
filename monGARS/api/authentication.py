from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from pydantic import BaseModel

from monGARS.config import get_settings
from monGARS.core.security import SecurityManager

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class WSTicketResponse(BaseModel):
    ticket: str
    ttl: int


def _ticket_signer() -> TimestampSigner:
    return TimestampSigner(settings.SECRET_KEY)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    sec = SecurityManager(
        secret_key=settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    try:
        payload = sec.verify_token(token)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e


def get_current_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Return the current user if they have admin privileges."""
    if not current_user.get("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin required"
        )
    return current_user


@router.post("/ws/ticket", response_model=WSTicketResponse)
async def issue_ws_ticket(
    current: dict = Depends(get_current_user),
) -> WSTicketResponse:
    uid = current.get("sub")
    if uid is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    signer = _ticket_signer()
    token = signer.sign(uid.encode()).decode()
    return WSTicketResponse(ticket=token, ttl=settings.WS_TICKET_TTL_SECONDS)


def verify_ws_ticket(token: str) -> str:
    signer = _ticket_signer()
    try:
        uid = signer.unsign(token, max_age=settings.WS_TICKET_TTL_SECONDS).decode()
        return uid
    except SignatureExpired as exc:  # pragma: no cover - simple pass-through
        raise HTTPException(status_code=401, detail="Ticket expired") from exc
    except BadSignature as exc:  # pragma: no cover - simple pass-through
        raise HTTPException(status_code=401, detail="Invalid ticket") from exc
