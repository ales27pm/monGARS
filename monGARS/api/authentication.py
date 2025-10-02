"""Authentication helpers and dependencies for the FastAPI layer."""

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from monGARS.config import get_settings
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.security import SecurityManager
from monGARS.init_db import UserAccount

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = logging.getLogger(__name__)


async def authenticate_user(
    repo: PersistenceRepository,
    username: str,
    password: str,
    sec_manager: SecurityManager,
    defaults: Mapping[str, Mapping[str, Any]],
) -> UserAccount:
    """Authenticate ``username`` using persisted accounts or default fallbacks."""

    user = await repo.get_user_by_username(username)
    if user and sec_manager.verify_password(password, user.password_hash):
        return user

    default_user = defaults.get(username)
    if default_user and sec_manager.verify_password(
        password, default_user["password_hash"]
    ):
        try:
            return await repo.create_user(
                username,
                default_user["password_hash"],
                is_admin=default_user["is_admin"],
            )
        except ValueError as exc:
            logger.debug(
                "auth.default_user_race",
                extra={"username": username},
                exc_info=exc,
            )
            user = await repo.get_user_by_username(username)
            if user:
                return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    sec = SecurityManager(
        secret_key=settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    try:
        payload = sec.verify_token(token)
    except Exception as exc:  # pragma: no cover - FastAPI handles response
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc
    subject = None
    if isinstance(payload, Mapping):
        subject = payload.get("sub")
    if subject is None or (isinstance(subject, str) and not subject):
        logger.warning(
            "auth.invalid_token_missing_sub",
            extra={"payload_type": type(payload).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject",
        )
    return payload


def get_current_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Return the current user if they have admin privileges."""
    if not current_user.get("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin required"
        )
    return current_user
