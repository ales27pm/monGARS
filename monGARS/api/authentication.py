"""Authentication helpers and dependencies for the FastAPI layer."""

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

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
) -> UserAccount:
    """Authenticate ``username`` using persisted accounts only."""

    user = await repo.get_user_by_username(username)
    if user and sec_manager.verify_password(password, user.password_hash):
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


async def ensure_bootstrap_users(
    repo: PersistenceRepository,
    defaults: Mapping[str, Mapping[str, Any]],
) -> None:
    """Persist default demo users without relying on in-memory fallbacks."""

    for username, config in defaults.items():
        try:
            existing = await repo.get_user_by_username(username)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "auth.bootstrap.lookup_failed",
                extra={"username": username},
                exc_info=exc,
            )
            continue

        if existing:
            continue

        password_hash = config.get("password_hash")
        if not isinstance(password_hash, str) or not password_hash:
            logger.warning(
                "auth.bootstrap.invalid_password_hash",
                extra={"username": username},
            )
            continue

        is_admin = bool(config.get("is_admin", False))

        try:
            await repo.create_user(
                username,
                password_hash,
                is_admin=is_admin,
            )
        except ValueError:
            logger.debug(
                "auth.bootstrap.user_exists",
                extra={"username": username},
            )
        except Exception as exc:  # pragma: no cover - unexpected failure
            logger.warning(
                "auth.bootstrap.create_failed",
                extra={"username": username},
                exc_info=exc,
            )


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    sec = SecurityManager(
        secret_key=settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    try:
        payload = sec.verify_token(token)
    except Exception as exc:  # pragma: no cover - FastAPI handles response
        missing_subject = False
        try:
            claims = jwt.get_unverified_claims(token)
        except JWTError:
            claims = {}
        if isinstance(claims, Mapping):
            subject = claims.get("sub")
            missing_subject = subject is None or (
                isinstance(subject, str) and not subject
            )
        if missing_subject:
            logger.warning(
                "auth.invalid_token_missing_sub_unverified",
                extra={"payload_type": type(claims).__name__},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            ) from exc
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
