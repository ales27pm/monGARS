"""Authentication helpers and dependencies for the FastAPI layer."""

import inspect
import logging
from collections.abc import Mapping
from typing import Any, Optional, Tuple

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
        if await _user_exists(repo, username):
            continue

        parsed = _parse_config(username, config)
        if parsed is None:
            continue
        password_hash, is_admin = parsed

        await _create_user_safely(repo, username, password_hash, is_admin)


async def _user_exists(repo: PersistenceRepository, username: str) -> bool:
    """Return ``True`` when ``username`` is already present in the repository."""

    try:
        return await repo.get_user_by_username(username) is not None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "auth.bootstrap.lookup_failed",
            extra={"username": username},
            exc_info=exc,
        )
        return False


def _parse_config(
    username: str, config: Mapping[str, Any]
) -> Optional[Tuple[str, bool]]:
    """Validate bootstrap configuration for ``username``.

    Returns a tuple of ``(password_hash, is_admin)`` when configuration is valid,
    otherwise ``None``.
    """

    password_hash = config.get("password_hash")
    if not isinstance(password_hash, str) or not password_hash:
        logger.warning(
            "auth.bootstrap.invalid_password_hash",
            extra={"username": username},
        )
        return None

    is_admin = config.get("is_admin", False)
    if not isinstance(is_admin, bool):
        logger.warning(
            "auth.bootstrap.invalid_is_admin_type",
            extra={"username": username, "is_admin": is_admin},
        )
        return None

    return password_hash, is_admin


async def _create_user_safely(
    repo: PersistenceRepository,
    username: str,
    password_hash: str,
    is_admin: bool,
) -> None:
    """Create ``username`` while tolerating races and transient errors."""

    try:
        create_sig = inspect.signature(repo.create_user_atomic)
        if "is_admin" in create_sig.parameters:
            await repo.create_user_atomic(
                username,
                password_hash,
                is_admin=is_admin,
            )
        else:
            await repo.create_user(
                username,
                password_hash,
                is_admin=is_admin,
            )
    except ValueError:
        logger.debug(
            "auth.bootstrap.user_exists_or_race",
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
