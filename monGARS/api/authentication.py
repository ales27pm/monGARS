from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from monGARS.config import get_settings
from monGARS.core.security import SecurityManager

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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
