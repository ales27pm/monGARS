from datetime import datetime, timezone

from fastapi import Depends, HTTPException
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
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
