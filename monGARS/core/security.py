import re
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import bleach
from cryptography.fernet import Fernet, InvalidToken
from jose import JWTError, jwt
from passlib.context import CryptContext

from monGARS.config import get_settings

settings = get_settings()


class SecurityManager:
    def __init__(
        self,
        secret_key: str = settings.SECRET_KEY,
        algorithm: str = settings.JWT_ALGORITHM,
    ) -> None:
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (
            expires_delta
            if expires_delta
            else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire.timestamp()})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if datetime.fromtimestamp(payload["exp"], tz=timezone.utc) <= datetime.now(
                timezone.utc
            ):
                raise ValueError("Token expired")
            return payload
        except JWTError as exc:
            raise ValueError(f"Token verification failed: {exc}")

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)


def _get_fernet(key: Union[str, bytes, None] = None) -> Fernet:
    key_bytes = (
        key.encode()
        if isinstance(key, str)
        else key if key is not None else settings.SECRET_KEY.encode()
    )
    if len(key_bytes) != 32:
        key_bytes = urlsafe_b64encode(key_bytes[:32])
    return Fernet(key_bytes)


def encrypt_token(token: str, key: Union[str, bytes, None] = None) -> str:
    f = _get_fernet(key)
    return f.encrypt(token.encode()).decode()


def decrypt_token(token: str, key: Union[str, bytes, None] = None) -> str:
    f = _get_fernet(key)
    try:
        return f.decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Invalid encrypted token") from exc


def validate_user_input(data: dict) -> dict:
    for key, value in data.items():
        if isinstance(value, str):
            clean_value = bleach.clean(value, strip=True)
            data[key] = clean_value
    if not data.get("user_id") or not data.get("query"):
        raise ValueError("Missing required fields: user_id and query.")
    return data


class Credentials:
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
