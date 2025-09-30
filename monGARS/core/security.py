import logging
import re
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import bleach
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from jose import JWTError, jwt
from passlib.context import CryptContext

from monGARS.config import Settings, ensure_secret_key, get_settings

log = logging.getLogger(__name__)


class SecurityManager:
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: Optional[str] = None,
        settings: Optional[Settings] = None,
        private_key: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> None:
        base_settings = settings or get_settings()
        configured_algorithm = (algorithm or base_settings.JWT_ALGORITHM).upper()

        resolved_secret = secret_key or base_settings.SECRET_KEY
        resolved_private = private_key or getattr(
            base_settings, "JWT_PRIVATE_KEY", None
        )
        resolved_public = public_key or getattr(base_settings, "JWT_PUBLIC_KEY", None)

        self.algorithm = configured_algorithm
        self._is_asymmetric = configured_algorithm.startswith("RS")

        if self._is_asymmetric:
            if resolved_private is None and resolved_secret is not None:
                resolved_private = resolved_secret
            if not resolved_private or not resolved_public:
                raise ValueError(
                    "RSA JWT algorithms require both private and public keys."
                )
            self.secret_key = resolved_private
            self.private_key = resolved_private
            self.public_key = resolved_public
            self._settings = base_settings
        else:
            if secret_key and not base_settings.SECRET_KEY:
                base_settings = base_settings.model_copy(
                    update={"SECRET_KEY": secret_key}
                )
            resolved_secret = secret_key or base_settings.SECRET_KEY
            if not resolved_secret:
                base_settings, _ = ensure_secret_key(
                    base_settings,
                    log_message=(
                        "SECRET_KEY missing during SecurityManager init; using ephemeral key."
                    ),
                )
                resolved_secret = base_settings.SECRET_KEY
            elif secret_key and base_settings.SECRET_KEY != secret_key:
                base_settings = base_settings.model_copy(
                    update={"SECRET_KEY": secret_key}
                )

            self._settings = base_settings
            self.secret_key = resolved_secret
            self.private_key = None
            self.public_key = None

        self._signing_key = self.private_key or self.secret_key
        self._verification_key = self.public_key or self.secret_key
        # ``passlib`` defaults to bcrypt, but that backend is optional and
        # raises a ``ValueError`` when the optimized extension is absent.  This
        # broke our test environment, so we switch to ``pbkdf2_sha256`` which is
        # implemented in pure Python while still providing a strong password
        # hashing story.  We still accept legacy bcrypt hashes when the
        # dependency is available so existing credentials remain valid.
        schemes = ["pbkdf2_sha256"]
        context_kwargs = {
            "schemes": schemes,
            "deprecated": "auto",
            "default": "pbkdf2_sha256",
            "pbkdf2_sha256__rounds": 390000,
        }
        try:
            import importlib

            importlib.import_module("bcrypt")
        except ModuleNotFoundError:
            pass
        else:
            schemes.append("bcrypt")
        self.pwd_context = CryptContext(**context_kwargs)

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (
            expires_delta
            or timedelta(minutes=self._settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire.timestamp()})
        if not self._signing_key:
            raise ValueError("Signing key is not configured")
        return jwt.encode(to_encode, self._signing_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        try:
            if not self._verification_key:
                raise ValueError("Verification key is not configured")
            payload = jwt.decode(
                token, self._verification_key, algorithms=[self.algorithm]
            )
            if datetime.fromtimestamp(payload["exp"], tz=timezone.utc) <= datetime.now(
                timezone.utc
            ):
                raise ValueError("Token expired")
            return payload
        except JWTError as exc:
            raise ValueError(f"Token verification failed: {exc}") from exc

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)


def _get_fernet(key: Union[str, bytes, None] = None) -> Fernet:
    """Return a Fernet instance derived from the provided key."""
    settings = get_settings()
    raw_key = (
        key.encode()
        if isinstance(key, str)
        else key if key is not None else settings.SECRET_KEY.encode()
    )
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"monGARS-fernet-salt",
        iterations=390000,
        backend=default_backend(),
    )
    derived = kdf.derive(raw_key)
    return Fernet(urlsafe_b64encode(derived))


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
