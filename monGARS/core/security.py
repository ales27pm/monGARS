import re
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from monGARS.config import get_settings
import bleach

settings = get_settings()

class SecurityManager:
    def __init__(self, secret_key: str = settings.SECRET_KEY, algorithm: str = settings.JWT_ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire.timestamp()})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if datetime.fromtimestamp(payload['exp'], tz=timezone.utc) <= datetime.now(timezone.utc):
                raise ValueError("Token expired")
            return payload
        except JWTError as e:
            raise ValueError(f"Token verification failed: {e}")

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

def validate_user_input(data: dict) -> dict:
    for key, value in data.items():
        if isinstance(value, str):
            clean_value = bleach.clean(value, strip=True)
            data[key] = clean_value
    if not data.get("user_id") or not data.get("query"):
        raise ValueError("Missing required fields: user_id and query.")
    return data

class Credentials:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password