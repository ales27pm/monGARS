"""Custom exceptions raised by the monGARS Python SDK."""

from __future__ import annotations

from typing import Any


class SDKError(Exception):
    """Base class for all SDK errors."""


class APIError(SDKError):
    """Raised when the HTTP API responds with an error status."""

    def __init__(
        self,
        status_code: int,
        detail: str | None = None,
        *,
        payload: Any | None = None,
    ) -> None:
        message = detail or f"API request failed with status {status_code}"
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or message
        self.payload = payload


class AuthenticationError(APIError):
    """Raised when authentication fails or a token is missing."""

    def __init__(
        self,
        status_code: int = 401,
        detail: str | None = None,
        *,
        payload: Any | None = None,
    ) -> None:
        super().__init__(
            status_code, detail or "Authentication failed", payload=payload
        )
