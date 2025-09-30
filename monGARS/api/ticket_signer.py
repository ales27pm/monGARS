"""Internal timestamped token signer used for WebSocket tickets.

This module replaces the external :mod:`itsdangerous` dependency with a small
implementation that signs payloads using HMAC SHA-256 and embeds the creation
timestamp inside the token.  Only the limited surface that the API layer
requires is implemented: signing, verifying with a time-to-live check, and the
``BadSignature`` / ``SignatureExpired`` exception hierarchy that the FastAPI
routes rely on.

The functionality lives in the API package so it can be imported without
introducing additional dependencies at application start-up.  The
``TicketSigner`` is deterministic and testable; callers may provide a custom
``clock`` callable to control the notion of ``now`` in unit tests.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from collections.abc import Callable

__all__ = [
    "BadSignature",
    "SignatureExpired",
    "TicketSigner",
]


class BadSignature(Exception):
    """Raised when a token fails HMAC validation or cannot be decoded."""


class SignatureExpired(BadSignature):
    """Raised when a token is structurally valid but past its TTL."""


class TicketSigner:
    """Sign and verify opaque tokens that embed their creation timestamp.

    Parameters
    ----------
    secret_key:
        Secret value used as the HMAC key.  The same key must be supplied when
        verifying a token.
    salt:
        Optional namespace value.  Changing the salt invalidates all existing
        tokens even if the secret key stays the same.
    digestmod:
        Hashlib digest algorithm name.  Defaults to ``"sha256"``.
    clock:
        Optional callable returning the current UNIX timestamp as a float.  This
        is primarily intended for unit tests.
    """

    _separator = "."

    def __init__(
        self,
        secret_key: str,
        *,
        salt: str = "monGARS.ws_ticket",
        digestmod: str = "sha256",
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not secret_key:
            msg = "secret_key must be a non-empty string"
            raise ValueError(msg)
        if not salt:
            msg = "salt must be a non-empty string"
            raise ValueError(msg)

        try:
            self._digestmod = getattr(hashlib, digestmod)
        except AttributeError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Unsupported digest algorithm: {digestmod}") from exc

        self._secret = secret_key.encode("utf-8")
        self._salt = salt.encode("utf-8")
        self._clock = clock or time.time

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    def _signature(self, payload: str, timestamp: str) -> str:
        message = (
            self._salt
            + b"|"
            + payload.encode("utf-8")
            + b"|"
            + timestamp.encode("utf-8")
        )
        digest = hmac.new(self._secret, message, self._digestmod).digest()
        return self._b64encode(digest)

    def sign(self, value: bytes) -> str:
        """Return a signed token that encodes *value* and the current timestamp."""

        timestamp = str(int(self._clock()))
        payload = self._b64encode(value)
        signature = self._signature(payload, timestamp)
        return self._separator.join((payload, timestamp, signature))

    def unsign(self, token: str, *, max_age: int | float | None = None) -> bytes:
        """Validate *token* and return the original payload bytes.

        ``max_age`` expresses the allowed age of the token in seconds.  When it
        is ``None`` the token never expires.
        """

        parts = token.split(self._separator)
        if len(parts) != 3:
            raise BadSignature("Token structure is invalid")

        payload, timestamp_str, provided_signature = parts
        expected_signature = self._signature(payload, timestamp_str)
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise BadSignature("Token signature mismatch")

        try:
            issued_at = int(timestamp_str)
        except ValueError as exc:
            raise BadSignature("Timestamp is not an integer") from exc

        if max_age is not None:
            current_time = self._clock()
            if current_time - issued_at > max_age:
                raise SignatureExpired("Token has expired")

        try:
            return self._b64decode(payload)
        except (binascii.Error, ValueError) as exc:
            raise BadSignature("Payload is not valid base64") from exc
