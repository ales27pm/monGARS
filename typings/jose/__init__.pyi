from __future__ import annotations

from typing import Any, Mapping

class JWTError(Exception):
    ...

def jwt_encode(claims: Mapping[str, Any], key: Any, algorithm: str, headers: Mapping[str, Any] | None = ...) -> str: ...

def jwt_decode(token: str, key: Any, algorithms: list[str] | tuple[str, ...], options: Mapping[str, Any] | None = ..., audience: str | None = ..., issuer: str | None = ...) -> Mapping[str, Any]: ...

class _JWTModule:
    def encode(
        self,
        claims: Mapping[str, Any],
        key: Any,
        algorithm: str,
        headers: Mapping[str, Any] | None = ...,
    ) -> str: ...

    def decode(
        self,
        token: str,
        key: Any,
        algorithms: list[str] | tuple[str, ...],
        *,
        options: Mapping[str, Any] | None = ...,
        audience: str | None = ...,
        issuer: str | None = ...,
    ) -> Mapping[str, Any]: ...

jwt: _JWTModule

__all__ = ["JWTError", "jwt"]
