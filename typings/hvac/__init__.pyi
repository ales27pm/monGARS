from __future__ import annotations

from typing import Any, Mapping

class _KVv2:
    def read_secret_version(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...

class _KVNamespace:
    v2: _KVv2

class _SecretsNamespace:
    kv: _KVNamespace

class Client:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    secrets: _SecretsNamespace

__all__ = ["Client"]
