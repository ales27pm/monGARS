from __future__ import annotations

import hashlib
from typing import Iterable


def stable_hash(parts: Iterable[str]) -> str:
    sha1 = hashlib.sha1()
    for part in parts:
        sha1.update(part.encode("utf-8"))
        sha1.update(b"\x00")
    return sha1.hexdigest()[:16]
