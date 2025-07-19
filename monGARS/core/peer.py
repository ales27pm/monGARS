"""Peer-to-peer communication utilities."""

from __future__ import annotations

import json
from typing import Iterable, List, Optional

import httpx

from .security import decrypt_token, encrypt_token


class PeerCommunicator:
    """Send encrypted messages to peer nodes."""

    def __init__(
        self,
        peers: Iterable[str] | None = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.peers = list(peers or [])
        self._client = client

    async def send(self, message: dict) -> List[bool]:
        """Encrypt and broadcast message to all configured peers."""
        encoded = encrypt_token(json.dumps(message))
        results = []
        if self._client is None:
            async with httpx.AsyncClient() as client:
                for url in self.peers:
                    try:
                        resp = await client.post(url, json={"payload": encoded})
                        results.append(resp.status_code == 200)
                    except Exception:
                        results.append(False)
        else:
            for url in self.peers:
                try:
                    resp = await self._client.post(url, json={"payload": encoded})
                    results.append(resp.status_code == 200)
                except Exception:
                    results.append(False)
        return results

    @staticmethod
    def decode(payload: str) -> dict:
        """Decrypt incoming payload into a message dict."""
        data = decrypt_token(payload)
        return json.loads(data)
