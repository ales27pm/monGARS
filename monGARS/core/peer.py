"""Peer-to-peer communication utilities."""

from __future__ import annotations

import asyncio
import json
import logging
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

    async def send(self, message: Optional[dict]) -> List[bool]:
        """Encrypt and broadcast message to all configured peers."""
        if not message:
            return [False] * len(self.peers)

        payload = encrypt_token(json.dumps(message))

        async def _post(client: httpx.AsyncClient, url: str) -> bool:
            try:
                resp = await client.post(url, json={"payload": payload})
                return resp.status_code == 200
            except httpx.HTTPError as exc:
                logging.error("Peer request to %s failed: %s", url, exc)
                return False
            except Exception as exc:  # pragma: no cover - unexpected errors
                logging.error("Peer request to %s error: %s", url, exc)
                return False

        async def _broadcast(client: httpx.AsyncClient) -> List[bool]:
            tasks = [_post(client, url) for url in self.peers]
            return await asyncio.gather(*tasks) if tasks else []

        if self._client:
            return await _broadcast(self._client)

        async with httpx.AsyncClient() as client:
            return await _broadcast(client)

    @staticmethod
    def decode(payload: str) -> dict:
        """Decrypt incoming payload into a message dict."""
        data = decrypt_token(payload)
        return json.loads(data)
