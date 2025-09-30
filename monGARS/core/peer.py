"""Peer-to-peer communication utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, List, Optional, Set

import httpx

from .security import decrypt_token, encrypt_token


class PeerCommunicator:
    """Send encrypted messages to peer nodes."""

    def __init__(
        self,
        peers: Iterable[str] | None = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        # Store peers in a set to avoid duplicates
        self.peers: Set[str] = {p.rstrip("/") for p in peers or []}
        self._client = client
        self._load_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None

    async def send(self, message: Optional[dict]) -> List[bool]:
        """Encrypt and broadcast message to all configured peers."""
        return await self._send_to_targets(message, sorted(self.peers))

    async def send_to(
        self, peers: Iterable[str], message: Optional[dict]
    ) -> List[bool]:
        """Encrypt and send a message to a subset of peers."""

        return await self._send_to_targets(message, [p.rstrip("/") for p in peers])

    async def _send_to_targets(
        self, message: Optional[dict], targets: Iterable[str]
    ) -> List[bool]:
        target_list = [t.rstrip("/") for t in targets]
        if not target_list:
            return []
        if not message:
            return [False] * len(target_list)

        payload = encrypt_token(json.dumps(message))

        async def _post(client: httpx.AsyncClient, url: str) -> bool:
            try:
                resp = await client.post(url, json={"payload": payload})
                success = resp.status_code == 200
                await resp.aclose()
                return success
            except httpx.HTTPError as exc:
                logging.error("Peer request to %s failed: %s", url, exc)
                return False
            except Exception as exc:  # pragma: no cover - unexpected errors
                logging.error("Peer request to %s error: %s", url, exc)
                return False

        async def _broadcast(client: httpx.AsyncClient) -> List[bool]:
            tasks = [_post(client, url) for url in target_list]
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

    def register_load_provider(
        self, provider: Callable[[], Awaitable[dict[str, Any]]]
    ) -> None:
        """Register an async callable that reports the local scheduler load."""

        self._load_provider = provider

    async def get_local_load(self) -> dict[str, Any]:
        """Return the most recent load snapshot reported by the scheduler."""

        if self._load_provider is None:
            return self._default_load_snapshot()
        try:
            snapshot = await self._load_provider()
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Peer load provider failed: %s", exc, exc_info=True)
            return self._default_load_snapshot()
        return self._normalise_load_snapshot(snapshot)

    async def fetch_peer_loads(self) -> dict[str, float]:
        """Query peers for their load factor to aid routing decisions."""

        if not self.peers:
            return {}

        targets = sorted(self.peers)

        async def _load_task(
            client: httpx.AsyncClient, peer_url: str
        ) -> tuple[str, float] | None:
            load_url = self._build_peer_endpoint(peer_url, "load")
            try:
                response = await client.get(load_url)
            except httpx.HTTPError as exc:
                logging.warning("Failed to fetch load from %s: %s", load_url, exc)
                return None
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning(
                    "Unexpected error fetching load from %s: %s", load_url, exc
                )
                return None
            try:
                if response.status_code != 200:
                    return None
                data = response.json()
            except ValueError:
                logging.warning("Invalid JSON load response from %s", load_url)
                data = None
            finally:
                await response.aclose()

            if not isinstance(data, dict):
                return None
            load = data.get("load_factor")
            if isinstance(load, (int, float)) and math.isfinite(load):
                return peer_url, float(load)
            return None

        async def _collect(client: httpx.AsyncClient) -> dict[str, float]:
            tasks = [_load_task(client, url) for url in targets]
            results = await asyncio.gather(*tasks) if tasks else []
            loads: dict[str, float] = {}
            for item in results:
                if item is None:
                    continue
                peer_url, value = item
                loads[peer_url] = value
            return loads

        if self._client:
            return await _collect(self._client)

        async with httpx.AsyncClient() as client:
            return await _collect(client)

    def _build_peer_endpoint(self, peer_url: str, suffix: str) -> str:
        base = peer_url.rstrip("/")
        if base.endswith("/message"):
            base = base[: -len("/message")]
        suffix_clean = suffix.lstrip("/")
        return f"{base}/{suffix_clean}"

    def _default_load_snapshot(self) -> dict[str, Any]:
        return {
            "scheduler_id": None,
            "queue_depth": 0,
            "active_workers": 0,
            "concurrency": 0,
            "load_factor": 0.0,
        }

    def _normalise_load_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        default = self._default_load_snapshot()
        if not isinstance(snapshot, dict):
            return default
        normalised = default.copy()
        normalised.update(
            {
                "scheduler_id": snapshot.get("scheduler_id", default["scheduler_id"]),
                "queue_depth": int(snapshot.get("queue_depth", default["queue_depth"])),
                "active_workers": int(
                    snapshot.get("active_workers", default["active_workers"])
                ),
                "concurrency": int(snapshot.get("concurrency", default["concurrency"])),
                "load_factor": float(
                    snapshot.get("load_factor", default["load_factor"])
                ),
            }
        )
        return normalised
