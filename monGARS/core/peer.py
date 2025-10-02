"""Peer-to-peer communication utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, List, Optional, Set

import httpx

from .security import decrypt_token, encrypt_token


class PeerCommunicator:
    """Send encrypted messages to peer nodes."""

    def __init__(
        self,
        peers: Iterable[str] | None = None,
        client: Optional[httpx.AsyncClient] = None,
        identity: str | None = None,
    ) -> None:
        # Store peers in a set to avoid duplicates
        self.peers: Set[str] = {p.rstrip("/") for p in peers or []}
        self._client = client
        self._load_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
        self.identity = identity.rstrip("/") if identity else None
        self._telemetry_cache: dict[str, dict[str, Any]] = {}
        self._telemetry_lock = threading.Lock()
        self._telemetry_ttl_seconds = 120.0

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

    def set_identity(self, identity: str | None) -> None:
        """Set the local identity advertised to peers."""

        self.identity = identity.rstrip("/") if identity else None

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def update_local_telemetry(self, snapshot: Mapping[str, Any]) -> None:
        """Store the latest local telemetry snapshot."""

        data = self._normalise_telemetry(snapshot, source="local")
        self._store_telemetry("local", data)

    def ingest_remote_telemetry(
        self, peer_identifier: str | None, snapshot: Mapping[str, Any]
    ) -> None:
        """Record telemetry received from a remote peer."""

        source = (
            peer_identifier.rstrip("/") if isinstance(peer_identifier, str) else None
        )
        data = self._normalise_telemetry(snapshot, source=source)
        key = source or data.get("scheduler_id") or f"unknown:{id(snapshot)}"
        self._store_telemetry(key, data)

    async def broadcast_telemetry(self, snapshot: Mapping[str, Any]) -> None:
        """Publish telemetry to peers and cache it locally."""

        local_payload = self._normalise_telemetry(snapshot, source="local")
        self._store_telemetry("local", local_payload)
        if not self.peers:
            return

        remote_source = (
            self.identity
            or snapshot.get("source")
            or local_payload.get("scheduler_id")
            or "anonymous"
        )
        payload = self._normalise_telemetry(snapshot, source=remote_source)

        body = payload.copy()
        body.pop("monotonic_ts", None)

        async def _post(client: httpx.AsyncClient, peer_url: str) -> None:
            endpoint = self._build_peer_endpoint(peer_url, "telemetry")
            try:
                response = await client.post(endpoint, json=body)
            except httpx.HTTPError as exc:
                logging.warning("Failed to push telemetry to %s: %s", endpoint, exc)
                return
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning(
                    "Unexpected error pushing telemetry to %s: %s", endpoint, exc
                )
                return
            finally:
                if "response" in locals():
                    await response.aclose()

        async def _broadcast(client: httpx.AsyncClient) -> None:
            tasks = [_post(client, peer) for peer in sorted(self.peers)]
            if tasks:
                await asyncio.gather(*tasks)

        if self._client:
            await _broadcast(self._client)
            return

        async with httpx.AsyncClient() as client:
            await _broadcast(client)

    def get_cached_peer_loads(self, max_age: float = 30.0) -> dict[str, float]:
        """Return recently observed peer load factors keyed by identifier."""

        telemetry = self.get_peer_telemetry_map(max_age=max_age)
        loads: dict[str, float] = {}
        for peer_id, data in telemetry.items():
            if data.get("source") == "local":
                continue
            load = data.get("load_factor")
            if isinstance(load, (int, float)) and math.isfinite(load):
                loads[peer_id] = float(load)
        return loads

    def get_peer_telemetry(
        self, include_self: bool = False, max_age: float | None = None
    ) -> list[dict[str, Any]]:
        """Return telemetry snapshots ordered by recency."""

        telemetry_map = self.get_peer_telemetry_map(
            max_age=max_age or self._telemetry_ttl_seconds,
            include_self=include_self,
        )
        return sorted(
            [
                {k: v for k, v in data.items() if k != "age_seconds"}
                for data in telemetry_map.values()
            ],
            key=lambda item: item.get("observed_at") or "",
            reverse=True,
        )

    def get_peer_telemetry_map(
        self, max_age: float = 120.0, include_self: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Return telemetry keyed by peer identifier with age metadata."""

        now = time.monotonic()
        result: dict[str, dict[str, Any]] = {}
        with self._telemetry_lock:
            self._prune_telemetry_locked(now)
            for key, data in self._telemetry_cache.items():
                if not include_self and data.get("source") == "local":
                    continue
                age = now - data.get("monotonic_ts", now)
                if age > max_age:
                    continue
                record = data.copy()
                record["age_seconds"] = age
                record.pop("monotonic_ts", None)
                source_key = record.get("source") or key
                result[source_key] = record
        return result

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
            self.ingest_remote_telemetry(peer_url, data)
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

    def _store_telemetry(self, key: str, data: dict[str, Any]) -> None:
        timestamp = time.monotonic()
        record = data.copy()
        record["monotonic_ts"] = timestamp
        with self._telemetry_lock:
            self._telemetry_cache[key] = record
            self._prune_telemetry_locked(timestamp)

    def _prune_telemetry_locked(self, now: float) -> None:
        ttl = self._telemetry_ttl_seconds
        expired: list[str] = []
        for key, value in self._telemetry_cache.items():
            ts = value.get("monotonic_ts")
            if ts is None:
                continue
            if now - ts > ttl:
                expired.append(key)
        for key in expired:
            self._telemetry_cache.pop(key, None)

    def _normalise_telemetry(
        self, snapshot: Mapping[str, Any], source: str | None = None
    ) -> dict[str, Any]:
        observed_at_val = snapshot.get("observed_at")
        observed_at_dt: datetime | None = None
        if isinstance(observed_at_val, str):
            try:
                observed_at_dt = datetime.fromisoformat(
                    observed_at_val.replace("Z", "+00:00")
                )
            except ValueError:
                observed_at_dt = None
        elif isinstance(observed_at_val, datetime):
            observed_at_dt = observed_at_val

        if observed_at_dt:
            if observed_at_dt.tzinfo is None:
                observed_at_dt = observed_at_dt.replace(tzinfo=timezone.utc)
            observed_at = observed_at_dt.astimezone(timezone.utc).isoformat()
        else:
            observed_at = datetime.now(timezone.utc).isoformat()

        def _float(value: Any, default: float = 0.0) -> float:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(result) or result < 0:
                return default
            return result

        def _int(value: Any, default: int = 0) -> int:
            try:
                result = int(value)
            except (TypeError, ValueError):
                return default
            return max(default, result)

        scheduler_id = snapshot.get("scheduler_id")
        if scheduler_id is not None:
            scheduler_id = str(scheduler_id)

        normalised = {
            "scheduler_id": scheduler_id,
            "queue_depth": _int(snapshot.get("queue_depth")),
            "active_workers": _int(snapshot.get("active_workers")),
            "concurrency": max(1, _int(snapshot.get("concurrency"), 1)),
            "load_factor": _float(snapshot.get("load_factor")),
            "worker_uptime_seconds": _float(snapshot.get("worker_uptime_seconds"), 0.0),
            "tasks_processed": _int(snapshot.get("tasks_processed")),
            "tasks_failed": _int(snapshot.get("tasks_failed")),
            "task_failure_rate": _float(snapshot.get("task_failure_rate")),
            "observed_at": observed_at,
            "source": source or snapshot.get("source"),
        }
        return normalised

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
