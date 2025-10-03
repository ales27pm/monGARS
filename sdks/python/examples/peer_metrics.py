"""Publish telemetry updates to a monGARS peer endpoint."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from monGARS_sdk import APIError, MonGARSAsyncClient, PeerTelemetryPayload

INTERVAL_SECONDS = float(os.environ.get("MONGARS_PUBLISH_INTERVAL", "30"))
MAX_BACKOFF_SECONDS = float(os.environ.get("MONGARS_MAX_BACKOFF", "300"))

logger = logging.getLogger("mongars.telemetry")


async def publish_loop(base_url: str, username: str, password: str) -> None:
    logging.basicConfig(level=logging.INFO)
    async with MonGARSAsyncClient(base_url) as client:
        await client.login(username, password)
        consecutive_errors = 0
        while True:
            payload = PeerTelemetryPayload(
                scheduler_id="research-node",
                queue_depth=1,
                active_workers=2,
                concurrency=4,
                load_factor=0.5,
                worker_uptime_seconds=3600,
                tasks_processed=128,
                tasks_failed=3,
                task_failure_rate=3 / 128,
                observed_at=datetime.utcnow(),
                source="research-node.local",
            )
            try:
                await client.publish_peer_telemetry(payload)
            except asyncio.CancelledError:
                raise
            except APIError as exc:
                consecutive_errors += 1
                backoff = min(
                    INTERVAL_SECONDS * 2 ** (consecutive_errors - 1),
                    MAX_BACKOFF_SECONDS,
                )
                logger.warning(
                    "Telemetry publish failed with API error (attempt %s): %s",
                    consecutive_errors,
                    exc,
                )
                await asyncio.sleep(backoff)
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                consecutive_errors += 1
                backoff = min(
                    INTERVAL_SECONDS * 2 ** (consecutive_errors - 1),
                    MAX_BACKOFF_SECONDS,
                )
                logger.exception(
                    "Unexpected telemetry failure (attempt %s)", consecutive_errors
                )
                await asyncio.sleep(backoff)
                continue

            consecutive_errors = 0
            await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    asyncio.run(
        publish_loop(
            os.environ.get("MONGARS_BASE_URL", "http://localhost:8000"),
            os.environ.get("MONGARS_USERNAME", "u1"),
            os.environ.get("MONGARS_PASSWORD", "x"),
        )
    )
