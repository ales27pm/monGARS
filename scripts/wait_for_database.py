#!/usr/bin/env python3
"""CLI helper to block until the PostgreSQL database is available."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Final

from sqlalchemy.engine import URL

try:
    from init_db import (
        DEFAULT_DB_STARTUP_RETRY_INTERVAL,
        DEFAULT_DB_STARTUP_TIMEOUT,
        build_sync_url,
        coerce_positive_float,
        render_url,
        wait_for_database,
    )
except (
    ModuleNotFoundError
):  # pragma: no cover - script executed without package installed
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from init_db import (
        DEFAULT_DB_STARTUP_RETRY_INTERVAL,
        DEFAULT_DB_STARTUP_TIMEOUT,
        build_sync_url,
        coerce_positive_float,
        render_url,
        wait_for_database,
    )

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_TIMEOUT_ENV: Final[str] = "DB_STARTUP_TIMEOUT"
DEFAULT_INTERVAL_ENV: Final[str] = "DB_STARTUP_RETRY_INTERVAL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block until the configured PostgreSQL database is ready to accept connections.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Maximum seconds to wait before failing. Defaults to the value in "
            f"${DEFAULT_TIMEOUT_ENV} or {DEFAULT_DB_STARTUP_TIMEOUT}."
        ),
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=None,
        help=(
            "Seconds to sleep between attempts. Defaults to the value in "
            f"${DEFAULT_INTERVAL_ENV} or {DEFAULT_DB_STARTUP_RETRY_INTERVAL}."
        ),
    )
    return parser.parse_args()


def resolve_timeout(raw: float | None, *, env_key: str, default: float) -> float:
    if raw is not None and raw > 0:
        return raw
    return coerce_positive_float(os.getenv(env_key), default=default, minimum=0.0)


def resolve_retry_interval(raw: float | None, *, env_key: str, default: float) -> float:
    if raw is not None and raw > 0:
        return raw
    return coerce_positive_float(os.getenv(env_key), default=default, minimum=0.0)


def main() -> int:
    args = parse_args()
    url: URL = build_sync_url()
    timeout = resolve_timeout(
        args.timeout,
        env_key=DEFAULT_TIMEOUT_ENV,
        default=DEFAULT_DB_STARTUP_TIMEOUT,
    )
    retry_interval = resolve_retry_interval(
        args.retry_interval,
        env_key=DEFAULT_INTERVAL_ENV,
        default=DEFAULT_DB_STARTUP_RETRY_INTERVAL,
    )

    LOGGER.info("Waiting for PostgreSQL at %s", render_url(url, hide_password=True))
    wait_for_database(url, timeout=timeout, retry_interval=retry_interval)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TimeoutError as exc:  # pragma: no cover - CLI surface area
        LOGGER.error("%s", exc)
        sys.exit(1)
    except Exception:  # pragma: no cover - defensive guard for CLI usage
        LOGGER.exception("Unexpected failure while waiting for database")
        sys.exit(1)
