#!/usr/bin/env python3
"""Expose a host-only Ollama instance to Docker bridge networks.

This forwards raw TCP traffic from a configurable listen socket to the local
Ollama daemon bound to 127.0.0.1. It is intentionally protocol-agnostic so it
works for standard JSON requests and streamed responses without HTTP parsing.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging

LOGGER = logging.getLogger("mongars.ollama_host_proxy")


async def _pipe(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def _handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    target_host: str,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            target_host,
            target_port,
        )
    except Exception:
        LOGGER.exception(
            "proxy.connect_failed",
            extra={"peer": str(peer), "target_host": target_host, "target_port": target_port},
        )
        client_writer.close()
        await client_writer.wait_closed()
        return

    LOGGER.info(
        "proxy.connection_open",
        extra={"peer": str(peer), "target_host": target_host, "target_port": target_port},
    )
    try:
        await asyncio.gather(
            _pipe(client_reader, upstream_writer),
            _pipe(upstream_reader, client_writer),
        )
    finally:
        LOGGER.info("proxy.connection_closed", extra={"peer": str(peer)})


async def _serve(args: argparse.Namespace) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: _handle_client(
            reader,
            writer,
            target_host=args.target_host,
            target_port=args.target_port,
        ),
        host=args.listen_host,
        port=args.listen_port,
    )

    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    LOGGER.info(
        "proxy.started",
        extra={
            "listen": sockets,
            "target_host": args.target_host,
            "target_port": args.target_port,
        },
    )

    async with server:
        await server.serve_forever()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=11435)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=11434)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
