"""Interactive CLI demonstrating the monGARS Python SDK."""

from __future__ import annotations

import getpass
import sys
from typing import Iterable

from monGARS_sdk import APIError, ChatRequest, MonGARSSyncClient


def _prompt(prompt: str, *, secret: bool = False) -> str:
    return getpass.getpass(prompt) if secret else input(prompt)


def _print_history(rows: Iterable[str]) -> None:
    for row in rows:
        sys.stdout.write(f"{row}\n")
    sys.stdout.flush()


def main() -> None:
    base_url = (
        _prompt("API base URL [http://localhost:8000]: ") or "http://localhost:8000"
    )
    username = _prompt("Username: ")
    password = _prompt("Password: ", secret=True)

    with MonGARSSyncClient(base_url) as client:
        try:
            client.login(username, password)
        except APIError as exc:  # pragma: no cover - manual run only
            sys.stderr.write(f"Login failed: {exc}\n")
            return

        sys.stdout.write("Connected! Type 'quit' to exit.\n")
        while True:
            message = _prompt("You: ")
            if message.lower() in {"quit", "exit"}:
                break
            try:
                reply = client.chat(ChatRequest(message=message))
            except APIError as exc:  # pragma: no cover - manual run only
                sys.stderr.write(f"Request failed: {exc}\n")
                continue
            _print_history([f"Assistant: {reply.response}"])


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
