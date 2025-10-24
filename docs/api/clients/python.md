# Python Client Example

> **Last updated:** 2025-10-03 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

Use [`httpx`](https://www.python-httpx.org/) and
[`pydantic`](https://docs.pydantic.dev/) to interact with the monGARS API. The
snippet below mirrors request/response schemas defined in `monGARS.api.schemas`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install httpx[http2] pydantic
```

## Usage
```python
from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    confidence: float
    processing_time: float


async def main() -> None:
    async with httpx.AsyncClient(base_url="https://mongars.example.com") as client:
        token_resp = await client.post(
            "/token",
            data={"username": "u1", "password": "x"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        token = TokenResponse.model_validate_json(token_resp.text)

        headers = {"Authorization": f"Bearer {token.access_token}"}
        payload = ChatRequest(message="Summarise our latest incident report")
        response = await client.post(
            "/api/v1/conversation/chat",
            json=payload.model_dump(mode="json"),
            headers=headers,
            timeout=60,
        )
        chat = ChatResponse.model_validate_json(response.text)
        print(chat.response)

        # For WebSocket streaming request a signed ticket first, then connect to
        # /ws/chat/?t=<ticket> using the same Authorization header if needed:
        # ticket = await client.post("/api/v1/auth/ws/ticket", headers=headers)
        # connect using websockets.connect("wss://.../ws/chat/?t=" + ticket.json()["ticket"])


if __name__ == "__main__":
    asyncio.run(main())
```

## Extending
- Generate a full SDK:
  ```bash
  openapi-python-client generate --path docs/api/openapi.json \
    --config openapi-python-client.toml
  ```
- Import `monGARS.api.schemas` directly inside automation or data pipelines to
  reuse validation logic.
