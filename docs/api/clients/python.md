# Python Client Example

The following example uses [`httpx`](https://www.python-httpx.org/) and
[`pydantic`](https://docs.pydantic.dev/) to provide a lightweight, fully typed
client for the monGARS API. It mirrors the request/response schemas defined in
`monGARS.api.schemas` and can be extended or generated automatically with tools
like [`openapi-python-client`](https://github.com/openapi-generators/openapi-python-client).

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
from typing import Any

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
        )
        token = TokenResponse.model_validate_json(token_resp.text)

        headers = {"Authorization": f"Bearer {token.access_token}"}
        payload = ChatRequest(message="Summarise our latest incident report")
        response = await client.post(
            "/api/v1/conversation/chat",
            json=payload.model_dump(mode="json"),
            headers=headers,
        )
        chat = ChatResponse.model_validate_json(response.text)
        print(chat.response)


if __name__ == "__main__":
    asyncio.run(main())
```

## Extending

- Generate a fully featured SDK:

  ```bash
  openapi-python-client generate --path docs/api/openapi.json --config openapi-python-client.toml
  ```

- Use `monGARS.api.schemas` directly for shared validation logic inside custom
data pipelines or CLI tooling.
