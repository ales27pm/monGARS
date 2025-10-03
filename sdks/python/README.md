# monGARS Python SDK

The official Python helper library for interacting with the monGARS FastAPI
service. It provides both synchronous and asynchronous clients built on top of
[`httpx`](https://www.python-httpx.org/), typed request/response models, and
robust error handling.

## Installation

The SDK is published from this repository. To use it in another project, add
`sdks/python` to your dependency path or package it via `pip install .` from the
SDK directory. The package targets Python 3.11 and depends on `httpx` and
`pydantic` which are already part of the main application requirements.

```bash
cd sdks/python
pip install .
```

## Quick start

```python
from monGARS_sdk import ChatRequest, MonGARSSyncClient

client = MonGARSSyncClient("https://api.mongars.local")
token = client.login("alice", "secret-password")

response = client.chat(ChatRequest(message="Hello there!"))
print(response.response)
```

For asynchronous usage:

```python
import asyncio
from monGARS_sdk import ChatRequest, MonGARSAsyncClient

async def main():
    async with MonGARSAsyncClient("https://api.mongars.local") as client:
        await client.login("alice", "secret-password")
        result = await client.chat(ChatRequest(message="Hi!"))
        print(result.response)

asyncio.run(main())
```

## Reference clients

Two reference client implementations live in [`examples/`](examples/):

- `chat_cli.py` demonstrates an interactive terminal chat experience with token
  management and graceful error handling.
- `peer_metrics.py` showcases periodic telemetry publication.

These scripts are safe to copy and adapt for bespoke workflows.

## API coverage

The SDK currently wraps the following endpoints:

- Authentication: `/token`, `/api/v1/user/register`
- Conversation: `/api/v1/conversation/chat`, `/api/v1/conversation/history`
- Review & RAG: `/api/v1/review/rag-context`
- UI assistance: `/api/v1/ui/suggestions`
- Peer coordination: `/api/v1/peer/*`
- Model management: `/api/v1/models`, `/api/v1/models/provision`

Each helper returns typed models and raises `APIError` (or a specialised
`AuthenticationError`) when the API responds with a non-success status code.

## Development

The package ships with `py.typed` for type-checker support and follows the root
project formatting rules (`black` and `isort`). Run the shared test suite from
repo root:

```bash
pytest -q tests/test_python_sdk.py
```
