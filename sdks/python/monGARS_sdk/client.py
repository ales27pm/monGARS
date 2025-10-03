"""HTTP clients for interacting with the monGARS API."""

from __future__ import annotations

import json
from typing import Any, Mapping

import httpx

from .exceptions import APIError, AuthenticationError
from .models import (
    ChatRequest,
    ChatResponse,
    MemoryItem,
    ModelConfiguration,
    PeerLoadSnapshot,
    PeerRegistration,
    PeerTelemetryEnvelope,
    PeerTelemetryPayload,
    ProvisionReport,
    ProvisionRequest,
    RagContextRequest,
    RagContextResponse,
    SuggestRequest,
    SuggestResponse,
    TokenResponse,
    UserRegistration,
)

USER_AGENT = "monGARS-SDK/1.0"


def _default_headers(token: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_error(response: httpx.Response) -> APIError:
    detail: str | None = None
    payload: Any | None = None
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, Mapping) else None
    except json.JSONDecodeError:
        detail = response.text or None
    if response.status_code in {401, 403}:
        return AuthenticationError(response.status_code, detail, payload=payload)
    return APIError(response.status_code, detail, payload=payload)


class _BaseClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | httpx.Timeout | None = 30.0,
        verify: bool | str = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._verify = verify
        self._token: str | None = None

    @property
    def token(self) -> str | None:
        return self._token

    def set_token(self, token: str | None) -> None:
        self._token = token

    def _handle_response(self, response: httpx.Response) -> httpx.Response:
        if response.is_success:
            return response
        raise _parse_error(response)


class MonGARSSyncClient(_BaseClient):
    """Synchronous client built on top of :class:`httpx.Client`."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | httpx.Timeout | None = 30.0,
        verify: bool | str = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(base_url, timeout=timeout, verify=verify)
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=_default_headers(),
            verify=verify,
            transport=transport,
        )

    def __enter__(self) -> "MonGARSSyncClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401, ANN001
        self.close()

    def close(self) -> None:
        self._client.close()

    # Authentication -----------------------------------------------------------------
    def login(self, username: str, password: str) -> TokenResponse:
        response = self._client.post(
            "/token",
            data={"username": username, "password": password},
            headers={"User-Agent": USER_AGENT},
        )
        data = self._handle_response(response).json()
        token = TokenResponse.model_validate(data)
        self.set_token(token.access_token)
        self._client.headers.update(_default_headers(token.access_token))
        return token

    def register_user(self, payload: UserRegistration) -> dict[str, Any]:
        response = self._client.post(
            "/api/v1/user/register",
            json=payload.model_dump(),
        )
        return self._handle_response(response).json()

    # Conversation --------------------------------------------------------------------
    def chat(self, payload: ChatRequest) -> ChatResponse:
        response = self._client.post(
            "/api/v1/conversation/chat",
            json=payload.model_dump(exclude_none=True),
        )
        return ChatResponse.model_validate(self._handle_response(response).json())

    def history(self, user_id: str, *, limit: int = 10) -> list[MemoryItem]:
        response = self._client.get(
            "/api/v1/conversation/history",
            params={"user_id": user_id, "limit": limit},
        )
        payload = self._handle_response(response).json()
        return [MemoryItem.model_validate(item) for item in payload]

    # RAG ------------------------------------------------------------------------------
    def fetch_rag_context(self, request: RagContextRequest) -> RagContextResponse:
        response = self._client.post(
            "/api/v1/review/rag-context",
            json=request.model_dump(exclude_none=True),
        )
        return RagContextResponse.model_validate(self._handle_response(response).json())

    # UI -------------------------------------------------------------------------------
    def suggest_actions(self, request: SuggestRequest) -> SuggestResponse:
        response = self._client.post(
            "/api/v1/ui/suggestions",
            json=request.model_dump(exclude_none=True),
        )
        return SuggestResponse.model_validate(self._handle_response(response).json())

    # Peer coordination ---------------------------------------------------------------
    def send_peer_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            "/api/v1/peer/message",
            json=payload,
        )
        return self._handle_response(response).json()

    def register_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        response = self._client.post(
            "/api/v1/peer/register",
            json=registration.model_dump(),
        )
        return self._handle_response(response).json()

    def unregister_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        response = self._client.post(
            "/api/v1/peer/unregister",
            json=registration.model_dump(),
        )
        return self._handle_response(response).json()

    def list_peers(self) -> list[str]:
        response = self._client.get("/api/v1/peer/list")
        return list(self._handle_response(response).json())

    def peer_load(self) -> PeerLoadSnapshot:
        response = self._client.get("/api/v1/peer/load")
        return PeerLoadSnapshot.model_validate(self._handle_response(response).json())

    def publish_peer_telemetry(self, payload: PeerTelemetryPayload) -> dict[str, str]:
        response = self._client.post(
            "/api/v1/peer/telemetry",
            json=payload.model_dump(),
        )
        return self._handle_response(response).json()

    def peer_telemetry(self) -> PeerTelemetryEnvelope:
        response = self._client.get("/api/v1/peer/telemetry")
        return PeerTelemetryEnvelope.model_validate(
            self._handle_response(response).json()
        )

    # Model management ---------------------------------------------------------------
    def model_configuration(self) -> ModelConfiguration:
        response = self._client.get("/api/v1/models")
        return ModelConfiguration.model_validate(self._handle_response(response).json())

    def provision_models(self, request: ProvisionRequest) -> ProvisionReport:
        response = self._client.post(
            "/api/v1/models/provision",
            json=request.model_dump(exclude_none=True),
        )
        return ProvisionReport.model_validate(self._handle_response(response).json())


class MonGARSAsyncClient(_BaseClient):
    """Asynchronous client implemented using :class:`httpx.AsyncClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | httpx.Timeout | None = 30.0,
        verify: bool | str = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(base_url, timeout=timeout, verify=verify)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=_default_headers(),
            verify=verify,
            transport=transport,
        )

    async def __aenter__(self) -> "MonGARSAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401, ANN001
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def login(self, username: str, password: str) -> TokenResponse:
        response = await self._client.post(
            "/token",
            data={"username": username, "password": password},
            headers={"User-Agent": USER_AGENT},
        )
        data = self._handle_response(response).json()
        token = TokenResponse.model_validate(data)
        self.set_token(token.access_token)
        self._client.headers.update(_default_headers(token.access_token))
        return token

    async def register_user(self, payload: UserRegistration) -> dict[str, Any]:
        response = await self._client.post(
            "/api/v1/user/register",
            json=payload.model_dump(),
        )
        return self._handle_response(response).json()

    async def chat(self, payload: ChatRequest) -> ChatResponse:
        response = await self._client.post(
            "/api/v1/conversation/chat",
            json=payload.model_dump(exclude_none=True),
        )
        return ChatResponse.model_validate(self._handle_response(response).json())

    async def history(self, user_id: str, *, limit: int = 10) -> list[MemoryItem]:
        response = await self._client.get(
            "/api/v1/conversation/history",
            params={"user_id": user_id, "limit": limit},
        )
        payload = self._handle_response(response).json()
        return [MemoryItem.model_validate(item) for item in payload]

    async def fetch_rag_context(self, request: RagContextRequest) -> RagContextResponse:
        response = await self._client.post(
            "/api/v1/review/rag-context",
            json=request.model_dump(exclude_none=True),
        )
        return RagContextResponse.model_validate(self._handle_response(response).json())

    async def suggest_actions(self, request: SuggestRequest) -> SuggestResponse:
        response = await self._client.post(
            "/api/v1/ui/suggestions",
            json=request.model_dump(exclude_none=True),
        )
        return SuggestResponse.model_validate(self._handle_response(response).json())

    async def send_peer_message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._client.post(
            "/api/v1/peer/message",
            json=dict(payload),
        )
        return self._handle_response(response).json()

    async def register_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        response = await self._client.post(
            "/api/v1/peer/register",
            json=registration.model_dump(),
        )
        return self._handle_response(response).json()

    async def unregister_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        response = await self._client.post(
            "/api/v1/peer/unregister",
            json=registration.model_dump(),
        )
        return self._handle_response(response).json()

    async def list_peers(self) -> list[str]:
        response = await self._client.get("/api/v1/peer/list")
        payload = self._handle_response(response).json()
        return [str(item) for item in payload]

    async def peer_load(self) -> PeerLoadSnapshot:
        response = await self._client.get("/api/v1/peer/load")
        return PeerLoadSnapshot.model_validate(self._handle_response(response).json())

    async def publish_peer_telemetry(
        self, payload: PeerTelemetryPayload
    ) -> dict[str, str]:
        response = await self._client.post(
            "/api/v1/peer/telemetry",
            json=payload.model_dump(),
        )
        return self._handle_response(response).json()

    async def peer_telemetry(self) -> PeerTelemetryEnvelope:
        response = await self._client.get("/api/v1/peer/telemetry")
        return PeerTelemetryEnvelope.model_validate(
            self._handle_response(response).json()
        )

    async def model_configuration(self) -> ModelConfiguration:
        response = await self._client.get("/api/v1/models")
        return ModelConfiguration.model_validate(self._handle_response(response).json())

    async def provision_models(self, request: ProvisionRequest) -> ProvisionReport:
        response = await self._client.post(
            "/api/v1/models/provision",
            json=request.model_dump(exclude_none=True),
        )
        return ProvisionReport.model_validate(self._handle_response(response).json())


__all__ = ["MonGARSSyncClient", "MonGARSAsyncClient"]
