"""HTTP clients for interacting with the monGARS API."""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar, cast

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


T = TypeVar("T")


class _BaseClient(ABC):
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

    @abstractmethod
    def _request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        data: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response | Awaitable[httpx.Response]:
        """Issue an HTTP request using the underlying transport."""
        ...

    def _update_auth_header(self, token: str | None) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.headers.update(_default_headers(token))

    def _handle_response(self, response: httpx.Response) -> httpx.Response:
        if response.is_success:
            return response
        raise _parse_error(response)


class _EndpointMixin(_BaseClient):
    def _execute(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        data: Any | None = None,
        headers: Mapping[str, str] | None = None,
        transform: Callable[[httpx.Response], T],
    ) -> T | Awaitable[T]:
        result = self._request(
            method,
            url,
            json=json,
            params=params,
            data=data,
            headers=headers,
        )
        if inspect.isawaitable(result):

            async def _async_wrapper() -> T:
                response = await result
                return transform(self._handle_response(response))

            return _async_wrapper()
        response = self._handle_response(result)
        return transform(response)

    def _login(
        self, username: str, password: str
    ) -> TokenResponse | Awaitable[TokenResponse]:
        def _transform(response: httpx.Response) -> TokenResponse:
            data = response.json()
            token = TokenResponse.model_validate(data)
            self.set_token(token.access_token)
            self._update_auth_header(token.access_token)
            return token

        return self._execute(
            "POST",
            "/token",
            data={"username": username, "password": password},
            headers={"User-Agent": USER_AGENT},
            transform=_transform,
        )

    def _register_user(
        self, payload: UserRegistration
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        return self._execute(
            "POST",
            "/api/v1/user/register",
            json=payload.model_dump(),
            transform=lambda response: response.json(),
        )

    def _chat(self, payload: ChatRequest) -> ChatResponse | Awaitable[ChatResponse]:
        return self._execute(
            "POST",
            "/api/v1/conversation/chat",
            json=payload.model_dump(exclude_none=True),
            transform=lambda response: ChatResponse.model_validate(response.json()),
        )

    def _history(
        self, user_id: str, *, limit: int = 10
    ) -> list[MemoryItem] | Awaitable[list[MemoryItem]]:
        def _transform(response: httpx.Response) -> list[MemoryItem]:
            payload = response.json()
            return [MemoryItem.model_validate(item) for item in payload]

        return self._execute(
            "GET",
            "/api/v1/conversation/history",
            params={"user_id": user_id, "limit": limit},
            transform=_transform,
        )

    def _fetch_rag_context(
        self, request: RagContextRequest
    ) -> RagContextResponse | Awaitable[RagContextResponse]:
        return self._execute(
            "POST",
            "/api/v1/review/rag-context",
            json=request.model_dump(exclude_none=True),
            transform=lambda response: RagContextResponse.model_validate(
                response.json()
            ),
        )

    def _suggest_actions(
        self, request: SuggestRequest
    ) -> SuggestResponse | Awaitable[SuggestResponse]:
        return self._execute(
            "POST",
            "/api/v1/ui/suggestions",
            json=request.model_dump(exclude_none=True),
            transform=lambda response: SuggestResponse.model_validate(response.json()),
        )

    def _send_peer_message(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        return self._execute(
            "POST",
            "/api/v1/peer/message",
            json=dict(payload),
            transform=lambda response: response.json(),
        )

    def _register_peer(
        self, registration: PeerRegistration
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        return self._execute(
            "POST",
            "/api/v1/peer/register",
            json=registration.model_dump(),
            transform=lambda response: response.json(),
        )

    def _unregister_peer(
        self, registration: PeerRegistration
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        return self._execute(
            "POST",
            "/api/v1/peer/unregister",
            json=registration.model_dump(),
            transform=lambda response: response.json(),
        )

    def _list_peers(self) -> list[str] | Awaitable[list[str]]:
        return self._execute(
            "GET",
            "/api/v1/peer/list",
            transform=lambda response: [str(item) for item in response.json()],
        )

    def _peer_load(self) -> PeerLoadSnapshot | Awaitable[PeerLoadSnapshot]:
        return self._execute(
            "GET",
            "/api/v1/peer/load",
            transform=lambda response: PeerLoadSnapshot.model_validate(response.json()),
        )

    def _publish_peer_telemetry(
        self, payload: PeerTelemetryPayload
    ) -> dict[str, str] | Awaitable[dict[str, str]]:
        return self._execute(
            "POST",
            "/api/v1/peer/telemetry",
            json=payload.model_dump(),
            transform=lambda response: response.json(),
        )

    def _peer_telemetry(
        self,
    ) -> PeerTelemetryEnvelope | Awaitable[PeerTelemetryEnvelope]:
        return self._execute(
            "GET",
            "/api/v1/peer/telemetry",
            transform=lambda response: PeerTelemetryEnvelope.model_validate(
                response.json()
            ),
        )

    def _model_configuration(
        self,
    ) -> ModelConfiguration | Awaitable[ModelConfiguration]:
        return self._execute(
            "GET",
            "/api/v1/models",
            transform=lambda response: ModelConfiguration.model_validate(
                response.json()
            ),
        )

    def _provision_models(
        self, request: ProvisionRequest
    ) -> ProvisionReport | Awaitable[ProvisionReport]:
        return self._execute(
            "POST",
            "/api/v1/models/provision",
            json=request.model_dump(exclude_none=True),
            transform=lambda response: ProvisionReport.model_validate(response.json()),
        )


class MonGARSSyncClient(_EndpointMixin):
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

    def _request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        data: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return self._client.request(
            method,
            url,
            json=json,
            params=params,
            data=data,
            headers=headers,
        )

    # Authentication -----------------------------------------------------------------
    def login(self, username: str, password: str) -> TokenResponse:
        return cast(TokenResponse, self._login(username, password))

    def register_user(self, payload: UserRegistration) -> dict[str, Any]:
        return cast(dict[str, Any], self._register_user(payload))

    # Conversation --------------------------------------------------------------------
    def chat(self, payload: ChatRequest) -> ChatResponse:
        return cast(ChatResponse, self._chat(payload))

    def history(self, user_id: str, *, limit: int = 10) -> list[MemoryItem]:
        return cast(list[MemoryItem], self._history(user_id, limit=limit))

    # RAG ------------------------------------------------------------------------------
    def fetch_rag_context(self, request: RagContextRequest) -> RagContextResponse:
        return cast(RagContextResponse, self._fetch_rag_context(request))

    # UI -------------------------------------------------------------------------------
    def suggest_actions(self, request: SuggestRequest) -> SuggestResponse:
        return cast(SuggestResponse, self._suggest_actions(request))

    # Peer coordination ---------------------------------------------------------------
    def send_peer_message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self._send_peer_message(payload))

    def register_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        return cast(dict[str, Any], self._register_peer(registration))

    def unregister_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        return cast(dict[str, Any], self._unregister_peer(registration))

    def list_peers(self) -> list[str]:
        return cast(list[str], self._list_peers())

    def peer_load(self) -> PeerLoadSnapshot:
        return cast(PeerLoadSnapshot, self._peer_load())

    def publish_peer_telemetry(self, payload: PeerTelemetryPayload) -> dict[str, str]:
        return cast(dict[str, str], self._publish_peer_telemetry(payload))

    def peer_telemetry(self) -> PeerTelemetryEnvelope:
        return cast(PeerTelemetryEnvelope, self._peer_telemetry())

    # Model management ---------------------------------------------------------------
    def model_configuration(self) -> ModelConfiguration:
        return cast(ModelConfiguration, self._model_configuration())

    def provision_models(self, request: ProvisionRequest) -> ProvisionReport:
        return cast(ProvisionReport, self._provision_models(request))


class MonGARSAsyncClient(_EndpointMixin):
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

    def _request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        data: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Awaitable[httpx.Response]:
        return self._client.request(
            method,
            url,
            json=json,
            params=params,
            data=data,
            headers=headers,
        )

    # Authentication -----------------------------------------------------------------
    async def login(self, username: str, password: str) -> TokenResponse:
        return await cast(Awaitable[TokenResponse], self._login(username, password))

    async def register_user(self, payload: UserRegistration) -> dict[str, Any]:
        return await cast(Awaitable[dict[str, Any]], self._register_user(payload))

    # Conversation --------------------------------------------------------------------
    async def chat(self, payload: ChatRequest) -> ChatResponse:
        return await cast(Awaitable[ChatResponse], self._chat(payload))

    async def history(self, user_id: str, *, limit: int = 10) -> list[MemoryItem]:
        return await cast(
            Awaitable[list[MemoryItem]], self._history(user_id, limit=limit)
        )

    # RAG ------------------------------------------------------------------------------
    async def fetch_rag_context(self, request: RagContextRequest) -> RagContextResponse:
        return await cast(
            Awaitable[RagContextResponse], self._fetch_rag_context(request)
        )

    # UI -------------------------------------------------------------------------------
    async def suggest_actions(self, request: SuggestRequest) -> SuggestResponse:
        return await cast(Awaitable[SuggestResponse], self._suggest_actions(request))

    # Peer coordination ---------------------------------------------------------------
    async def send_peer_message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await cast(Awaitable[dict[str, Any]], self._send_peer_message(payload))

    async def register_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        return await cast(Awaitable[dict[str, Any]], self._register_peer(registration))

    async def unregister_peer(self, registration: PeerRegistration) -> dict[str, Any]:
        return await cast(
            Awaitable[dict[str, Any]], self._unregister_peer(registration)
        )

    async def list_peers(self) -> list[str]:
        return await cast(Awaitable[list[str]], self._list_peers())

    async def peer_load(self) -> PeerLoadSnapshot:
        return await cast(Awaitable[PeerLoadSnapshot], self._peer_load())

    async def publish_peer_telemetry(
        self, payload: PeerTelemetryPayload
    ) -> dict[str, str]:
        return await cast(
            Awaitable[dict[str, str]], self._publish_peer_telemetry(payload)
        )

    async def peer_telemetry(self) -> PeerTelemetryEnvelope:
        return await cast(Awaitable[PeerTelemetryEnvelope], self._peer_telemetry())

    # Model management ---------------------------------------------------------------
    async def model_configuration(self) -> ModelConfiguration:
        return await cast(Awaitable[ModelConfiguration], self._model_configuration())

    async def provision_models(self, request: ProvisionRequest) -> ProvisionReport:
        return await cast(Awaitable[ProvisionReport], self._provision_models(request))


__all__ = ["MonGARSSyncClient", "MonGARSAsyncClient"]
