from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

try:
    from datetime import UTC  # Python 3.11+
except ImportError:  # Python 3.10 fallback
    from datetime import timezone

    UTC = timezone.utc
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from httpx import HTTPError
from sqlalchemy.exc import SQLAlchemyError

from monGARS.api.authentication import (
    authenticate_user,
    get_current_admin_user,
    get_current_user,
)
from monGARS.api.dependencies import (
    get_adaptive_response_generator,
    get_hippocampus,
    get_peer_communicator,
    get_persistence_repository,
    get_personality_engine,
)
from monGARS.api.schemas import (
    ChatRequest,
    ChatResponse,
    PeerLoadSnapshot,
    PeerMessage,
    PeerRegistration,
    PeerTelemetryEnvelope,
    PeerTelemetryPayload,
    UserRegistration,
)
from monGARS.api.ws_ticket import router as ws_ticket_router
from monGARS.core.conversation import ConversationalModule
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.hippocampus import MemoryItem
from monGARS.core.llm_integration import CircuitBreakerOpenError
from monGARS.core.peer import PeerCommunicator
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.personality import PersonalityEngine
from monGARS.core.security import SecurityManager, validate_user_input
from monGARS.core.ui_events import BackendUnavailable, event_bus, make_event

from . import authentication as auth_routes
from . import model_management
from . import rag as rag_routes
from . import ui as ui_routes
from . import ws_manager

_ws_manager = ws_manager.ws_manager
sec_manager = SecurityManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate dependency overrides and prepare application state."""

    override = app.dependency_overrides.get(get_persistence_repository)
    if override is not None and not callable(override):
        logger.error("lifespan.invalid_override", extra={"override": override})
        raise TypeError("Dependency override must be callable")
    yield


app = FastAPI(title="monGARS API", lifespan=lifespan)
logger = logging.getLogger(__name__)

app.include_router(ws_manager.router)
app.include_router(auth_routes.router)
app.include_router(ui_routes.router)
app.include_router(model_management.router)
app.include_router(ws_ticket_router)
app.include_router(rag_routes.router)

conversation_module: ConversationalModule | None = None
ws_manager = _ws_manager


def _redact_user_id(user_id: str) -> str:
    """Return a short, stable identifier suitable for logs."""

    if not isinstance(user_id, str) or not user_id:
        return "u:unknown"
    digest = hashlib.blake2s(user_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"u:{digest}"


def _get_adaptive_response_generator_for_personality(
    personality: Annotated[PersonalityEngine, Depends(get_personality_engine)],
) -> AdaptiveResponseGenerator:
    """Resolve the adaptive response generator for the provided personality."""

    return get_adaptive_response_generator(personality)


def get_conversational_module(
    personality: Annotated[PersonalityEngine, Depends(get_personality_engine)],
    dynamic: Annotated[
        AdaptiveResponseGenerator,
        Depends(_get_adaptive_response_generator_for_personality),
    ],
) -> ConversationalModule:
    global conversation_module
    if conversation_module is None:
        conversation_module = ConversationalModule(
            personality=personality,
            dynamic=dynamic,
        )
    return conversation_module


@app.post("/token")
async def login(
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> dict:
    """Return a simple access token."""
    user = await authenticate_user(
        repo,
        form_data.username,
        form_data.password,
        sec_manager,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = sec_manager.create_access_token(
        {
            "sub": user.username,
            "admin": user.is_admin,
        }
    )
    return {"access_token": token, "token_type": "bearer"}


async def _persist_registration(
    repo: PersistenceRepository,
    reg: UserRegistration,
    *,
    is_admin: bool,
) -> dict:
    try:
        await repo.create_user_atomic(
            reg.username,
            sec_manager.get_password_hash(reg.password),
            is_admin=is_admin,
        )
    except ValueError as exc:
        logger.debug(
            "auth.register.conflict",
            extra={"username": reg.username},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return {"status": "registered", "is_admin": is_admin}


@app.post("/api/v1/user/register")
async def register_user(
    reg: UserRegistration,
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
) -> dict:
    return await _persist_registration(repo, reg, is_admin=False)


@app.post("/api/v1/user/register/admin")
async def register_admin_user(
    reg: UserRegistration,
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
) -> dict:
    try:
        if await repo.has_admin_user():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin already exists",
            )
    except HTTPException:
        raise
    except (
        RuntimeError,
        SQLAlchemyError,
    ) as exc:  # pragma: no cover - unexpected failure
        logger.error(
            "auth.register_admin.state_check_failed",
            extra={"username": reg.username},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to determine admin availability",
        ) from exc
    return await _persist_registration(repo, reg, is_admin=True)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    return {"status": "ready"}


@app.get("/api/v1/conversation/history", response_model=list[MemoryItem])
async def conversation_history(
    user_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    store: Annotated[Any, Depends(get_hippocampus)],
    limit: int = 10,
) -> list[MemoryItem]:
    if user_id != current_user.get("sub"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    try:
        return await store.history(user_id, limit=limit)
    except (
        RuntimeError,
        SQLAlchemyError,
    ) as exc:  # pragma: no cover - unexpected errors
        logger.exception(
            "conversation.history_failed",
            extra={"user": _redact_user_id(user_id), "limit": limit},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to load conversation history",
        ) from exc


@app.post("/api/v1/conversation/chat", response_model=ChatResponse)
async def chat(
    chat: ChatRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    conv: Annotated[ConversationalModule, Depends(get_conversational_module)],
) -> ChatResponse:
    user_id = current_user["sub"]
    try:
        data = validate_user_input({"user_id": user_id, "query": chat.message})
    except ValueError as exc:
        logger.warning(
            "web_api.chat_invalid_input",
            extra={"user": _redact_user_id(user_id), "detail": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    try:
        result = await conv.generate_response(
            user_id, data["query"], session_id=chat.session_id
        )
    except asyncio.TimeoutError as exc:
        logger.warning(
            "web_api.chat_inference_timeout",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Chat response timed out",
        ) from exc
    except HTTPException:
        raise
    except (
        CircuitBreakerOpenError,
        HTTPError,
        RuntimeError,
        SQLAlchemyError,
        ValueError,
    ) as exc:
        logger.exception(
            "web_api.chat_inference_failed",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate chat response",
        ) from exc
    response_payload = {
        "query": data["query"],
        "response": result.get("text", ""),
        "timestamp": datetime.now(UTC).isoformat(),
        "speech_turn": result.get("speech_turn"),
    }
    try:
        event = make_event(
            "chat.message",
            user_id,
            response_payload,
        )
        await event_bus().publish(event)
    except BackendUnavailable:
        logger.exception(
            "web_api.chat_event_publish_failed",
            extra={"user": _redact_user_id(user_id)},
        )
    except (OSError, RuntimeError):
        logger.exception(
            "web_api.chat_event_publish_failed",
            extra={"user": _redact_user_id(user_id)},
        )
    return ChatResponse(
        response=response_payload["response"],
        confidence=result.get("confidence", 0.0),
        processing_time=result.get("processing_time", 0.0),
        speech_turn=result.get("speech_turn"),
    )


@app.post("/api/v1/peer/message")
async def peer_message(
    message: PeerMessage,
    current_user: dict = Depends(get_current_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict:
    """Receive an encrypted message from a peer."""
    try:
        data = communicator.decode(message.payload)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "peer.message_invalid_payload",
            extra={"detail": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid peer payload",
        ) from exc
    except (RuntimeError, OSError, TypeError) as exc:
        logger.exception("peer.message_decode_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process peer payload",
        ) from exc
    return {"status": "received", "data": data}


@app.post("/api/v1/peer/register")
async def peer_register(
    registration: PeerRegistration,
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict:
    """Register a peer URL for future broadcasts."""
    url = registration.url
    if url in communicator.peers:
        return {"status": "already registered", "count": len(communicator.peers)}
    communicator.peers.add(url)
    return {"status": "registered", "count": len(communicator.peers)}


@app.post("/api/v1/peer/unregister")
async def peer_unregister(
    registration: PeerRegistration,
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict:
    """Remove a previously registered peer URL."""
    url = registration.url
    if url in communicator.peers:
        communicator.peers.remove(url)
        return {"status": "unregistered", "count": len(communicator.peers)}
    return {"status": "not registered", "count": len(communicator.peers)}


@app.get("/api/v1/peer/list", response_model=list[str])
async def peer_list(
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> list[str]:
    """Return the list of registered peer URLs."""
    return sorted(communicator.peers)


@app.get("/api/v1/peer/load", response_model=PeerLoadSnapshot)
async def peer_load(
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> PeerLoadSnapshot:
    """Expose this node's scheduler load metrics to peers."""

    snapshot = await communicator.get_local_load()
    return PeerLoadSnapshot(**snapshot)


@app.post("/api/v1/peer/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def peer_telemetry_ingest(
    report: PeerTelemetryPayload,
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict[str, str]:
    """Accept telemetry published by peer schedulers."""

    communicator.ingest_remote_telemetry(report.source, report.model_dump())
    return {"status": "accepted"}


@app.get("/api/v1/peer/telemetry", response_model=PeerTelemetryEnvelope)
async def peer_telemetry_snapshot(
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> PeerTelemetryEnvelope:
    """Return cached telemetry for local and remote schedulers."""

    telemetry = communicator.get_peer_telemetry(include_self=True)
    payloads = [PeerTelemetryPayload(**entry) for entry in telemetry]
    return PeerTelemetryEnvelope(telemetry=payloads)
