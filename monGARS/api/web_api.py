from __future__ import annotations

import logging
from datetime import datetime

try:
    from datetime import UTC  # Python 3.11+
except ImportError:  # Python 3.10 fallback
    from datetime import timezone

    UTC = timezone.utc
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

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
from monGARS.core.peer import PeerCommunicator
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.personality import PersonalityEngine
from monGARS.core.security import SecurityManager, validate_user_input
from monGARS.core.ui_events import event_bus, make_event

app = FastAPI(title="monGARS API")
logger = logging.getLogger(__name__)

from . import authentication as auth_routes
from . import model_management
from . import rag as rag_routes
from . import ui as ui_routes
from . import ws_manager

app.include_router(ws_manager.router)
app.include_router(auth_routes.router)
app.include_router(ui_routes.router)
app.include_router(model_management.router)
app.include_router(ws_ticket_router)
app.include_router(rag_routes.router)
_ws_manager = ws_manager.ws_manager
sec_manager = SecurityManager()
conversation_module: ConversationalModule | None = None
ws_manager = _ws_manager
DEFAULT_USERS: dict[str, dict[str, Any]] = {
    "u1": {
        "password_hash": sec_manager.get_password_hash("x"),
        "is_admin": True,
    },
    "u2": {
        "password_hash": sec_manager.get_password_hash("y"),
        "is_admin": False,
    },
}


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
        DEFAULT_USERS,
    )
    token = sec_manager.create_access_token(
        {
            "sub": user.username,
            "admin": user.is_admin,
        }
    )
    return {"access_token": token, "token_type": "bearer"}


@app.post("/api/v1/user/register")
async def register_user(
    reg: UserRegistration,
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
) -> dict:
    try:
        await repo.create_user_atomic(
            reg.username,
            sec_manager.get_password_hash(reg.password),
            reserved_usernames=DEFAULT_USERS.keys(),
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
    return {"status": "registered"}


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
    except Exception as exc:  # pragma: no cover - unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
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
        logging.getLogger(__name__).warning(
            "web_api.chat_invalid_input",
            extra={"user": user_id, "detail": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    try:
        result = await conv.generate_response(
            user_id, data["query"], session_id=chat.session_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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
    except Exception:  # pragma: no cover - defensive logging
        logging.getLogger(__name__).exception("web_api.chat_event_publish_failed")
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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
