from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Annotated, Any, List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, HttpUrl, field_validator

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
from monGARS.api.ws_ticket import router as ws_ticket_router
from monGARS.core.conversation import ConversationalModule
from monGARS.core.hippocampus import MemoryItem
from monGARS.core.peer import PeerCommunicator
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.security import SecurityManager, validate_user_input
from monGARS.core.ui_events import event_bus, make_event

app = FastAPI(title="monGARS API")
logger = logging.getLogger(__name__)

from . import authentication as auth_routes
from . import ui as ui_routes
from . import ws_manager

app.include_router(ws_manager.router)
app.include_router(auth_routes.router)
app.include_router(ui_routes.router)
app.include_router(ws_ticket_router)
_ws_manager = ws_manager.ws_manager
sec_manager = SecurityManager()
_shared_personality = get_personality_engine()
_shared_dynamic = get_adaptive_response_generator(_shared_personality)
conversation_module = ConversationalModule(
    personality=_shared_personality,
    dynamic=_shared_dynamic,
)
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


def get_conversational_module() -> ConversationalModule:
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


class UserRegistration(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 150:
            raise ValueError("invalid username")
        if not re.match(r"^[A-Za-z0-9_-]+$", v):
            raise ValueError("invalid username")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password too short")
        return v


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


@app.get("/api/v1/conversation/history", response_model=List[MemoryItem])
async def conversation_history(
    user_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    store: Annotated[Any, Depends(get_hippocampus)],
    limit: int = 10,
) -> List[MemoryItem]:
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


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty")
        if len(v) > 1000:
            raise ValueError("message too long")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 100:
            raise ValueError("session_id too long")
        return v


class ChatResponse(BaseModel):
    response: str
    confidence: float
    processing_time: float


@app.post("/api/v1/conversation/chat", response_model=ChatResponse)
async def chat(
    chat: ChatRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    conv: Annotated[ConversationalModule, Depends(get_conversational_module)],
) -> ChatResponse:
    data = validate_user_input(
        {"user_id": current_user.get("sub"), "query": chat.message}
    )
    try:
        result = await conv.generate_response(
            current_user.get("sub"), data["query"], session_id=chat.session_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    response_payload = {
        "query": data["query"],
        "response": result.get("text", ""),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        event = make_event(
            "chat.message",
            current_user.get("sub"),
            response_payload,
        )
        await event_bus().publish(event)
    except Exception:  # pragma: no cover - defensive logging
        logger = logging.getLogger(__name__)
        logger.exception("web_api.chat_event_publish_failed")
    return ChatResponse(
        response=response_payload["response"],
        confidence=result.get("confidence", 0.0),
        processing_time=result.get("processing_time", 0.0),
    )


class PeerMessage(BaseModel):
    payload: str


class PeerRegistration(BaseModel):
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def normalize(cls, v: HttpUrl) -> str:
        return str(v).rstrip("/")


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


@app.get("/api/v1/peer/list", response_model=List[str])
async def peer_list(
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> List[str]:
    """Return the list of registered peer URLs."""
    return sorted(communicator.peers)
