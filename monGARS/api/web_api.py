from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Dict, List

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, HttpUrl, field_validator

from monGARS.api.authentication import get_current_user
from monGARS.api.dependencies import get_hippocampus, get_peer_communicator
from monGARS.core.conversation import ConversationalModule
from monGARS.core.hippocampus import MemoryItem
from monGARS.core.peer import PeerCommunicator
from monGARS.core.security import SecurityManager, validate_user_input

from .ws_manager import WebSocketManager

app = FastAPI(title="monGARS API")
sec_manager = SecurityManager()
conversation_module = ConversationalModule()
ws_manager = WebSocketManager()


def get_conversational_module() -> ConversationalModule:
    return conversation_module


users_db: Dict[str, str] = {
    "u1": sec_manager.get_password_hash("x"),
    "u2": sec_manager.get_password_hash("y"),
}


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> dict:
    """Return a simple access token."""
    hashed = users_db.get(form_data.username)
    if not hashed or not sec_manager.verify_password(form_data.password, hashed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = sec_manager.create_access_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}


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


@app.websocket("/ws/chat/")
async def websocket_chat(websocket: WebSocket, token: str = Query(...)) -> None:
    """Stream conversation history and live updates over WebSocket."""
    try:
        payload = sec_manager.verify_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("Invalid token payload")
    except Exception:
        await websocket.close(code=1008)
        return

    await ws_manager.connect(user_id, websocket)
    store = get_hippocampus()
    try:
        history = await store.history(user_id, limit=10)
        for item in history:
            await websocket.send_json(
                {
                    "query": item.query,
                    "response": item.response,
                    "timestamp": item.timestamp.isoformat(),
                }
            )
        while True:
            try:
                await websocket.receive_text()
                await websocket.close(code=1003)
                break
            except WebSocketDisconnect:
                break
    finally:
        await ws_manager.disconnect(user_id, websocket)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

    @field_validator("message")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty")
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
        await ws_manager.broadcast(current_user.get("sub"), response_payload)
    except Exception:
        pass
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
    current_user: dict = Depends(get_current_user),
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
    current_user: dict = Depends(get_current_user),
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
    current_user: dict = Depends(get_current_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> List[str]:
    """Return the list of registered peer URLs."""
    return sorted(communicator.peers)
