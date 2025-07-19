from __future__ import annotations

from typing import Dict, List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from monGARS.api.authentication import get_current_user
from monGARS.api.dependencies import get_hippocampus, get_peer_communicator
from monGARS.core.hippocampus import MemoryItem
from monGARS.core.peer import PeerCommunicator
from monGARS.core.security import SecurityManager

app = FastAPI(title="monGARS API")
sec_manager = SecurityManager()
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
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    store=Depends(get_hippocampus),
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


peer_comm = PeerCommunicator()


class PeerMessage(BaseModel):
    payload: str


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
