from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.aui import DEFAULT_ACTIONS, AUISuggester
from .authentication import get_current_user

router = APIRouter(prefix="/api/v1/ui", tags=["ui"])
logger = logging.getLogger(__name__)
_suggester = AUISuggester()


class SuggestRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    actions: list[str] | None = None


class SuggestResponse(BaseModel):
    actions: list[str]
    scores: dict[str, float]
    model: str


@router.post("/suggestions", response_model=SuggestResponse)
async def suggestions(
    body: SuggestRequest, current=Depends(get_current_user)
) -> SuggestResponse:  # noqa: ANN001
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is empty")

    if body.actions:
        desc_map = {
            "code": "Write, refactor or generate code and tests.",
            "summarize": "Summarize long texts or chats.",
            "explain": "Explain a concept simply with examples.",
        }
        actions = [(key, desc_map.get(key, key)) for key in body.actions]
    else:
        actions = DEFAULT_ACTIONS

    ordered, scores = await _suggester.order(prompt, actions)
    model_name = _suggester.model_name
    logger.debug(
        "aui_suggestions_generated",
        extra={
            "user_id": getattr(current, "id", None),
            "actions": ordered,
            "model": model_name,
        },
    )
    return SuggestResponse(actions=ordered, scores=scores, model=model_name)
