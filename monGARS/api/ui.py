from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends

from ..core.aui import DEFAULT_ACTIONS, LLMActionSuggester
from .authentication import get_current_user
from .schemas import SuggestRequest, SuggestResponse

router = APIRouter(prefix="/api/v1/ui", tags=["ui"])
logger = logging.getLogger(__name__)
_suggester = LLMActionSuggester()


@router.post("/suggestions", response_model=SuggestResponse)
async def suggestions(
    body: SuggestRequest,
    current=Depends(get_current_user),
) -> SuggestResponse:  # noqa: ANN001
    prompt = body.prompt

    if body.actions:
        desc_map = {
            "code": "Write, refactor or generate code and tests.",
            "summarize": "Summarize long texts or chats.",
            "explain": "Explain a concept simply with examples.",
        }
        action_names = list(body.actions)
    else:
        desc_map = dict(DEFAULT_ACTIONS)
        action_names = [key for key, _ in DEFAULT_ACTIONS]

    context = {
        "user_id": getattr(current, "id", None),
        "action_descriptions": {
            action: desc_map.get(action, action) for action in action_names
        },
    }

    ordered = await asyncio.to_thread(
        _suggester.suggest,
        prompt,
        action_names,
        context,
    )
    if not ordered:
        ordered = list(action_names)
    else:
        remaining = [action for action in action_names if action not in ordered]
        ordered = ordered + remaining

    scores = {
        action: float(len(ordered) - index) for index, action in enumerate(ordered)
    }
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
