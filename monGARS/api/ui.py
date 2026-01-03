from __future__ import annotations

import asyncio
import inspect
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
        "user_id": current.get("sub") or current.get("id"),
        "action_descriptions": {
            action: desc_map.get(action, action) for action in action_names
        },
    }

    ordered: list[str] = []
    explicit_scores: dict[str, float] | None = None
    action_pairs = [(action, desc_map.get(action, action)) for action in action_names]

    if hasattr(_suggester, "order"):
        try:
            order_callable = getattr(_suggester, "order")
            result = order_callable(prompt, action_pairs)
            if inspect.isawaitable(result):
                ordered_result, score_map = await result  # type: ignore[misc]
            else:
                ordered_result, score_map = result  # type: ignore[misc]
            ordered = [action for action in ordered_result if isinstance(action, str)]
            explicit_scores = {
                str(action): float(value) for action, value in (score_map or {}).items()
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("aui.order_failed_fallback", extra={"error": str(exc)})
            ordered = await asyncio.to_thread(
                _suggester.suggest,
                prompt,
                action_names,
                context,
            )
    else:
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

    scores: dict[str, float] = {}
    for index, action in enumerate(ordered):
        if explicit_scores is not None:
            value = explicit_scores.get(action)
            if value is None:
                value = float(len(ordered) - index)
        else:
            value = float(len(ordered) - index)
        scores[action] = float(value)
    model_name = _suggester.model_name
    logger.debug(
        "aui_suggestions_generated",
        extra={
            "user_id": current.get("sub") or current.get("id"),
            "actions": ordered,
            "model": model_name,
        },
    )
    return SuggestResponse(actions=ordered, scores=scores, model=model_name)
