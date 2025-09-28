from __future__ import annotations

from asgiref.sync import async_to_sync
from celery import Celery

from monGARS.api.dependencies import (
    get_adaptive_response_generator,
    get_personality_engine,
)
from monGARS.core.conversation import ConversationalModule

celery_app = Celery("tasks", broker="redis://redis:6379/0")

_shared_personality = get_personality_engine()
_shared_dynamic = get_adaptive_response_generator(_shared_personality)
_conversation_module = ConversationalModule(
    personality=_shared_personality,
    dynamic=_shared_dynamic,
)


@celery_app.task
def process_interaction(user_id: str, query: str) -> dict:
    """Run a single ConversationalModule interaction."""
    sync_generate = async_to_sync(_conversation_module.generate_response)
    return sync_generate(user_id, query)
