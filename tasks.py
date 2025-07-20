from __future__ import annotations

from asgiref.sync import async_to_sync
from celery import Celery

from monGARS.core.conversation import ConversationalModule

celery_app = Celery("tasks", broker="redis://redis:6379/0")


@celery_app.task
def process_interaction(user_id: str, query: str) -> dict:
    """Run a single ConversationalModule interaction."""
    sync_generate = async_to_sync(ConversationalModule().generate_response)
    return sync_generate(user_id, query)
