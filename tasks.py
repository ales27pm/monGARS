from __future__ import annotations

import asyncio

from celery import Celery

from monGARS.core.conversation import ConversationalModule

celery_app = Celery("tasks", broker="redis://redis:6379/0")


@celery_app.task
def process_interaction(user_id: str, query: str) -> dict:
    """Run a single ConversationalModule interaction."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(
        ConversationalModule().generate_response(user_id, query)
    )
    loop.close()
    return response
