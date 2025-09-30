from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..init_db import ConversationHistory, Interaction, async_session_factory

logger = logging.getLogger(__name__)


SessionCallable = Callable[[Any], Awaitable[Any]]


class PersistenceRepository:
    def __init__(self, session_factory=async_session_factory):
        self._session_factory = session_factory

    async def _execute_with_retry(
        self, operation: SessionCallable, *, operation_name: str
    ) -> Any:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type(
                (OperationalError, InterfaceError, DBAPIError)
            ),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    async with self._session_factory() as session:
                        try:
                            return await operation(session)
                        except Exception as exc:  # pragma: no cover - defensive
                            in_tx = getattr(session, "in_transaction", None)
                            if callable(in_tx) and in_tx():
                                await session.rollback()
                            max_attempts = getattr(
                                attempt.retry_state.retry_object.stop,
                                "max_attempt_number",
                                None,
                            )
                            if (
                                max_attempts is None
                                or attempt.retry_state.attempt_number < max_attempts
                            ):
                                logger.warning(
                                    "persistence.%s.retry", operation_name, exc_info=exc
                                )
                            raise
        except Exception:
            logger.exception("persistence.%s.failed", operation_name)
            raise

    async def save_interaction(
        self,
        interaction: Interaction,
        *,
        history_query: str | None = None,
        history_response: str | None = None,
    ) -> None:
        async def operation(session) -> None:
            async with session.begin():
                await session.merge(interaction)
                if interaction.user_id and (interaction.message or history_query):
                    session.add(
                        ConversationHistory(
                            user_id=interaction.user_id,
                            query=history_query or interaction.message,
                            response=history_response or interaction.response,
                        )
                    )

        await self._execute_with_retry(operation, operation_name="save_interaction")

    async def save_history_entry(
        self, *, user_id: str, query: str, response: str
    ) -> None:
        async def operation(session) -> None:
            async with session.begin():
                session.add(
                    ConversationHistory(
                        user_id=user_id,
                        query=query,
                        response=response,
                    )
                )

        await self._execute_with_retry(operation, operation_name="save_history_entry")

    async def get_history(self, user_id: str, limit: int = 10):
        async def operation(session):
            result = await session.execute(
                select(ConversationHistory)
                .where(ConversationHistory.user_id == user_id)
                .order_by(desc(ConversationHistory.timestamp))
                .limit(limit)
            )
            return result.scalars().all()

        return await self._execute_with_retry(operation, operation_name="get_history")
