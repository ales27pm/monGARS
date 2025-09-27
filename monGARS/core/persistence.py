import logging

from sqlalchemy import desc, select

from ..init_db import ConversationHistory, Interaction, async_session_factory

logger = logging.getLogger(__name__)


class PersistenceRepository:
    def __init__(self, session_factory=async_session_factory):
        self._session_factory = session_factory

    async def save_interaction(
        self,
        interaction: Interaction,
        *,
        history_query: str | None = None,
        history_response: str | None = None,
    ):
        async with self._session_factory() as session:
            try:
                session.add(interaction)
                if interaction.user_id and (interaction.message or history_query):
                    session.add(
                        ConversationHistory(
                            user_id=interaction.user_id,
                            query=history_query or interaction.message,
                            response=history_response or interaction.response,
                        )
                    )
                await session.commit()
            except Exception:
                logger.exception("Exception occurred while saving interaction")
                await session.rollback()
                raise

    async def get_history(self, user_id: str, limit: int = 10):
        async with self._session_factory() as session:
            result = await session.execute(
                select(ConversationHistory)
                .where(ConversationHistory.user_id == user_id)
                .order_by(desc(ConversationHistory.timestamp))
                .limit(limit)
            )
            return result.scalars().all()
