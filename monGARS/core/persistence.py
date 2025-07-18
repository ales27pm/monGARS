from sqlalchemy import desc, select

from monGARS.core.init_db import ConversationHistory, Interaction, async_session_factory


class PersistenceRepository:
    def __init__(self, session_factory=async_session_factory):
        self._session_factory = session_factory

    async def save_interaction(self, interaction: Interaction):
        async with self._session_factory() as session:
            session.add(interaction)
            await session.commit()

    async def get_history(self, user_id: str, limit: int = 10):
        async with self._session_factory() as session:
            result = await session.execute(
                select(ConversationHistory)
                .where(ConversationHistory.user_id == user_id)
                .order_by(desc(ConversationHistory.timestamp))
                .limit(limit)
            )
            return result.scalars().all()
