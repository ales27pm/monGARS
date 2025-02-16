import asyncio
import logging
from collections import deque
from monGARS.core.init_db import async_session_factory, UserPreferences
from sqlalchemy import select, update
from datetime import datetime

logger = logging.getLogger(__name__)

class MimicryModule:
    def __init__(self, long_term_weight: float = 0.9, short_term_weight: float = 0.1, history_length: int = 10):
        self.long_term_weight = long_term_weight
        self.short_term_weight = short_term_weight
        self.history_length = history_length
        self.user_profiles = {}
        self.lock = asyncio.Lock()

    async def _get_profile(self, user_id: str) -> dict:
        async with async_session_factory() as session:
            try:
                result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
                user_preferences = result.scalars().first()
                if user_preferences and user_preferences.interaction_style:
                    return user_preferences.interaction_style
                else:
                    return {"long_term": {}, "short_term": deque(maxlen=self.history_length)}
            except Exception as e:
                logger.error(f"Error retrieving profile for user {user_id}: {e}")
                return {"long_term": {}, "short_term": deque(maxlen=self.history_length)}

    async def _update_profile_db(self, user_id: str, profile: dict):
        async with async_session_factory() as session:
            try:
                stmt = update(UserPreferences).where(UserPreferences.user_id == user_id).values(interaction_style=profile)
                await session.execute(stmt)
                await session.commit()
                logger.info(f"Mimicry profile updated for user {user_id}")
            except Exception as e:
                await session.rollback()
                logger.error(f"DB error updating mimicry profile for user {user_id}: {e}")
                raise

    async def update_profile(self, user_id: str, interaction: dict) -> dict:
        async with self.lock:
            profile = await self._get_profile(user_id)
            new_features = {
                "sentence_length": len(interaction.get("message", "").split()),
                "positive_sentiment": interaction.get("feedback", 0.5)
            }
            for feature, value in new_features.items():
                if feature in profile.get("long_term", {}):
                    profile["long_term"][feature] = (
                        self.long_term_weight * profile["long_term"][feature]
                        + (1 - self.long_term_weight) * value
                    )
                else:
                    profile.setdefault("long_term", {})[feature] = value
            profile.setdefault("short_term", deque(maxlen=self.history_length)).append(new_features)
            await self._update_profile_db(user_id, profile)
            self.user_profiles[user_id] = profile
            logger.info(f"Updated mimicry profile for {user_id}: {profile}")
            return profile

    async def adapt_response_style(self, response: str, user_id: str) -> str:
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = await self._get_profile(user_id)
        if not profile:
            return response
        combined_features = {}
        for feature in profile.get("long_term", {}):
            short_term_values = [p.get(feature, profile["long_term"][feature]) for p in profile.get("short_term", [])]
            short_term_avg = sum(short_term_values) / len(short_term_values) if short_term_values else profile["long_term"][feature]
            combined_features[feature] = (
                self.long_term_weight * profile["long_term"][feature]
                + self.short_term_weight * short_term_avg
            )
        if combined_features.get("positive_sentiment", 0.5) > 0.7:
            response = self._add_positive_sentiment(response)
        if combined_features.get("sentence_length", 10) > 15:
            response = self._increase_sentence_length(response)
        return response

    def _add_positive_sentiment(self, response: str) -> str:
        return response + " Je suis vraiment content que vous posiez cette question !"

    def _increase_sentence_length(self, response: str) -> str:
        return response + " De plus, il convient de noter que des détails supplémentaires peuvent être pertinents."