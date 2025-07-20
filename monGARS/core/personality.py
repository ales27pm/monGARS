import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

try:  # prefer patched init_db in tests
    from init_db import UserPersonality, async_session_factory
except Exception:  # pragma: no cover - fallback for runtime use
    from monGARS.init_db import UserPersonality, async_session_factory

logger = logging.getLogger(__name__)


@dataclass
class PersonalityProfile:
    traits: dict
    interaction_style: dict
    context_preferences: dict
    adaptation_rate: float
    confidence: float


class PersonalityEngine:
    def __init__(self, session_factory=async_session_factory) -> None:
        self.user_profiles = defaultdict(lambda: self._generate_default_profile())
        self.learning_rate = 0.1
        self._lock = asyncio.Lock()
        self._session_factory = session_factory
        logger.info("PersonalityEngine initialized.")

    def _generate_default_profile(self) -> PersonalityProfile:
        default_traits = {
            "openness": random.uniform(0.4, 0.7),
            "conscientiousness": random.uniform(0.4, 0.7),
            "extraversion": random.uniform(0.4, 0.7),
            "agreeableness": random.uniform(0.4, 0.7),
            "neuroticism": random.uniform(0.4, 0.7),
        }
        default_style = {
            "formality": 0.5,
            "humor": 0.5,
            "enthusiasm": 0.5,
            "directness": 0.5,
        }
        default_preferences = {"technical": 0.5, "casual": 0.5, "professional": 0.5}
        return PersonalityProfile(
            default_traits, default_style, default_preferences, 0.1, 0.5
        )

    async def load_profile(self, user_id: str) -> PersonalityProfile:
        async with self._lock:
            try:
                async with self._session_factory() as session:
                    result = await session.execute(
                        select(UserPersonality).where(
                            UserPersonality.user_id == user_id
                        )
                    )
                    record = result.scalar_one_or_none()
                    if record:
                        profile = PersonalityProfile(
                            traits=record.traits,
                            interaction_style=record.interaction_style,
                            context_preferences=record.context_preferences,
                            adaptation_rate=record.adaptation_rate,
                            confidence=record.confidence,
                        )
                        self.user_profiles[user_id] = profile
            except Exception as exc:
                logger.exception("Failed to load profile for %s: %s", user_id, exc)
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._generate_default_profile()
            return self.user_profiles[user_id]

    async def save_profile(self, user_id: str) -> None:
        async with self._lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._generate_default_profile()
            profile = self.user_profiles[user_id]
            try:
                async with self._session_factory() as session:
                    await session.merge(
                        UserPersonality(
                            user_id=user_id,
                            traits=profile.traits,
                            interaction_style=profile.interaction_style,
                            context_preferences=profile.context_preferences,
                            adaptation_rate=profile.adaptation_rate,
                            confidence=profile.confidence,
                            last_updated=datetime.now(timezone.utc),
                        )
                    )
                    await session.commit()
            except Exception as exc:
                logger.exception("Failed to save profile for %s: %s", user_id, exc)

    async def analyze_personality(self, user_id: str, interactions: list) -> dict:
        if user_id not in self.user_profiles:
            await self.load_profile(user_id)
        return self.user_profiles[user_id].traits
