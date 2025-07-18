import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from init_db import UserPersonality, async_session_factory

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
        async with self._session_factory() as session:
            result = await session.execute(
                select(UserPersonality).where(UserPersonality.user_id == user_id)
            )
            record = result.scalar_one_or_none()
            if record:
                profile = PersonalityProfile(
                    traits=record.traits,
                    interaction_style=record.interaction_style,
                    context_preferences=getattr(record, "context_preferences", {}),
                    adaptation_rate=getattr(record, "adaptation_rate", 0.1),
                    confidence=getattr(record, "confidence", 0.5),
                )
                self.user_profiles[user_id] = profile
        return self.user_profiles[user_id]

    async def save_profile(self, user_id: str) -> None:
        profile = self.user_profiles[user_id]
        async with self._session_factory() as session:
            result = await session.execute(
                select(UserPersonality).where(UserPersonality.user_id == user_id)
            )
            record = result.scalar_one_or_none()
            if record:
                record.traits = profile.traits
                record.interaction_style = profile.interaction_style
                record.context_preferences = profile.context_preferences
                record.adaptation_rate = profile.adaptation_rate
                record.confidence = profile.confidence
                record.last_updated = datetime.utcnow()
            else:
                session.add(
                    UserPersonality(
                        user_id=user_id,
                        traits=profile.traits,
                        interaction_style=profile.interaction_style,
                        context_preferences=profile.context_preferences,
                        adaptation_rate=profile.adaptation_rate,
                        confidence=profile.confidence,
                        last_updated=datetime.utcnow(),
                    )
                )
            await session.commit()

    async def analyze_personality(self, user_id: str, interactions: list) -> dict:
        if user_id not in self.user_profiles:
            await self.load_profile(user_id)
        return self.user_profiles[user_id].traits
