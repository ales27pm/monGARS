import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass

# TODO: add database-backed persistence in future iteration

logger = logging.getLogger(__name__)


@dataclass
class PersonalityProfile:
    traits: dict
    interaction_style: dict
    context_preferences: dict
    adaptation_rate: float
    confidence: float


class PersonalityEngine:
    def __init__(self):
        self.user_profiles = defaultdict(lambda: self._generate_default_profile())
        self.learning_rate = 0.1
        self._lock = asyncio.Lock()
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

    async def analyze_personality(self, user_id: str, interactions: list) -> dict:
        return self.user_profiles[user_id].traits
