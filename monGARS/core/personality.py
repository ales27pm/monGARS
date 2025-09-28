from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select

try:  # prefer patched init_db in tests
    from init_db import UserPersonality, async_session_factory
except Exception:  # pragma: no cover - fallback for runtime use
    from monGARS.init_db import UserPersonality, async_session_factory

from monGARS.core.style_finetuning import StyleAnalysis, StyleFineTuner

logger = logging.getLogger(__name__)


@dataclass
class PersonalityProfile:
    traits: dict[str, float]
    interaction_style: dict[str, float]
    context_preferences: dict[str, float]
    adaptation_rate: float
    confidence: float


class PersonalityEngine:
    """Persist and evolve user personalities using learned style adapters."""

    def __init__(
        self,
        session_factory=async_session_factory,
        *,
        style_tuner: StyleFineTuner | None = None,
    ) -> None:
        self.user_profiles: defaultdict[str, PersonalityProfile] = defaultdict(
            lambda: self._generate_default_profile()
        )
        self.learning_rate = 0.05
        self._lock = asyncio.Lock()
        self._session_factory = session_factory
        self._style_tuner = style_tuner or StyleFineTuner()
        logger.info("PersonalityEngine initialized with style fine-tuning module.")

    def _generate_default_profile(self) -> PersonalityProfile:
        default_traits = {
            "openness": 0.55,
            "conscientiousness": 0.55,
            "extraversion": 0.55,
            "agreeableness": 0.55,
            "neuroticism": 0.45,
        }
        default_style = {
            "formality": 0.5,
            "humor": 0.5,
            "enthusiasm": 0.5,
            "directness": 0.5,
        }
        default_preferences = {
            "technical": 0.5,
            "casual": 0.5,
            "professional": 0.5,
        }
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
                    if record := result.scalar_one_or_none():
                        profile = PersonalityProfile(
                            traits=record.traits,
                            interaction_style=record.interaction_style,
                            context_preferences=record.context_preferences,
                            adaptation_rate=record.adaptation_rate,
                            confidence=record.confidence,
                        )
                        self.user_profiles[user_id] = profile
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to load profile for %s: %s", user_id, exc)
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._generate_default_profile()
            return self.user_profiles[user_id]

    @property
    def style_tuner(self) -> StyleFineTuner:
        return self._style_tuner

    def set_style_tuner(self, style_tuner: StyleFineTuner) -> None:
        self._style_tuner = style_tuner

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
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to save profile for %s: %s", user_id, exc)

    async def analyze_personality(
        self, user_id: str, interactions: Iterable[dict[str, str]]
    ) -> dict[str, float]:
        """Update the user's profile based on recent interactions."""

        profile = await self.load_profile(user_id)
        interactions_list = list(interactions)
        try:
            analysis: StyleAnalysis = await self._style_tuner.estimate_personality(
                user_id, interactions_list
            )
        except Exception as exc:  # pragma: no cover - graceful degradation
            logger.exception("Style analysis failed for %s: %s", user_id, exc)
            return profile.traits

        self._blend_traits(profile, analysis)
        self._schedule_persistence(user_id)
        return profile.traits

    def _blend_traits(
        self, profile: PersonalityProfile, analysis: StyleAnalysis
    ) -> None:
        for trait, value in analysis.traits.items():
            current = profile.traits.get(trait, 0.5)
            profile.traits[trait] = self._update_trait(current, value)

        for style_dim, value in analysis.style.items():
            current = profile.interaction_style.get(style_dim, 0.5)
            profile.interaction_style[style_dim] = self._update_trait(current, value)

        for context_key, value in analysis.context_preferences.items():
            current = profile.context_preferences.get(context_key, 0.5)
            profile.context_preferences[context_key] = self._update_trait(
                current, value
            )

        profile.confidence = max(profile.confidence, analysis.confidence)

    def _schedule_persistence(self, user_id: str) -> None:
        task = asyncio.create_task(self.save_profile(user_id))
        task.add_done_callback(self._handle_persistence_error)

    @staticmethod
    def _handle_persistence_error(task: asyncio.Task[None]) -> None:
        if exception := task.exception():  # pragma: no cover - logged once
            logger.error("Failed to persist personality profile: %s", exception)

    def _update_trait(self, current_value: float, new_value: float) -> float:
        """Nudge the current trait toward ``new_value`` using the learning rate."""

        return current_value + self.learning_rate * (new_value - current_value)
