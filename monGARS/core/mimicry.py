import asyncio
import logging
import string
from collections import deque

from sqlalchemy import select, update

from ..init_db import UserPreferences, async_session_factory

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation + "«»“”‘’…")
POSITIVE_WORDS = {
    "heureux",
    "heureuse",
    "ravi",
    "ravie",
    "content",
    "contente",
    "excellent",
    "excellente",
    "fantastique",
    "formidable",
    "super",
    "merci",
    "satisfait",
    "satisfaite",
    "positif",
    "positive",
    "agréable",
    "brillant",
    "génial",
}
NEGATIVE_WORDS = {
    "triste",
    "furieux",
    "furieuse",
    "mauvais",
    "mauvaise",
    "terrible",
    "horrible",
    "déçu",
    "déçue",
    "problème",
    "problèmes",
    "mécontent",
    "mécontente",
    "négatif",
    "négative",
    "inquiet",
    "inquiète",
    "fâché",
    "fâchée",
}

logger = logging.getLogger(__name__)


def _make_default_profile(history_length: int) -> dict:
    """Return a new default mimicry profile."""

    return {"long_term": {}, "short_term": deque(maxlen=history_length)}


class MimicryModule:
    """Adapt responses based on long- and short-term interaction patterns."""

    def __init__(
        self,
        long_term_weight: float = 0.9,
        short_term_weight: float = 0.1,
        history_length: int = 10,
    ) -> None:
        """Create a mimicry module with configurable weighting."""

        self.long_term_weight = long_term_weight
        self.short_term_weight = short_term_weight
        self.history_length = history_length
        self.user_profiles: dict[str, dict] = {}
        self._user_locks: dict[str, asyncio.Lock] = {}

    async def _get_profile(self, user_id: str) -> dict:
        """Retrieve a stored profile or build a default one."""

        async with async_session_factory() as session:
            try:
                result = await session.execute(
                    select(UserPreferences).where(UserPreferences.user_id == user_id)
                )
                user_preferences = result.scalars().first()
                if user_preferences and user_preferences.interaction_style:
                    stored = user_preferences.interaction_style
                    short_term = stored.get("short_term", [])
                    stored["short_term"] = deque(short_term, maxlen=self.history_length)
                    return stored
                return _make_default_profile(self.history_length)
            except Exception as e:
                logger.error(
                    "mimicry.get_profile.error",
                    exc_info=True,
                    extra={"user_id": user_id, "error": str(e)},
                )
                return _make_default_profile(self.history_length)

    async def _update_profile_db(self, user_id: str, profile: dict) -> None:
        """Persist a profile update back to the database."""

        async with async_session_factory() as session:
            try:
                payload = dict(profile)
                payload["short_term"] = list(profile.get("short_term", []))
                stmt = (
                    update(UserPreferences)
                    .where(UserPreferences.user_id == user_id)
                    .values(interaction_style=payload)
                )
                result = await session.execute(stmt)
                if not result.rowcount:
                    exists = await session.execute(
                        select(UserPreferences.user_id).where(
                            UserPreferences.user_id == user_id
                        )
                    )
                    if exists.scalar_one_or_none() is None:
                        session.add(
                            UserPreferences(
                                user_id=user_id,
                                interaction_style=payload,
                                preferred_topics={},
                            )
                        )
                await session.commit()
                logger.info(
                    "mimicry.profile_db.updated",
                    extra={"user_id": user_id},
                )
            except Exception as e:
                await session.rollback()
                logger.error(
                    "mimicry.profile_db.error",
                    exc_info=True,
                    extra={"user_id": user_id, "error": str(e)},
                )
                raise

    async def update_profile(self, user_id: str, interaction: dict) -> dict:
        """Blend new interaction signals into the user profile."""

        lock = self._user_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            profile = self.user_profiles.get(user_id) or await self._get_profile(
                user_id
            )
            new_features = {
                "sentence_length": self._count_words(interaction.get("message", "")),
                "positive_sentiment": self._analyze_sentiment(
                    interaction.get("response", "")
                ),
            }
            for feature, value in new_features.items():
                if feature in profile.get("long_term", {}):
                    profile["long_term"][feature] = (
                        self.long_term_weight * profile["long_term"][feature]
                        + (1 - self.long_term_weight) * value
                    )
                else:
                    profile.setdefault("long_term", {})[feature] = value
            profile.setdefault("short_term", deque(maxlen=self.history_length)).append(
                new_features
            )
            await self._update_profile_db(user_id, profile)
            self.user_profiles[user_id] = profile
            logger.info(
                "mimicry.profile.updated",
                extra={
                    "user_id": user_id,
                    "long_term_keys": list(profile.get("long_term", {}).keys()),
                    "short_term_len": len(profile.get("short_term", [])),
                },
            )
            return profile

    def _count_words(self, text: str) -> int:
        """Return the number of words detected in a text snippet."""

        if not text.strip():
            return 0
        tokens = [
            token.translate(PUNCTUATION_TABLE)
            for token in text.split()
            if token.translate(PUNCTUATION_TABLE).strip()
        ]
        return len(tokens)

    def _analyze_sentiment(self, text: str) -> float:
        """Estimate sentiment score between 0 (negative) and 1 (positive)."""

        if not text.strip():
            return 0.5
        cleaned_tokens = [
            token.translate(PUNCTUATION_TABLE).lower()
            for token in text.split()
            if token.translate(PUNCTUATION_TABLE).strip()
        ]
        scored_tokens = [
            token
            for token in cleaned_tokens
            if token in POSITIVE_WORDS or token in NEGATIVE_WORDS
        ]
        if not scored_tokens:
            return 0.5
        score = sum(1 if token in POSITIVE_WORDS else -1 for token in scored_tokens)
        total = len(scored_tokens)
        normalized = (score + total) / (2 * total)
        return max(0.0, min(1.0, normalized))

    async def adapt_response_style(self, response: str, user_id: str) -> str:
        """Shape a response string using the stored interaction profile."""

        profile = self.user_profiles.get(user_id) or await self._get_profile(user_id)
        if not profile:
            return response
        combined_features = {}
        for feature in profile.get("long_term", {}):
            short_term_values = [
                p.get(feature, profile["long_term"][feature])
                for p in profile.get("short_term", [])
            ]
            short_term_avg = (
                sum(short_term_values) / len(short_term_values)
                if short_term_values
                else profile["long_term"][feature]
            )
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
        """Add a friendly reinforcement to the response text."""

        return f"{response} Je suis vraiment content que vous posiez cette question !"

    def _increase_sentence_length(self, response: str) -> str:
        """Append clarifying detail to extend the response length."""

        return (
            response
            + " De plus, il convient de noter que des détails supplémentaires peuvent être pertinents."
        )
