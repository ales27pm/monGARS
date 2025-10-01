from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
from collections import deque
from collections.abc import MutableMapping
from functools import lru_cache
from typing import Deque, TypedDict

from monGARS.config import get_settings

from .caching.tiered_cache import TieredCache
from .mimicry_lexicon import get_sentiment_lexicon
from .persistence import PersistenceRepository

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PROFILE_CACHE_PREFIX = "mimicry.profile."


class FeatureSnapshot(TypedDict, total=False):
    """Representation of the signals tracked for mimicry."""

    sentence_length: float
    positive_sentiment: float
    question_ratio: float
    exclamation_ratio: float


ProfileDict = MutableMapping[str, object]


@lru_cache(maxsize=1)
def _redaction_secret() -> bytes:
    settings = get_settings()
    secret = settings.SECRET_KEY or settings.app_name
    return secret.encode("utf-8")


def _redact_user(user_id: str | None) -> str:
    """Return a short, non-reversible token for logs."""

    if not user_id:
        return "anon"
    digest = hmac.new(_redaction_secret(), user_id.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:12]


def _tokenize(text: str) -> list[str]:
    """Return lowercase word tokens extracted from the provided text."""

    return _WORD_RE.findall(text.lower())


def _make_default_profile(history_length: int) -> ProfileDict:
    """Return a new default mimicry profile."""

    return {"long_term": {}, "short_term": deque(maxlen=history_length)}


class MimicryModule:
    """Adapt responses based on long- and short-term interaction patterns."""

    def __init__(
        self,
        long_term_weight: float = 0.9,
        short_term_weight: float = 0.1,
        history_length: int = 10,
        cache_ttl_seconds: float = 300.0,
        persistence_repo: PersistenceRepository | None = None,
        profile_cache: TieredCache | None = None,
    ) -> None:
        """Create a mimicry module with configurable weighting."""

        self.long_term_weight = long_term_weight
        self.short_term_weight = short_term_weight
        self.history_length = history_length
        self.cache_ttl_seconds = cache_ttl_seconds
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._persistence = persistence_repo or PersistenceRepository()
        self._profile_cache = profile_cache or TieredCache()
        self.positive_words, self.negative_words = get_sentiment_lexicon()

    async def _cache_profile(self, user_id: str, profile: ProfileDict) -> None:
        """Store profile locally with a refreshed TTL."""

        if not user_id:
            return
        ttl = int(self.cache_ttl_seconds)
        await self._profile_cache.set(
            f"{_PROFILE_CACHE_PREFIX}{user_id}",
            profile,
            ttl=ttl if ttl > 0 else None,
        )

    def _normalise_profile(self, profile: ProfileDict) -> ProfileDict:
        short_term = profile.get("short_term")
        profile["short_term"] = deque(short_term or [], maxlen=self.history_length)
        profile.setdefault("long_term", {})
        return profile

    async def _load_profile_from_storage(self, user_id: str) -> ProfileDict:
        """Retrieve a stored profile or build a default one from persistence."""

        try:
            record = await self._persistence.get_user_preferences(user_id)
        except Exception as exc:
            logger.error(
                "mimicry.get_profile.error",
                exc_info=True,
                extra={"user": _redact_user(user_id), "error": str(exc)},
            )
            return _make_default_profile(self.history_length)

        if record and getattr(record, "interaction_style", None):
            stored: ProfileDict = dict(record.interaction_style)
            return self._normalise_profile(stored)
        return _make_default_profile(self.history_length)

    async def _get_profile(
        self, user_id: str, *, acquire_lock: bool = True
    ) -> ProfileDict:
        """Retrieve a stored profile or build a default one."""

        if not user_id:
            return _make_default_profile(self.history_length)

        if acquire_lock:
            lock = self._user_locks.setdefault(user_id, asyncio.Lock())
            async with lock:
                return await self._get_profile(user_id, acquire_lock=False)

        cache_key = f"{_PROFILE_CACHE_PREFIX}{user_id}"
        cached_profile = await self._profile_cache.get(cache_key)
        if cached_profile:
            return self._normalise_profile(cached_profile)

        profile = await self._load_profile_from_storage(user_id)
        await self._cache_profile(user_id, profile)
        return profile

    async def _update_profile_db(self, user_id: str, profile: ProfileDict) -> None:
        """Persist a profile update back to the database."""

        try:
            payload: ProfileDict = dict(profile)
            payload["short_term"] = list(profile.get("short_term", []))
            await self._persistence.upsert_user_preferences(
                user_id=user_id,
                interaction_style=payload,
            )
            logger.info(
                "mimicry.profile_db.updated",
                extra={"user": _redact_user(user_id)},
            )
        except Exception as exc:
            logger.error(
                "mimicry.profile_db.error",
                exc_info=True,
                extra={"user": _redact_user(user_id), "error": str(exc)},
            )
            raise

    async def update_profile(self, user_id: str, interaction: dict) -> dict:
        """Blend new interaction signals into the user profile."""

        lock = self._user_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            profile = await self._get_profile(user_id, acquire_lock=False)
            new_features = self._extract_features(interaction)
            long_term: MutableMapping[str, float] = profile.setdefault("long_term", {})
            for feature, value in new_features.items():
                if value is None:
                    continue
                if feature in profile.get("long_term", {}):
                    long_term[feature] = (
                        self.long_term_weight * float(long_term[feature])
                        + (1 - self.long_term_weight) * value
                    )
                else:
                    long_term[feature] = value
            short_term: Deque[FeatureSnapshot] = profile.setdefault(
                "short_term", deque(maxlen=self.history_length)
            )
            short_term.append(new_features)
            if user_id:
                await self._update_profile_db(user_id, profile)
                await self._cache_profile(user_id, profile)
            logger.info(
                "mimicry.profile.updated",
                extra={
                    "user": _redact_user(user_id),
                    "long_term_keys": list(long_term.keys()),
                    "short_term_len": len(short_term),
                },
            )
            return profile

    def _extract_features(self, interaction: dict) -> FeatureSnapshot:
        """Extract measurable features from the user interaction payload."""

        message = str(interaction.get("message", ""))
        response = str(interaction.get("response", ""))
        features: FeatureSnapshot = FeatureSnapshot(
            sentence_length=float(self._count_words(message)),
            positive_sentiment=self._analyze_sentiment(response),
            question_ratio=self._punctuation_ratio(message, "?"),
            exclamation_ratio=self._punctuation_ratio(message, "!"),
        )
        return features

    def _count_words(self, text: str) -> int:
        """Return the number of words detected in a text snippet."""

        return len(_tokenize(text))

    def _analyze_sentiment(self, text: str) -> float:
        """Estimate sentiment score between 0 (negative) and 1 (positive)."""

        tokens = _tokenize(text)
        scored = [
            1 if token in self.positive_words else -1
            for token in tokens
            if token in self.positive_words or token in self.negative_words
        ]
        if not scored:
            return 0.5
        total = len(scored)
        normalized = (sum(scored) + total) / (2 * total)
        return max(0.0, min(1.0, normalized))

    def _punctuation_ratio(self, text: str, marker: str) -> float:
        """Return the ratio of punctuation marks to total sentences."""

        if not text:
            return 0.0
        sentences = [
            segment.strip() for segment in re.split(r"[.!?]+", text) if segment
        ]
        if not sentences:
            return 0.0
        marker_count = text.count(marker)
        return marker_count / len(sentences)

    async def adapt_response_style(self, response: str, user_id: str) -> str:
        """Shape a response string using the stored interaction profile."""

        profile = await self._get_profile(user_id)
        if not profile:
            return response
        combined_features: dict[str, float] = {}
        long_term = profile.get("long_term", {})
        short_term_list: list[FeatureSnapshot] = list(profile.get("short_term", []))
        for feature, long_val in long_term.items():
            short_term_values = [p.get(feature, long_val) for p in short_term_list]
            short_term_avg = (
                sum(short_term_values) / len(short_term_values)
                if short_term_values
                else long_val
            )
            combined_features[feature] = (
                self.long_term_weight * long_val
                + self.short_term_weight * short_term_avg
            )
        if combined_features.get("positive_sentiment", 0.5) > 0.7:
            response = self._add_positive_sentiment(response)
        elif combined_features.get("positive_sentiment", 0.5) < 0.3:
            response = self._add_supportive_sentiment(response)
        if combined_features.get("sentence_length", 10) > 15:
            response = self._increase_sentence_length(response)
        if combined_features.get("question_ratio", 0.0) > 0.3:
            response = self._mirror_question_style(response)
        if combined_features.get("exclamation_ratio", 0.0) > 0.25:
            response = self._mirror_excitement(response)
        return response.strip()

    def _add_positive_sentiment(self, response: str) -> str:
        """Add a friendly reinforcement to the response text."""

        if response.endswith("!"):
            return (
                f"{response} Je suis vraiment content que vous posiez cette question !"
            )
        return f"{response} Je suis vraiment content que vous posiez cette question !"

    def _add_supportive_sentiment(self, response: str) -> str:
        """Add an empathetic follow-up when the user expresses negative sentiment."""

        return (
            f"{response} Je comprends que la situation puisse être difficile, "
            "restons concentrés sur des solutions concrètes."
        )

    def _increase_sentence_length(self, response: str) -> str:
        """Append clarifying detail to extend the response length."""

        return (
            response
            + " De plus, il convient de noter que des détails supplémentaires peuvent être pertinents."
        )

    def _mirror_question_style(self, response: str) -> str:
        """Encourage dialogue when the user tends to ask many questions."""

        if response.strip().endswith("?"):
            return response
        return (
            response + " Souhaitez-vous que j'approfondisse un aspect en particulier ?"
        )

    def _mirror_excitement(self, response: str) -> str:
        """Match enthusiastic tones detected in the conversation."""

        if response.endswith("!!"):
            return response
        return response + " C'est enthousiasmant de pouvoir partager cela avec vous !"
