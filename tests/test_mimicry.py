from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from monGARS import config
from monGARS.core.mimicry import MimicryModule


@dataclass
class _InMemoryCache:
    store: dict[str, dict]

    async def get(self, key: str) -> dict | None:
        return self.store.get(key)

    async def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        self.store[key] = value


class _InMemoryPreferences:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    async def get_user_preferences(self, user_id: str) -> SimpleNamespace | None:
        if user_id not in self._data:
            return None
        return SimpleNamespace(user_id=user_id, interaction_style=self._data[user_id])

    async def upsert_user_preferences(
        self,
        *,
        user_id: str,
        interaction_style: dict,
        preferred_topics: dict | None = None,
    ) -> None:
        self._data[user_id] = interaction_style


@pytest.fixture
def mimicry_module(monkeypatch: pytest.MonkeyPatch) -> MimicryModule:
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    config.get_settings.cache_clear()
    cache = _InMemoryCache(store={})
    preferences = _InMemoryPreferences()
    module = MimicryModule(persistence_repo=preferences, profile_cache=cache)
    yield module
    config.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_update_profile_uses_message_and_response(
    mimicry_module: MimicryModule,
) -> None:
    module = mimicry_module

    interaction = {
        "message": "Bonjour et merci pour votre aide précieuse",
        "response": "Je suis ravi de vous assister aujourd'hui.",
    }

    profile = await module.update_profile("user-positive", interaction)

    assert profile["long_term"]["sentence_length"] == pytest.approx(7.0)
    assert profile["long_term"]["positive_sentiment"] > 0.5
    last_entry = profile["short_term"][-1]
    assert last_entry["sentence_length"] == pytest.approx(7.0)
    assert last_entry["positive_sentiment"] > 0.5


@pytest.mark.asyncio
async def test_update_profile_detects_negative_sentiment(
    mimicry_module: MimicryModule,
) -> None:
    module = mimicry_module

    interaction = {
        "message": "Je suis contrarié par la situation actuelle",
        "response": "C'est terrible et mauvais pour tout le monde.",
    }

    profile = await module.update_profile("user-negative", interaction)

    assert profile["long_term"]["sentence_length"] == pytest.approx(7.0)
    assert profile["long_term"]["positive_sentiment"] < 0.5


@pytest.mark.asyncio
async def test_update_profile_neutral_sentiment(
    mimicry_module: MimicryModule,
) -> None:
    module = mimicry_module

    interaction = {
        "message": "La météo est acceptable aujourd'hui.",
        "response": "Oui, la situation est simplement normale.",
    }

    profile = await module.update_profile("user-neutral", interaction)

    assert profile["long_term"]["positive_sentiment"] == pytest.approx(0.5)
    assert profile["short_term"][-1]["positive_sentiment"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_update_profile_empty_input(
    mimicry_module: MimicryModule,
) -> None:
    module = mimicry_module

    interaction = {"message": "", "response": ""}

    profile = await module.update_profile("user-empty", interaction)

    assert profile["long_term"]["sentence_length"] == pytest.approx(0.0)
    assert profile["long_term"]["positive_sentiment"] == pytest.approx(0.5)
    assert profile["short_term"][-1]["sentence_length"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_update_profile_tracks_question_and_exclamation(
    mimicry_module: MimicryModule,
) -> None:
    module = mimicry_module

    interaction = {
        "message": "Pouvez-vous expliquer ce point ? J'ai encore une question !",
        "response": "Je vais détailler la marche à suivre.",
    }

    profile = await module.update_profile("user-curious", interaction)

    assert profile["long_term"]["question_ratio"] > 0.4
    assert profile["long_term"]["exclamation_ratio"] > 0.4


@pytest.mark.asyncio
async def test_adapt_response_style_mirrors_questions(
    mimicry_module: MimicryModule,
) -> None:
    module = mimicry_module

    interaction = {
        "message": "Pourquoi cela se produit-il ? Pouvez-vous préciser ?",
        "response": "Je vais regarder cela.",
    }

    await module.update_profile("user-questions", interaction)

    adapted = await module.adapt_response_style(
        "Voici ce que je vois.", "user-questions"
    )

    assert adapted.endswith("?")


@pytest.mark.asyncio
async def test_adapt_response_style_supports_negative_sentiment(
    mimicry_module: MimicryModule,
) -> None:
    module = mimicry_module

    interaction = {
        "message": "Je suis profondément insatisfait.",
        "response": "C'est un vrai problème.",
    }

    await module.update_profile("user-sad", interaction)

    adapted = await module.adapt_response_style(
        "Je vais analyser la situation.", "user-sad"
    )

    assert "difficile" in adapted
    assert adapted.endswith((".", "!"))
