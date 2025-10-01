from collections import deque

import pytest

from monGARS.core.mimicry import MimicryModule


def _setup_in_memory_profile(
    module: MimicryModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_get_profile(user_id: str) -> dict:
        return module.user_profiles.get(user_id) or {
            "long_term": {},
            "short_term": deque(maxlen=module.history_length),
        }

    async def fake_update_profile_db(user_id: str, profile: dict) -> None:
        return None

    monkeypatch.setattr(module, "_get_profile", fake_get_profile)
    monkeypatch.setattr(module, "_update_profile_db", fake_update_profile_db)


@pytest.mark.asyncio
async def test_update_profile_uses_message_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MimicryModule()

    _setup_in_memory_profile(module, monkeypatch)

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MimicryModule()

    _setup_in_memory_profile(module, monkeypatch)

    interaction = {
        "message": "Je suis contrarié par la situation actuelle",
        "response": "C'est terrible et mauvais pour tout le monde.",
    }

    profile = await module.update_profile("user-negative", interaction)

    assert profile["long_term"]["sentence_length"] == pytest.approx(7.0)
    assert profile["long_term"]["positive_sentiment"] < 0.5


@pytest.mark.asyncio
async def test_update_profile_neutral_sentiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MimicryModule()
    _setup_in_memory_profile(module, monkeypatch)

    interaction = {
        "message": "La météo est acceptable aujourd'hui.",
        "response": "Oui, la situation est simplement normale.",
    }

    profile = await module.update_profile("user-neutral", interaction)

    assert profile["long_term"]["positive_sentiment"] == pytest.approx(0.5)
    assert profile["short_term"][-1]["positive_sentiment"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_update_profile_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MimicryModule()
    _setup_in_memory_profile(module, monkeypatch)

    interaction = {"message": "", "response": ""}

    profile = await module.update_profile("user-empty", interaction)

    assert profile["long_term"]["sentence_length"] == pytest.approx(0.0)
    assert profile["long_term"]["positive_sentiment"] == pytest.approx(0.5)
    assert profile["short_term"][-1]["sentence_length"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_update_profile_tracks_question_and_exclamation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MimicryModule()
    _setup_in_memory_profile(module, monkeypatch)

    interaction = {
        "message": "Pouvez-vous expliquer ce point ? J'ai encore une question !",
        "response": "Je vais détailler la marche à suivre.",
    }

    profile = await module.update_profile("user-curious", interaction)

    assert profile["long_term"]["question_ratio"] > 0.4
    assert profile["long_term"]["exclamation_ratio"] > 0.4


@pytest.mark.asyncio
async def test_adapt_response_style_mirrors_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MimicryModule()
    _setup_in_memory_profile(module, monkeypatch)

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MimicryModule()
    _setup_in_memory_profile(module, monkeypatch)

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
