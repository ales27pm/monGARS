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

    assert profile["long_term"]["sentence_length"] == 7
    assert profile["long_term"]["positive_sentiment"] > 0.5
    last_entry = profile["short_term"][-1]
    assert last_entry["sentence_length"] == 7
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

    assert profile["long_term"]["sentence_length"] == 7
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

    assert profile["long_term"]["sentence_length"] == 0
    assert profile["long_term"]["positive_sentiment"] == pytest.approx(0.5)
    assert profile["short_term"][-1]["sentence_length"] == 0
