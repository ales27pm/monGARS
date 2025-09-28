import pytest

from monGARS.core.dynamic_response import AdaptiveResponseGenerator


class StubPersonalityEngine:
    def __init__(self, responses: list[dict[str, float]]) -> None:
        self._responses = responses
        self.call_count = 0
        self.last_user: str | None = None
        self.last_interactions: list[dict[str, str]] | None = None

    async def analyze_personality(
        self, user_id: str, interactions: list[dict[str, str]]
    ) -> dict[str, float]:
        self.last_user = user_id
        self.last_interactions = interactions
        index = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return dict(self._responses[index])


@pytest.mark.asyncio
async def test_personality_traits_reused_within_ttl() -> None:
    current_time = 0.0

    def now() -> float:
        return current_time

    engine = StubPersonalityEngine(
        responses=[{"formality": 0.8, "humor": 0.4, "enthusiasm": 0.9}]
    )
    generator = AdaptiveResponseGenerator(
        engine, cache_ttl_seconds=60, time_provider=now
    )
    interactions = [{"message": "Bonjour", "response": "Salut"}]

    first = await generator.get_personality_traits("user-1", interactions)
    second = await generator.get_personality_traits("user-1", interactions)

    assert engine.call_count == 1
    assert first == second
    assert engine.last_user == "user-1"


@pytest.mark.asyncio
async def test_personality_traits_refresh_after_ttl_expiry() -> None:
    current_time = 0.0

    def now() -> float:
        return current_time

    engine = StubPersonalityEngine(
        responses=[
            {"formality": 0.3, "humor": 0.5, "enthusiasm": 0.6},
            {"formality": 0.6, "humor": 0.5, "enthusiasm": 0.6},
        ]
    )
    generator = AdaptiveResponseGenerator(
        engine, cache_ttl_seconds=5, time_provider=now
    )
    interactions = [{"message": "Bonjour", "response": "Salut"}]

    first = await generator.get_personality_traits("user-2", interactions)
    current_time += 10
    second = await generator.get_personality_traits("user-2", interactions)

    assert engine.call_count == 2
    assert first != second


@pytest.mark.asyncio
async def test_personality_traits_refresh_on_interaction_change() -> None:
    current_time = 0.0

    def now() -> float:
        return current_time

    engine = StubPersonalityEngine(
        responses=[
            {"formality": 0.4, "humor": 0.4, "enthusiasm": 0.4},
            {"formality": 0.5, "humor": 0.4, "enthusiasm": 0.4},
        ]
    )
    generator = AdaptiveResponseGenerator(
        engine, cache_ttl_seconds=60, time_provider=now
    )

    interactions_initial = [{"message": "Salut", "response": "Bonjour"}]
    interactions_new = [{"message": "Bonsoir", "response": "Salut"}]

    await generator.get_personality_traits("user-3", interactions_initial)
    await generator.get_personality_traits("user-3", interactions_new)

    assert engine.call_count == 2
    assert engine.last_interactions == interactions_new
