import asyncio

import pytest

from monGARS.core.dynamic_response import AdaptiveResponseGenerator


class StubStyleTuner:
    def __init__(self) -> None:
        self.applied: list[tuple[str, str, dict[str, float]]] = []

    async def estimate_personality(
        self, user_id: str, interactions: list[dict[str, str]]
    ) -> None:
        raise NotImplementedError

    def apply_style(
        self,
        user_id: str,
        base_text: str,
        personality: dict[str, float] | None,
    ) -> str:
        self.applied.append((user_id, base_text, personality or {}))
        return f"{base_text}::{user_id}"


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


class BlockingPersonalityEngine(StubPersonalityEngine):
    def __init__(
        self,
        responses: list[dict[str, float]],
        gate: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        super().__init__(responses)
        self._gate = gate
        self._release = release

    async def analyze_personality(
        self, user_id: str, interactions: list[dict[str, str]]
    ) -> dict[str, float]:
        self._gate.set()
        await self._release.wait()
        return await super().analyze_personality(user_id, interactions)


class EngineWithTuner:
    def __init__(self, style_tuner: StubStyleTuner) -> None:
        self.style_tuner = style_tuner

    async def analyze_personality(
        self, user_id: str, interactions: list[dict[str, str]]
    ) -> dict[str, float]:
        return {"formality": 0.5}

    def set_style_tuner(self, style_tuner: StubStyleTuner) -> None:
        self.style_tuner = style_tuner


@pytest.mark.asyncio
async def test_personality_traits_reused_within_ttl() -> None:
    current_time = 0.0

    def now() -> float:
        return current_time

    engine = StubPersonalityEngine(
        responses=[{"formality": 0.8, "humor": 0.4, "enthusiasm": 0.9}]
    )
    generator = AdaptiveResponseGenerator(
        engine,
        cache_ttl_seconds=60,
        time_provider=now,
        style_tuner=StubStyleTuner(),
    )
    interactions = [{"message": "Bonjour", "response": "Salut"}]

    first = await generator.get_personality_traits("user-1", interactions)
    second = await generator.get_personality_traits("user-1", interactions)

    assert engine.call_count == 1
    assert first == second
    assert engine.last_user == "user-1"


def test_generator_reuses_engine_style_tuner() -> None:
    style_tuner = StubStyleTuner()
    engine = EngineWithTuner(style_tuner)
    generator = AdaptiveResponseGenerator(personality_engine=engine)

    result = generator.generate_adaptive_response(
        "Salut", {"formality": 0.7}, user_id="user-99"
    )

    assert result == "Salut::user-99"
    assert style_tuner.applied == [("user-99", "Salut", {"formality": 0.7})]


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
        engine,
        cache_ttl_seconds=5,
        time_provider=now,
        style_tuner=StubStyleTuner(),
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
        engine,
        cache_ttl_seconds=60,
        time_provider=now,
        style_tuner=StubStyleTuner(),
    )

    interactions_initial = [{"message": "Salut", "response": "Bonjour"}]
    interactions_new = [{"message": "Bonsoir", "response": "Salut"}]

    await generator.get_personality_traits("user-3", interactions_initial)
    await generator.get_personality_traits("user-3", interactions_new)

    assert engine.call_count == 2
    assert engine.last_interactions == interactions_new


@pytest.mark.asyncio
async def test_personality_traits_cached_indefinitely_with_negative_ttl() -> None:
    responses = [
        {"openness": 0.1, "conscientiousness": 0.2},
        {"openness": 0.9, "conscientiousness": 0.8},
    ]
    engine = StubPersonalityEngine(responses)
    time_holder = {"now": 100.0}

    def fake_time() -> float:
        return time_holder["now"]

    generator = AdaptiveResponseGenerator(
        personality_engine=engine,
        cache_ttl_seconds=-1,
        time_provider=fake_time,
        style_tuner=StubStyleTuner(),
    )

    user_id = "user1"
    interactions = [{"text": "hello"}]

    traits1 = await generator.get_personality_traits(user_id, interactions)
    assert traits1 == responses[0]
    assert engine.call_count == 1

    time_holder["now"] += 1_000_000

    traits2 = await generator.get_personality_traits(user_id, interactions)
    assert traits2 == responses[0]
    assert engine.call_count == 1

    new_interactions = [{"text": "hi"}]
    traits3 = await generator.get_personality_traits(user_id, new_interactions)
    assert traits3 == responses[0]
    assert engine.call_count == 1

    traits4 = await generator.get_personality_traits("user2", interactions)
    assert traits4 == responses[1]
    assert engine.call_count == 2


@pytest.mark.asyncio
async def test_personality_traits_shared_across_concurrent_calls() -> None:
    gate = asyncio.Event()
    release = asyncio.Event()
    engine = BlockingPersonalityEngine(
        responses=[{"formality": 0.6, "humor": 0.4, "enthusiasm": 0.5}],
        gate=gate,
        release=release,
    )
    generator = AdaptiveResponseGenerator(
        engine, cache_ttl_seconds=60, style_tuner=StubStyleTuner()
    )
    interactions = [{"message": "Bonjour"}]

    first_task = asyncio.create_task(
        generator.get_personality_traits("user-4", interactions)
    )
    await gate.wait()

    second_task = asyncio.create_task(
        generator.get_personality_traits("user-4", interactions)
    )

    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(first_task, second_task)

    assert engine.call_count == 1
    assert results[0] == results[1]


def test_generate_adaptive_response_delegates_to_style_tuner() -> None:
    style_tuner = StubStyleTuner()
    generator = AdaptiveResponseGenerator(cache_ttl_seconds=0, style_tuner=style_tuner)
    personality = {
        "formality": " 0.8 ",
        "humor": "not-a-number",
        "enthusiasm": None,
    }

    adapted = generator.generate_adaptive_response(
        "Salut tu", personality, user_id="abc"
    )

    assert adapted == "Salut tu::abc"
    assert style_tuner.applied == [
        (
            "abc",
            "Salut tu",
            {"formality": " 0.8 ", "humor": "not-a-number", "enthusiasm": None},
        )
    ]
