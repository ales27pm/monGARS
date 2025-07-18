import importlib
import sys
import types

import pytest


class FakeSession:
    def __init__(self, record=None):
        self.record = record
        self.merged = None
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def execute(self, *args, **kwargs):
        class Result:
            def __init__(self, record):
                self._record = record

            def scalar_one_or_none(self):
                return self._record

        return Result(self.record)

    async def merge(self, obj):
        self.merged = obj
        self.record = obj
        return obj

    async def commit(self):
        self.committed = True


def fake_factory(record=None):
    session = FakeSession(record)

    def factory():
        return session

    factory.session = session
    return factory


def load_engine(factory, monkeypatch):
    class UP:
        user_id = None

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_init = types.SimpleNamespace(UserPersonality=UP, async_session_factory=factory)
    monkeypatch.setitem(sys.modules, "init_db", fake_init)
    module = importlib.import_module("monGARS.core.personality")
    importlib.reload(module)

    class FakeSelect:
        def where(self, *a, **k):
            return self

    monkeypatch.setattr(module, "select", lambda *a, **k: FakeSelect())
    return module.PersonalityEngine(session_factory=factory)


@pytest.mark.asyncio
async def test_save_new_profile_adds_record(monkeypatch):
    factory = fake_factory()
    engine = load_engine(factory, monkeypatch)
    await engine.save_profile("u1")
    assert factory.session.merged is not None
    assert factory.session.committed


@pytest.mark.asyncio
async def test_save_existing_profile_updates_record(monkeypatch):
    existing = types.SimpleNamespace(
        user_id="u1",
        traits={"t": 0},
        interaction_style={"s": 0},
        context_preferences={},
        adaptation_rate=0.1,
        confidence=0.5,
    )
    factory = fake_factory(existing)
    engine = load_engine(factory, monkeypatch)
    engine.user_profiles["u1"].traits["new"] = 1
    await engine.save_profile("u1")
    assert factory.session.merged.traits["new"] == 1
    assert factory.session.committed


@pytest.mark.asyncio
async def test_load_profile_returns_db_record(monkeypatch):
    db_profile = types.SimpleNamespace(
        traits={"t": 1},
        interaction_style={"s": 1},
        context_preferences={"c": 1},
        adaptation_rate=0.2,
        confidence=0.9,
    )
    factory = fake_factory(db_profile)
    engine = load_engine(factory, monkeypatch)
    profile = await engine.load_profile("u1")
    assert profile.traits == {"t": 1}
    assert engine.user_profiles["u1"].interaction_style == {"s": 1}
