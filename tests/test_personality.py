import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakeSession:
    def __init__(self, record=None):
        self.record = record
        self.added = None
        self.committed = False
        self.rolled_back = False

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

    def add(self, obj):
        self.added = obj
        self.record = obj

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


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
    style_module = types.ModuleType("monGARS.core.style_finetuning")

    class StyleAnalysis:
        def __init__(
            self,
            traits=None,
            style=None,
            context_preferences=None,
            confidence=0.5,
        ):
            self.traits = traits or {}
            self.style = style or {}
            self.context_preferences = context_preferences or {}
            self.confidence = confidence

    class StyleFineTuner:
        async def estimate_personality(self, *_args, **_kwargs):
            return StyleAnalysis()

    repo_root = Path(__file__).resolve().parents[1]
    personality_path = repo_root / "monGARS" / "core" / "personality.py"
    root_pkg = types.ModuleType("monGARS")
    root_pkg.__path__ = [str(repo_root / "monGARS")]
    core_pkg = types.ModuleType("monGARS.core")
    core_pkg.__path__ = [str(repo_root / "monGARS" / "core")]

    monkeypatch.setitem(sys.modules, "init_db", fake_init)
    monkeypatch.setitem(sys.modules, "monGARS", root_pkg)
    monkeypatch.setitem(sys.modules, "monGARS.core", core_pkg)
    style_module.StyleAnalysis = StyleAnalysis
    style_module.StyleFineTuner = StyleFineTuner
    monkeypatch.setitem(sys.modules, "monGARS.core.style_finetuning", style_module)
    sys.modules.pop("monGARS.core.personality", None)

    spec = importlib.util.spec_from_file_location(
        "monGARS.core.personality", personality_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["monGARS.core.personality"] = module
    spec.loader.exec_module(module)

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
    assert factory.session.added is not None
    assert factory.session.added.user_id == "u1"
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
    assert existing.traits["new"] == 1
    assert factory.session.added is None
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
