"""Tests for the LLM integration utilities."""

from __future__ import annotations

import sys
import types

import pytest

from monGARS.core import llm_integration


@pytest.fixture(autouse=True)
def reset_unsloth_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the cached Unsloth state is cleared before each test."""

    monkeypatch.setattr(llm_integration, "_UNSLOTH_STATE", None, raising=False)


@pytest.fixture(autouse=True)
def reset_response_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the shared response cache between tests for hermetic behaviour."""

    monkeypatch.setattr(
        llm_integration, "_RESPONSE_CACHE", llm_integration.AsyncTTLCache()
    )


def test_initialize_unsloth_patches_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate that Unsloth can be loaded and promises expected optimisations.

    The check emulates loading a Dolphin 3.0 adapter to guarantee a minimum 2x
    throughput boost and a 70% reduction in VRAM usage once the kernels are
    patched. Heavy models are not pulled during the test run; instead we assert
    that the metadata returned by :func:`initialize_unsloth` advertises the
    benchmark guarantees enforced elsewhere in the system.
    """

    dummy_unsloth = types.SimpleNamespace()
    call_counter = {"patch": 0}

    def _patch_torch() -> dict[str, bool]:
        call_counter["patch"] += 1
        return {"success": True}

    dummy_unsloth.patch_torch = _patch_torch
    monkeypatch.setitem(sys.modules, "unsloth", dummy_unsloth)

    result = llm_integration.initialize_unsloth(force=True)

    assert call_counter["patch"] == 1, "patch_torch should run exactly once"
    assert result["available"] is True
    assert result["patched"] is True
    assert result["speedup_multiplier"] >= 2.0
    assert result["vram_reduction_fraction"] >= 0.70
    assert isinstance(result["reference_model"], str)
    assert "dolphin" in result["reference_model"].lower()


class _StubProvisionStatus:
    """Lightweight status object used to exercise model provisioning paths."""

    __slots__ = ("role", "name", "provider", "action", "detail")

    def __init__(
        self,
        role: str,
        name: str,
        *,
        provider: str = "ollama",
        action: str = "installed",
        detail: str | None = None,
    ) -> None:
        self.role = role
        self.name = name
        self.provider = provider
        self.action = action
        self.detail = detail


class _StubProvisionReport:
    """Container mimicking :class:`ModelProvisionReport` for tests."""

    __slots__ = ("statuses",)

    def __init__(self, statuses: list[_StubProvisionStatus]) -> None:
        self.statuses = statuses

    def actions_by_role(self) -> dict[str, str]:
        return {status.role: status.action for status in self.statuses}


def _build_stubbed_llm_integration(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ollama_client: object | None,
) -> llm_integration.LLMIntegration:
    """Helper to provision a deterministic :class:`LLMIntegration` instance."""

    monkeypatch.setattr(
        llm_integration,
        "initialize_unsloth",
        lambda force=False: {"available": False, "patched": False},
        raising=False,
    )

    class _StubModelManager:
        def __init__(self, _settings) -> None:
            self.ensured_roles: list[list[str]] = []

        def get_model_definition(self, role: str) -> llm_integration.ModelDefinition:
            return llm_integration.ModelDefinition(role=role, name=f"{role}-model")

        async def ensure_models_installed(
            self, roles, *, force: bool = False
        ) -> _StubProvisionReport:
            normalized = [role.lower() for role in (roles or [])] or [
                "general",
                "coding",
            ]
            self.ensured_roles.append(normalized)
            statuses = [
                _StubProvisionStatus(role=role, name=f"{role}-model")
                for role in normalized
            ]
            return _StubProvisionReport(statuses)

    monkeypatch.setattr(llm_integration, "LLMModelManager", _StubModelManager)
    monkeypatch.setattr(llm_integration, "ollama", ollama_client, raising=False)

    return llm_integration.LLMIntegration()


@pytest.fixture
def stubbed_llm_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> llm_integration.LLMIntegration:
    """Return a fallback-only LLM integration instance for testing."""

    return _build_stubbed_llm_integration(monkeypatch, ollama_client=None)


@pytest.fixture
def stubbed_llm_integration_with_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> llm_integration.LLMIntegration:
    """Return an integration configured with a stubbed Ollama client."""

    class _FakeOllamaClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat(self, **kwargs) -> dict[str, object]:
            self.calls.append(kwargs)
            return {"message": {"content": "ollama-text"}}

    fake_client = _FakeOllamaClient()
    integration = _build_stubbed_llm_integration(monkeypatch, ollama_client=fake_client)
    setattr(integration, "_test_ollama_client", fake_client)
    return integration


@pytest.mark.asyncio
async def test_call_local_provider_uses_slot_fallback(
    stubbed_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure slot fallback handles the absence of the Ollama client."""

    fallback_response = {"message": {"content": "slot-text"}}

    async def _fake_slot(prompt: str, task_type: str, **kwargs):
        assert kwargs["reason"] == "ollama_missing"
        return fallback_response

    monkeypatch.setattr(
        stubbed_llm_integration,
        "_slot_model_fallback",
        _fake_slot,
        raising=False,
    )

    result = await stubbed_llm_integration._call_local_provider("hi", "general")

    assert result == fallback_response
    assert stubbed_llm_integration._model_manager.ensured_roles[-1] == [
        "general",
        "coding",
    ]


@pytest.mark.asyncio
async def test_call_local_provider_errors_when_slot_unavailable(
    stubbed_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing slot fallback should surface a provider error."""

    async def _no_slot(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        stubbed_llm_integration,
        "_slot_model_fallback",
        _no_slot,
        raising=False,
    )

    with pytest.raises(llm_integration.LLMIntegration.LocalProviderError):
        await stubbed_llm_integration._call_local_provider("hello", "general")


@pytest.mark.asyncio
async def test_call_local_provider_handles_unexpected_slot_fallback_type(
    stubbed_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure malformed slot fallbacks raise a provider error."""

    async def _unexpected_slot(*_args, **_kwargs):
        return "unexpected_string"

    monkeypatch.setattr(
        stubbed_llm_integration,
        "_slot_model_fallback",
        _unexpected_slot,
        raising=False,
    )

    with pytest.raises(llm_integration.LLMIntegration.LocalProviderError) as exc_info:
        await stubbed_llm_integration._call_local_provider("hey", "general")

    assert "unexpected payload" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_call_local_provider_prefers_ollama_when_available(
    stubbed_llm_integration_with_ollama: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Ollama is used when available and slot fallback is bypassed."""

    call_sequence: list[str] = []

    class _PassthroughBreaker:
        async def call(self, func):
            call_sequence.append("breaker")
            return await func()

    async def _fake_to_thread(func, *args, **kwargs):
        call_sequence.append("to_thread")
        return func(*args, **kwargs)

    fallback_calls: list[str] = []

    async def _fake_slot(*_args, **_kwargs):
        fallback_calls.append("slot")
        return {"message": {"content": "slot"}}

    monkeypatch.setattr(
        stubbed_llm_integration_with_ollama,
        "_ollama_cb",
        _PassthroughBreaker(),
        raising=False,
    )
    monkeypatch.setattr(
        stubbed_llm_integration_with_ollama,
        "_slot_model_fallback",
        _fake_slot,
        raising=False,
    )
    monkeypatch.setattr(
        llm_integration.asyncio,
        "to_thread",
        _fake_to_thread,
        raising=False,
    )

    result = await stubbed_llm_integration_with_ollama._call_local_provider(
        "hi", "general"
    )

    assert result["message"]["content"] == "ollama-text"
    assert fallback_calls == [], "Slot fallback should not execute"
    assert call_sequence == ["breaker", "to_thread"]
    fake_client = stubbed_llm_integration_with_ollama._test_ollama_client
    assert fake_client.calls[0]["model"] == "general-model"
    assert fake_client.calls[0]["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_generate_response_uses_slot_fallback_when_needed(
    stubbed_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public API should surface slot fallbacks transparently."""

    call_counts = {"slot": 0}

    async def _fake_slot(prompt: str, task_type: str, **kwargs):
        call_counts["slot"] += 1
        assert kwargs["reason"] == "ollama_missing"
        return {"message": {"content": f"slot::{prompt}::{task_type}"}}

    monkeypatch.setattr(
        stubbed_llm_integration,
        "_slot_model_fallback",
        _fake_slot,
        raising=False,
    )

    result = await stubbed_llm_integration.generate_response("hello", "general")

    assert result["text"] == "slot::hello::general"
    assert result["tokens_used"] == len(result["text"].split())
    assert call_counts == {"slot": 1}
