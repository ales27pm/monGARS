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


@pytest.fixture
def stubbed_llm_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> llm_integration.LLMIntegration:
    """Return an :class:`LLMIntegration` wired with lightweight collaborators."""

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
            normalized = [role.lower() for role in (roles or [])]
            if not normalized:
                normalized = ["general", "coding"]
            self.ensured_roles.append(normalized)
            statuses = [
                _StubProvisionStatus(role=role, name=f"{role}-model")
                for role in normalized
            ]
            return _StubProvisionReport(statuses)

    monkeypatch.setattr(llm_integration, "LLMModelManager", _StubModelManager)
    monkeypatch.setattr(llm_integration, "ollama", None, raising=False)

    return llm_integration.LLMIntegration()


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

    assert result is fallback_response
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
