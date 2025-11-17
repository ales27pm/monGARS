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


@pytest.fixture(autouse=True)
def reset_unified_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the unified runtime singleton is cleared between tests."""

    monkeypatch.setattr(
        llm_integration.LLMIntegration, "_unified_service", None, raising=False
    )


def test_initialize_unsloth_patches_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate that Unsloth can be loaded and promises expected optimisations.

    The check emulates loading a Dolphin-X1 adapter to guarantee a minimum 2x
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


class _FakeProvisionStatus:
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


class _FakeProvisionReport:
    """Container mimicking :class:`ModelProvisionReport` for tests."""

    __slots__ = ("statuses",)

    def __init__(self, statuses: list[_FakeProvisionStatus]) -> None:
        self.statuses = statuses

    def actions_by_role(self) -> dict[str, str]:
        return {status.role: status.action for status in self.statuses}


def _build_fake_llm_integration(
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

    class _FakeModelManager:
        def __init__(self, _settings) -> None:
            self.ensured_roles: list[list[str]] = []

        def get_model_definition(self, role: str) -> llm_integration.ModelDefinition:
            return llm_integration.ModelDefinition(role=role, name=f"{role}-model")

        async def ensure_models_installed(
            self, roles, *, force: bool = False
        ) -> _FakeProvisionReport:
            normalized = [role.lower() for role in (roles or [])] or [
                "general",
                "coding",
            ]
            self.ensured_roles.append(normalized)
            statuses = [
                _FakeProvisionStatus(role=role, name=f"{role}-model")
                for role in normalized
            ]
            return _FakeProvisionReport(statuses)

    monkeypatch.setattr(llm_integration, "LLMModelManager", _FakeModelManager)
    monkeypatch.setattr(llm_integration, "ollama", ollama_client, raising=False)

    return llm_integration.LLMIntegration()


@pytest.fixture
def fake_llm_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> llm_integration.LLMIntegration:
    """Return a fallback-only LLM integration instance for testing."""

    return _build_fake_llm_integration(monkeypatch, ollama_client=None)


@pytest.fixture
def fake_llm_integration_with_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> llm_integration.LLMIntegration:
    """Return an integration configured with a fake Ollama client."""

    class _FakeOllamaClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat(self, **kwargs) -> dict[str, object]:
            self.calls.append(kwargs)
            return {"message": {"content": "ollama-text"}}

    fake_client = _FakeOllamaClient()
    integration = _build_fake_llm_integration(monkeypatch, ollama_client=fake_client)
    setattr(integration, "_test_ollama_client", fake_client)
    return integration


@pytest.mark.asyncio
async def test_call_local_provider_uses_slot_fallback(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure slot fallback handles the absence of the Ollama client."""

    fallback_response = {"message": {"content": "slot-text"}}

    async def _fake_slot(prompt: str, task_type: str, **kwargs):
        assert kwargs["reason"] == "ollama_missing"
        return fallback_response

    monkeypatch.setattr(
        fake_llm_integration,
        "_slot_model_fallback",
        _fake_slot,
        raising=False,
    )

    result = await fake_llm_integration._call_local_provider("hi", "general")

    assert result == fallback_response
    assert fake_llm_integration._model_manager.ensured_roles[-1] == [
        "general",
        "coding",
    ]


@pytest.mark.asyncio
async def test_call_local_provider_errors_when_slot_unavailable(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing slot fallback should surface a provider error."""

    async def _no_slot(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        fake_llm_integration,
        "_slot_model_fallback",
        _no_slot,
        raising=False,
    )

    with pytest.raises(llm_integration.LLMIntegration.LocalProviderError):
        await fake_llm_integration._call_local_provider("hello", "general")


@pytest.mark.asyncio
async def test_call_local_provider_handles_unexpected_slot_fallback_type(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure malformed slot fallbacks raise a provider error."""

    async def _unexpected_slot(*_args, **_kwargs):
        return "unexpected_string"

    monkeypatch.setattr(
        fake_llm_integration,
        "_slot_model_fallback",
        _unexpected_slot,
        raising=False,
    )

    with pytest.raises(llm_integration.LLMIntegration.LocalProviderError) as exc_info:
        await fake_llm_integration._call_local_provider("hey", "general")

    assert "unexpected payload" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_call_local_provider_prefers_ollama_when_available(
    fake_llm_integration_with_ollama: llm_integration.LLMIntegration,
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
        fake_llm_integration_with_ollama,
        "_ollama_cb",
        _PassthroughBreaker(),
        raising=False,
    )
    monkeypatch.setattr(
        fake_llm_integration_with_ollama,
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

    result = await fake_llm_integration_with_ollama._call_local_provider(
        "hi", "general"
    )

    assert result["message"]["content"] == "ollama-text"
    assert fallback_calls == [], "Slot fallback should not execute"
    assert call_sequence == ["breaker", "to_thread"]
    fake_client = fake_llm_integration_with_ollama._test_ollama_client
    assert fake_client.calls[0]["model"] == "general-model"
    assert fake_client.calls[0]["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_generate_response_uses_slot_fallback_when_needed(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public API should surface slot fallbacks transparently."""

    call_counts = {"slot": 0}

    async def _fake_slot(prompt: str, task_type: str, **kwargs):
        call_counts["slot"] += 1
        assert kwargs["reason"] == "ollama_missing"
        return {"message": {"content": f"slot::{prompt}::{task_type}"}}

    monkeypatch.setattr(
        fake_llm_integration,
        "_slot_model_fallback",
        _fake_slot,
        raising=False,
    )

    result = await fake_llm_integration.generate_response("hello", "general")

    assert result["text"].startswith("slot::")
    assert result["text"].endswith("::general")
    payload = result["text"].split("::", 2)[1]
    assert "<|system|>" in payload
    assert "<|user|>" in payload
    assert "<|assistant|>" in payload
    assert result["tokens_used"] == len(result["text"].split())
    assert call_counts == {"slot": 1}


@pytest.mark.asyncio
async def test_generate_response_returns_expected_keys(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful generations should populate the standard response payload."""

    fake_llm_integration.use_ray = False

    async def _fake_local(prompt: str, task_type: str) -> dict[str, object]:
        assert task_type == "general"
        assert prompt.startswith(llm_integration.CHATML_BEGIN_OF_TEXT)
        return {"message": {"content": "final answer"}}

    monkeypatch.setattr(
        fake_llm_integration, "_call_local_provider", _fake_local, raising=False
    )

    result = await fake_llm_integration.generate_response("hello", task_type="general")

    assert result == {
        "text": "final answer",
        "confidence": pytest.approx(2 / 512),
        "tokens_used": 2,
        "source": "local",
        "adapter_version": fake_llm_integration._current_adapter_version,
    }

    active_prompt = fake_llm_integration._ensure_chatml_prompt("hello", None)
    cache_key = fake_llm_integration._cache_key("general", active_prompt)
    cached = await llm_integration._RESPONSE_CACHE.get(cache_key)
    assert cached == result


@pytest.mark.asyncio
async def test_generate_response_handles_local_provider_errors(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local provider failures should produce an error payload and cache entry."""

    fake_llm_integration.use_ray = False

    async def _raise_local(prompt: str, task_type: str) -> dict[str, object]:
        raise fake_llm_integration.LocalProviderError("Missing Ollama API key.")

    monkeypatch.setattr(
        fake_llm_integration, "_call_local_provider", _raise_local, raising=False
    )

    result = await fake_llm_integration.generate_response(
        "diagnostics", task_type="general"
    )

    assert result == {
        "text": "Missing Ollama API key.",
        "confidence": 0.0,
        "tokens_used": 0,
        "source": "error",
        "adapter_version": fake_llm_integration._current_adapter_version,
    }

    active_prompt = fake_llm_integration._ensure_chatml_prompt("diagnostics", None)
    cache_key = fake_llm_integration._cache_key("general", active_prompt)
    cached = await llm_integration._RESPONSE_CACHE.get(cache_key)
    assert cached == result


def test_infer_task_type_detects_coding(
    fake_llm_integration: llm_integration.LLMIntegration,
) -> None:
    """Ensure the heuristic flips to the coding role for code-heavy prompts."""

    assert (
        fake_llm_integration.infer_task_type("Outline a plan for the weekend")
        == "general"
    )
    assert (
        fake_llm_integration.infer_task_type("```python\nprint('hi')\n```") == "coding"
    )
    assert (
        fake_llm_integration.infer_task_type("Create a Java class with a main function")
        == "coding"
    )
    assert (
        fake_llm_integration.infer_task_type(
            "function greet() {\n    return 'hello';\n}"
        )
        == "coding"
    )


class _FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return list(text)


class _FakeUnifiedRuntime:
    def __init__(self) -> None:
        self.generate_calls: list[tuple[str, dict[str, object]]] = []
        self.embed_calls: list[list[str]] = []
        self.tokenizer = _FakeTokenizer()

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.generate_calls.append((prompt, dict(kwargs)))
        return f"generated::{prompt}"

    def embed(self, texts: list[str]) -> str:
        self.embed_calls.append(list(texts))
        return f"embedded::{len(texts)}"


def test_generate_method_uses_unified_runtime(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synchronous generate helper should delegate to the unified runtime."""

    runtime = _FakeUnifiedRuntime()
    monkeypatch.setattr(
        llm_integration.LLMIntegration, "_unified_service", runtime, raising=False
    )

    result = fake_llm_integration.generate("hello", temperature=0.25)

    assert result == "generated::hello"
    assert runtime.generate_calls == [("hello", {"temperature": 0.25})]


def test_embed_method_uses_unified_runtime(
    fake_llm_integration: llm_integration.LLMIntegration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddings should be routed through the cached runtime."""

    runtime = _FakeUnifiedRuntime()
    monkeypatch.setattr(
        llm_integration.LLMIntegration, "_unified_service", runtime, raising=False
    )

    result = fake_llm_integration.embed(["a", "b"])

    assert result == "embedded::2"
    assert runtime.embed_calls == [["a", "b"]]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("", "general"),
        ("   \n", "general"),
        ("Steps:\n    - item one\n    - item two", "general"),
        ("Please return soon", "general"),
        ("Importantly, we should regroup", "general"),
        (
            "Traceback (most recent call last):\n  File 'app.py', line 1\n"
            "NameError: name 'main' is not defined. This is a python error.",
            "coding",
        ),
        (
            "In JavaScript, console.log throws ReferenceError: please help",
            "coding",
        ),
        (
            "Linker output: undefined reference to std::cout in this C++ build",
            "coding",
        ),
        ("Discuss classifying imports and exports", "general"),
    ],
)
def test_infer_task_type_edge_cases(
    fake_llm_integration: llm_integration.LLMIntegration,
    prompt: str,
    expected: str,
) -> None:
    assert fake_llm_integration.infer_task_type(prompt) == expected


def test_infer_task_type_requires_multiple_signals(
    fake_llm_integration: llm_integration.LLMIntegration,
) -> None:
    """Single keyword references should not force the coding route."""

    assert (
        fake_llm_integration.infer_task_type(
            "Can you explain what a function is in mathematics?"
        )
        == "general"
    )
    assert (
        fake_llm_integration.infer_task_type("Talk about the Rust belt economy")
        == "general"
    )
    assert (
        fake_llm_integration.infer_task_type(
            "Share the agenda:\n    This indented paragraph describes outcomes."
        )
        == "general"
    )


@pytest.mark.asyncio
async def test_resolve_adapter_updates_version_for_reasoning(monkeypatch) -> None:
    """Selecting the reasoning adapter should update the cached version."""

    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("USE_RAY_SERVE", "true")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")

    integration = llm_integration.LLMIntegration()

    async def _fake_ensure_metadata():
        integration._adapter_metadata = {"version": "baseline"}
        integration._update_adapter_version("baseline")
        return integration._adapter_metadata

    monkeypatch.setattr(
        integration,
        "_ensure_adapter_metadata",
        _fake_ensure_metadata,
        raising=False,
    )

    def _fake_reasoning_payload():
        return {
            "version": "reasoning-v1",
            "adapter_path": "/tmp/adapter.safetensors",
            "weights_path": "/tmp/weights.safetensors",
            "wrapper_path": "/tmp/wrapper.py",
        }

    monkeypatch.setattr(
        integration,
        "_load_reasoning_adapter_payload",
        _fake_reasoning_payload,
        raising=False,
    )

    payload = await integration._resolve_adapter_for_task(
        "general", {"reasoning": True}
    )

    assert payload["version"] == "reasoning-v1"
    assert integration._current_adapter_version == "reasoning-v1"
