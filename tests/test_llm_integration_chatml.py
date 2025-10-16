import pytest

from monGARS.core.inference_utils import build_converged_chat_prompt
from monGARS.core.llm_integration import LLMIntegration


@pytest.mark.asyncio
async def test_generate_response_translates_legacy_chatml(monkeypatch):
    llm = LLMIntegration()
    llm.use_ray = False
    captured: dict[str, str] = {}

    async def fake_call(prompt: str, task_type: str):
        captured["prompt"] = prompt
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(llm, "_call_local_provider", fake_call)
    prompt_bundle = build_converged_chat_prompt("Hello world", system_prompt="Test sys")

    result = await llm.generate_response(
        prompt_bundle.text,
        task_type="general",
        formatted_prompt=prompt_bundle.chatml,
    )

    assert result["text"] == "ok"
    prompt = captured["prompt"]
    assert "<|start_header_id|>" not in prompt
    assert "<|end_header_id|>" not in prompt
    assert "<|eot_id|>" not in prompt
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt


@pytest.mark.asyncio
async def test_generate_response_wraps_plain_prompts(monkeypatch):
    llm = LLMIntegration()
    llm.use_ray = False
    captured: dict[str, str] = {}

    async def fake_call(prompt: str, task_type: str):
        captured["prompt"] = prompt
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(llm, "_call_local_provider", fake_call)

    await llm.generate_response("Simple question?", task_type="general")

    prompt = captured["prompt"]
    assert prompt.startswith("<|begin_of_text|>")
    assert "<|system|>" in prompt
    assert prompt.count("<|user|>") == 1
    assert prompt.count("<|assistant|>") == 1
