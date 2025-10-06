"""Utilities to generate GGUF and wrapper artefacts for Dolphin fine-tuning."""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_WRAPPER_TEMPLATE = """# Auto-generated wrapper: project_wrapper.py
import os
from collections.abc import Iterable

import torch
from llm2vec import LLM2Vec
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE_MODEL_ID = {base_model_id!r}
LORA_DIR = {lora_dir!r}
VRAM_BUDGET_MB = {vram_budget_mb}
ACTIVATION_BUFFER_MB = {activation_buffer_mb}
OFFLOAD_DIR = {offload_dir!r}
MAX_SEQ_LEN = {max_seq_len}

# Prefer the numerically stable SDPA kernels for Turing-era GPUs (e.g., RTX 2070).
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:  # pragma: no cover - backend availability differs per torch build
    pass


def _bnb4() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def _weight_budget_mb() -> int:
    reserve = max(0, ACTIVATION_BUFFER_MB)
    weight_budget = VRAM_BUDGET_MB - reserve
    if weight_budget < 512:
        weight_budget = max(VRAM_BUDGET_MB // 2, 512)
    return min(VRAM_BUDGET_MB, weight_budget)


def _max_memory() -> dict[int | str, str]:
    return {{0: f"{{_weight_budget_mb()}}MiB", "cpu": "48GiB"}}


class ChatAndEmbed:
    \"\"\"Load one model instance for both chat and embeddings.\"\"\"

    def __init__(self) -> None:
        os.makedirs(OFFLOAD_DIR, exist_ok=True)
        max_memory = _max_memory()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=OFFLOAD_DIR,
            quantization_config=_bnb4(),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.model, LORA_DIR)
        self.model.config.use_cache = True
        self.model.eval()

        self.l2v = LLM2Vec(
            self.model,
            self.tokenizer,
            pooling_mode="mean",
            max_length=min(512, MAX_SEQ_LEN),
        )

    @torch.inference_mode()
    def generate(
        self,
        user_text: str,
        system_prompt: str = "You are Dolphin, a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {{"role": "system", "content": system_prompt}},
                    {{"role": "user", "content": user_text}},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"User: {{user_text}}\\nAssistant:"

        batch = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = batch["input_ids"].shape[1]
        generated = output[0, prompt_length:]
        if generated.numel() == 0:
            generated = output[0]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1]
        return text.strip()

    @torch.inference_mode()
    def embed(self, texts: Iterable[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return self.l2v.encode(list(texts))


if __name__ == "__main__":
    cae = ChatAndEmbed()
    print(">> chat:", cae.generate("Say hello in 8 words."))
    embeddings = cae.embed(["a small embedding test", "another sentence"])
    print(">> embed shape:", tuple(embeddings.shape))
"""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WrapperConfig:
    """Describe how the chat and embedding wrapper should be rendered."""

    base_model_id: str
    lora_dir: Path
    max_seq_len: int
    vram_budget_mb: int
    offload_dir: Path
    activation_buffer_mb: int = 1024
    quantized_4bit: bool = True

    def to_json(self) -> Dict[str, object]:
        """Return a serialisable view for ``config.json``."""

        return {
            "base_model_id": self.base_model_id,
            "lora_dir": str(self.lora_dir),
            "max_seq_len": self.max_seq_len,
            "quantized_4bit": self.quantized_4bit,
            "vram_budget_mb": self.vram_budget_mb,
            "offload_dir": str(self.offload_dir),
            "activation_buffer_mb": self.activation_buffer_mb,
        }


def build_adapter_summary(
    *,
    adapter_dir: Path,
    weights_path: Path | None,
    wrapper_dir: Path | None = None,
    status: str = "success",
    labels: Mapping[str, str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest-friendly summary for freshly trained adapters."""

    artifacts: dict[str, str] = {"adapter": str(adapter_dir)}
    if weights_path is not None:
        artifacts["weights"] = str(weights_path)
    if wrapper_dir is not None:
        artifacts["wrapper"] = str(wrapper_dir)

    summary: dict[str, Any] = {"status": status, "artifacts": artifacts}

    if labels:
        summary["labels"] = dict(labels)
    if metrics:
        summary["metrics"] = dict(metrics)
    if training:
        summary["training"] = dict(training)

    return summary


def render_project_wrapper(config: WrapperConfig) -> str:
    """Render the ``project_wrapper.py`` module."""

    return PROJECT_WRAPPER_TEMPLATE.format(
        base_model_id=config.base_model_id,
        lora_dir=str(config.lora_dir),
        vram_budget_mb=config.vram_budget_mb,
        activation_buffer_mb=config.activation_buffer_mb,
        offload_dir=str(config.offload_dir),
        max_seq_len=config.max_seq_len,
    ).strip()


def render_wrapper_readme(config: WrapperConfig) -> str:
    """Render the ``README_integration.md`` helper."""

    return textwrap.dedent(
        f"""
        # Wrapper Integration

        Files generated:
        - `config.json` — paths and basic settings
        - `project_wrapper.py` — import `ChatAndEmbed` to use one model for chat + embeddings

        ## Quick use

        ```python
        from project_wrapper import ChatAndEmbed

        cae = ChatAndEmbed()
        print(cae.generate("Explain QLoRA in 1 sentence."))
        vec = cae.embed("Vectorize this text.")
        print(vec.shape)
        ```

        Model:
          • Base: {config.base_model_id}
          • LoRA: {config.lora_dir}

        Quantization: 4-bit (NF4) w/ BitsAndBytes.
        VRAM cap: {config.vram_budget_mb} MiB with {config.activation_buffer_mb} MiB reserved for
        activations (adjust inside `project_wrapper.py` if needed).
        """
    ).strip()


def render_output_bundle_readme(
    config: WrapperConfig,
    *,
    merged_fp16: bool,
    gguf_enabled: bool,
    gguf_method: str,
) -> str:
    """Render the top-level README describing bundle contents."""

    extras: list[str] = []
    if merged_fp16:
        extras.append(
            "- `merged_fp16/` — merged full weights (FP16; large) for conversion/export workflows"
        )
    if gguf_enabled:
        extras.append(
            f"- `gguf/` — GGUF export (e.g., {gguf_method}) for llama.cpp/Ollama"
        )

    extras_block = "\n".join(extras)
    if extras_block:
        extras_block = f"\n{extras_block}"

    return textwrap.dedent(
        f"""
        # Output bundle

        This folder contains all artifacts to integrate the fine-tuned Dolphin 8B model and a wrapper
        that enables both chat generation and embeddings (via LLM2Vec) from the SAME model instance.

        Layout
        - `chat_lora/` — LoRA adapters + tokenizer for instruction SFT{extras_block}
        - `wrapper/`
          - `project_wrapper.py` — load once, then `.generate(...)` and `.embed(...)`
          - `config.json` — base paths and VRAM settings
          - `README_integration.md`

        Typical use

        ```python
        from wrapper.project_wrapper import ChatAndEmbed

        cae = ChatAndEmbed()
        print(cae.generate("Give me 3 creative prompts."))
        embeddings = cae.embed(["first line", "second line"])
        print(embeddings.shape)
        ```

        If you change GPUs or memory limits, tweak `VRAM_BUDGET_MB`, `ACTIVATION_BUFFER_MB`, and
        `OFFLOAD_DIR` in `wrapper/project_wrapper.py`.
        """
    ).strip()


def write_wrapper_bundle(
    config: WrapperConfig, output_root: Path
) -> Mapping[str, Path]:
    """Write wrapper artefacts to ``output_root`` and return their paths."""

    wrapper_dir = output_root / "wrapper"
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    module_path = wrapper_dir / "project_wrapper.py"
    module_path.write_text(render_project_wrapper(config), encoding="utf-8")

    cfg_path = wrapper_dir / "config.json"
    cfg_path.write_text(json.dumps(config.to_json(), indent=2), encoding="utf-8")

    readme_path = wrapper_dir / "README_integration.md"
    readme_path.write_text(render_wrapper_readme(config), encoding="utf-8")

    logger.info("Wrote wrapper bundle to %s", wrapper_dir)
    return {"module": module_path, "config": cfg_path, "readme": readme_path}


__all__ = [
    "WrapperConfig",
    "build_adapter_summary",
    "render_output_bundle_readme",
    "render_project_wrapper",
    "render_wrapper_readme",
    "write_wrapper_bundle",
]
