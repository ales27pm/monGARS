"""Utilities for persisting and exporting trained artefacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def merge_lora_adapters(
    base_model_id: str,
    adapters_dir: Path,
    *,
    output_dir: Path,
) -> bool:
    """Merge LoRA adapters into the base model and persist FP16 weights."""

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency missing
        logger.warning("Unable to import transformers/peft for merge", exc_info=True)
        raise RuntimeError("transformers and peft are required to merge adapters") from exc

    logger.info(
        "Merging adapters into base model",
        extra={"base_model": base_model_id, "adapters_dir": str(adapters_dir)},
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map={"": "cpu"},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    merged = PeftModel.from_pretrained(base_model, str(adapters_dir)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged FP16 model saved", extra={"output_dir": str(output_dir)})
    return True


def export_gguf(
    source_dir: Path,
    *,
    gguf_dir: Path,
    quantization_method: str,
) -> bool:
    """Export a model directory to GGUF via Unsloth when available."""

    try:
        from unsloth import FastModel
    except Exception as exc:  # pragma: no cover - optional dependency missing
        raise RuntimeError("Unsloth is required for GGUF export") from exc

    logger.info(
        "Exporting GGUF",
        extra={"source_dir": str(source_dir), "gguf_dir": str(gguf_dir), "method": quantization_method},
    )
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastModel.from_pretrained(str(source_dir))
    model.save_pretrained_gguf(str(gguf_dir), tokenizer=tokenizer, quantization_method=quantization_method)
    logger.info("GGUF export complete", extra={"gguf_dir": str(gguf_dir)})
    return True


def run_generation_smoke_test(model: Any, tokenizer: Any, prompt: str) -> str | None:
    """Generate a short sample from ``model`` for validation."""

    if not prompt:
        return None
    try:
        batch = tokenizer(prompt, return_tensors="pt")
        target_device = getattr(model, "device", None)
        if target_device is not None:
            batch = batch.to(target_device)
        output = model.generate(
            **batch,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    except Exception:  # pragma: no cover - defensive guard
        logger.warning("Generation smoke test failed", exc_info=True)
        return None
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()
