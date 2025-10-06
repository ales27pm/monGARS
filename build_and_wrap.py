#!/usr/bin/env python3
"""QLoRA fine-tuning pipeline for Dolphin3.0-Llama3.1-8B."""

from __future__ import annotations

import json
import os
from pathlib import Path

from modules.neurons.registry import update_manifest
from monGARS.mlops.artifacts import (
    WrapperConfig,
    build_adapter_summary,
    render_output_bundle_readme,
    write_wrapper_bundle,
)
from monGARS.mlops.dataset import prepare_instruction_dataset
from monGARS.mlops.exporters import export_gguf, merge_lora_adapters
from monGARS.mlops.model import load_4bit_causal_lm, summarise_device_map
from monGARS.mlops.training import (
    LoraHyperParams,
    TrainerConfig,
    disable_training_mode,
    prepare_lora_model,
    save_lora_artifacts,
    train_qlora,
)
from monGARS.mlops.utils import (
    configure_cuda_allocator,
    describe_environment,
    ensure_dependencies,
    ensure_directory,
)

MODEL_ID = os.environ.get("MODEL_ID", "cognitivecomputations/Dolphin3.0-Llama3.1-8B")
DATASET_NAME = os.environ.get("DATASET", "yahma/alpaca-cleaned")
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", "0.10"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "1.0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
VRAM_BUDGET_MB = int(os.environ.get("VRAM_BUDGET_MB", "7300"))
OFFLOAD_DIR = Path(os.environ.get("OFFLOAD_DIR", "./offload"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./out"))
EXPORT_MERGED_FP16 = os.environ.get("EXPORT_MERGED_FP16", "0") == "1"
EXPORT_GGUF = os.environ.get("EXPORT_GGUF", "0") == "1"
GGUF_DIR = Path(os.environ.get("GGUF_DIR", OUTPUT_DIR / "gguf"))
GGUF_METHOD = os.environ.get("GGUF_METHOD", "q4_k_m")
AUTO_INSTALL = os.environ.get("AUTO_INSTALL", "1") == "1"
REGISTRY_PATH = os.environ.get("LLM_ADAPTER_REGISTRY_PATH") or os.environ.get(
    "ADAPTER_REGISTRY_PATH"
)
SUMMARY_FILENAME = "training_summary.json"


REQUIRED_PACKAGES = [
    "torch",
    "transformers>=4.44",
    "datasets",
    "peft>=0.11",
    "bitsandbytes>=0.44.1",
    "llm2vec",
    "sentencepiece",
]
OPTIONAL_PACKAGES = ["unsloth"]


def _locate_adapter_weights(adapter_dir: Path) -> Path | None:
    candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    configure_cuda_allocator()
    ensure_directory(OFFLOAD_DIR)
    ensure_directory(OUTPUT_DIR)
    ensure_dependencies(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, auto_install=AUTO_INSTALL)
    describe_environment()

    model, tokenizer = load_4bit_causal_lm(
        MODEL_ID,
        vram_budget_mb=VRAM_BUDGET_MB,
        offload_dir=OFFLOAD_DIR,
    )
    summarise_device_map(model)
    model = prepare_lora_model(model, LoraHyperParams())

    dataset = prepare_instruction_dataset(
        DATASET_NAME,
        tokenizer,
        MAX_SEQ_LEN,
        train_fraction=TRAIN_FRACTION,
    )

    trainer = train_qlora(
        model,
        dataset,
        config=TrainerConfig(
            output_dir=OUTPUT_DIR / "chat_lora",
            batch_size=BATCH_SIZE,
            grad_accum=GRAD_ACCUM,
            learning_rate=LR,
            epochs=EPOCHS,
            max_steps=MAX_STEPS,
        ),
    )

    adapters_dir = OUTPUT_DIR / "chat_lora"
    save_lora_artifacts(trainer.model, tokenizer, adapters_dir)
    disable_training_mode(trainer.model)
    weights_path = _locate_adapter_weights(adapters_dir)

    merged_dir = OUTPUT_DIR / "merged_fp16"
    merged = False
    if EXPORT_MERGED_FP16:
        merged = merge_lora_adapters(MODEL_ID, adapters_dir, output_dir=merged_dir)

    if EXPORT_GGUF:
        if not merged:
            raise RuntimeError("EXPORT_GGUF requires EXPORT_MERGED_FP16=1")
        export_gguf(merged_dir, gguf_dir=GGUF_DIR, quantization_method=GGUF_METHOD)

    wrapper_dir = OUTPUT_DIR / "wrapper"
    bundle_config = WrapperConfig(
        base_model_id=MODEL_ID,
        lora_dir=adapters_dir.resolve(),
        max_seq_len=MAX_SEQ_LEN,
        vram_budget_mb=VRAM_BUDGET_MB,
        offload_dir=OFFLOAD_DIR.resolve(),
    )
    write_wrapper_bundle(bundle_config, wrapper_dir)

    readme = render_output_bundle_readme(
        bundle_config,
        merged_fp16=merged,
        gguf_enabled=EXPORT_GGUF and merged,
        gguf_method=GGUF_METHOD,
    )
    (OUTPUT_DIR / "README_outputs.md").write_text(readme)

    summary = build_adapter_summary(
        adapter_dir=adapters_dir,
        weights_path=weights_path,
        wrapper_dir=wrapper_dir,
        status="success",
        labels={
            "category": "general_baseline",
            "quantization": "bnb_nf4",
            "first_run": "true",
        },
        metrics={
            "dataset_size": len(dataset),
            "train_fraction": TRAIN_FRACTION,
        },
        training={
            "base_model": MODEL_ID,
            "dataset": DATASET_NAME,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "max_steps": MAX_STEPS,
            "vram_budget_mb": VRAM_BUDGET_MB,
            "quantization_method": "bnb-4bit-nf4",
        },
    )
    artifacts = summary.setdefault("artifacts", {})
    if merged:
        artifacts["merged_fp16"] = str(merged_dir)
    if EXPORT_GGUF and merged:
        artifacts["gguf"] = str(GGUF_DIR)
        summary.setdefault("labels", {}).update({"gguf_method": GGUF_METHOD})
    summary_path = OUTPUT_DIR / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=== ALL DONE ===")
    print(f"- LoRA adapters: {adapters_dir}")
    if merged:
        print(f"- Merged FP16 model: {merged_dir}")
    if EXPORT_GGUF and merged:
        print(f"- GGUF export: {GGUF_DIR}")
    print(f"- Wrapper module: {wrapper_dir / 'project_wrapper.py'}")
    print(f"- Training summary: {summary_path}")

    if REGISTRY_PATH:
        manifest = update_manifest(REGISTRY_PATH, summary)
        print(f"- Adapter manifest updated: {manifest.path}")
    else:
        print(
            "- Adapter manifest not updated (set LLM_ADAPTER_REGISTRY_PATH to register output)"
        )


if __name__ == "__main__":
    main()
