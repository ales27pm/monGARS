#!/usr/bin/env python3
"""One-model chat + embedding pipeline for Dolphin3.0-Llama3.1-8B."""

from __future__ import annotations

import json
import os
from pathlib import Path

from llm2vec import LLM2Vec

from monGARS.mlops.dataset import prepare_instruction_dataset
from monGARS.mlops.exporters import (
    export_gguf,
    merge_lora_adapters,
    run_generation_smoke_test,
)
from monGARS.mlops.model import load_4bit_causal_lm, summarise_device_map
from monGARS.mlops.training import (
    LoraHyperParams,
    TrainerConfig,
    disable_training_mode,
    prepare_lora_model_light,
    run_embedding_smoke_test,
    save_lora_artifacts,
    train_qlora,
)
from monGARS.mlops.utils import (
    configure_cuda_allocator,
    describe_environment,
    ensure_dependencies,
    ensure_directory,
)

MODEL_ID = os.environ.get("MODEL_ID", "dphn/Dolphin3.0-Llama3.1-8B")
DATASET_ID = os.environ.get("DATASET_ID", "yahma/alpaca-cleaned")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs_dolphin8b"))
OFFLOAD_DIR = Path(os.environ.get("OFFLOAD_DIR", "./offload"))
VRAM_BUDGET_MB = int(os.environ.get("VRAM_BUDGET_MB", "7300"))
ACTIVATION_BUFFER_MB = int(
    os.environ.get(
        "ACTIVATION_BUFFER_MB", os.environ.get("VRAM_ACTIVATION_BUFFER_MB", "1024")
    )
)
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "1.0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))
SAVE_MERGED_FP16 = os.environ.get("SAVE_MERGED_FP16", "0") == "1"
EXPORT_GGUF = os.environ.get("EXPORT_GGUF", "0") == "1"
GGUF_DIR = Path(os.environ.get("GGUF_DIR", "gguf_out"))
GGUF_METHOD = os.environ.get("GGUF_METHOD", "q4_k_m")
RUN_TINY_TESTS = os.environ.get("RUN_TINY_TESTS", "1") == "1"
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", "1.0"))

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


def main() -> None:
    configure_cuda_allocator()
    ensure_directory(OUTPUT_DIR)
    ensure_directory(OFFLOAD_DIR)
    ensure_dependencies(REQUIRED_PACKAGES, OPTIONAL_PACKAGES)
    describe_environment()

    model, tokenizer = load_4bit_causal_lm(
        MODEL_ID,
        vram_budget_mb=VRAM_BUDGET_MB,
        activation_buffer_mb=ACTIVATION_BUFFER_MB,
        offload_dir=OFFLOAD_DIR,
    )
    summarise_device_map(model)
    model = prepare_lora_model_light(
        model,
        LoraHyperParams(r=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT),
    )

    dataset = prepare_instruction_dataset(
        DATASET_ID,
        tokenizer,
        MAX_SEQ_LEN,
        train_fraction=TRAIN_FRACTION,
    )

    trainer = train_qlora(
        model,
        dataset,
        config=TrainerConfig(
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
            grad_accum=GRAD_ACCUM,
            learning_rate=LR,
            epochs=EPOCHS,
            max_steps=MAX_STEPS,
        ),
    )

    save_lora_artifacts(trainer.model, tokenizer, OUTPUT_DIR)
    disable_training_mode(trainer.model)

    merged_dir = OUTPUT_DIR / "merged_fp16"
    merged = False
    if SAVE_MERGED_FP16:
        merged = merge_lora_adapters(MODEL_ID, OUTPUT_DIR, output_dir=merged_dir)

    if EXPORT_GGUF:
        source_dir = merged_dir if merged else OUTPUT_DIR
        export_gguf(source_dir, gguf_dir=GGUF_DIR, quantization_method=GGUF_METHOD)

    tokenizer_config = {
        "name_or_path": getattr(tokenizer, "name_or_path", MODEL_ID),
        "cls_token": getattr(tokenizer, "cls_token", None),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "additional_special_tokens": getattr(
            tokenizer, "additional_special_tokens", None
        ),
    }

    wrapper_config = {
        "base_model_id": MODEL_ID,
        "adapters_dir": str(OUTPUT_DIR),
        "tokenizer": tokenizer_config,
        "quantization": "bnb-4bit nf4 double-quant fp16 compute",
        "max_seq_len": MAX_SEQ_LEN,
        "activation_buffer_mb": ACTIVATION_BUFFER_MB,
        "chat_backend": {
            "provider": "ollama",
            "model": "dolphin3",
            "parameters": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": min(512, MAX_SEQ_LEN),
            },
        },
        "embedding_backend": "huggingface",
        "embedding_options": {
            "pooling_mode": "mean",
            "normalise": False,
            "attention_mask_weighting": "mean",
            "dtype": "float32",
            "do_sample": False,
            "top_p": 1.0,
            "max_length": min(512, MAX_SEQ_LEN),
        },
        "generation_defaults": {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_new_tokens": min(512, MAX_SEQ_LEN),
        },
        "artifacts": {
            "tokenizer_dir": str(OUTPUT_DIR / "tokenizer"),
            "adapter_subdir": "lora_adapter",
            "merged_subdir": "merged_fp16",
            "wrapper_config_path": str(OUTPUT_DIR / "wrapper_config.json"),
        },
        "notes": (
            "Reload the Dolphin3.0 base in 4-bit, apply PEFT adapters, and wrap with "
            "LLM2Vec for deterministic embedding extraction."
        ),
    }
    (OUTPUT_DIR / "wrapper_config.json").write_text(
        json.dumps(wrapper_config, indent=2)
    )

    if RUN_TINY_TESTS:
        prompt = (
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello in 7 words."},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            if hasattr(tokenizer, "apply_chat_template")
            else "User: Say hello in 7 words.\nAssistant:"
        )
        generation = run_generation_smoke_test(trainer.model, tokenizer, prompt)
        if generation:
            print("GEN:", generation)

        l2v = LLM2Vec(
            trainer.model,
            tokenizer,
            pooling_mode="mean",
            max_length=min(512, MAX_SEQ_LEN),
        )
        embedding_shape = run_embedding_smoke_test(l2v, ["a tiny embedding smoke test"])
        if embedding_shape:
            print("EMB shape:", embedding_shape)

    print("All done. Artifacts @", OUTPUT_DIR)


if __name__ == "__main__":
    main()
