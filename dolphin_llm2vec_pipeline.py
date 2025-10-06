#!/usr/bin/env python3
"""
One-model pipeline for RTX 2070 (8GB).

This script orchestrates an end-to-end QLoRA fine-tuning workflow around
Dolphin3.0-Llama3.1-8B with BitsAndBytes 4-bit quantisation, PEFT LoRA
adapters, and a reusable LLM2Vec wrapper for embeddings. It performs the
following steps:

1. Installs required dependencies when missing (no version pins).
2. Loads the base model in 4-bit using an explicit GPU VRAM budget with CPU
   offload fallbacks.
3. Prepares an instruction-tuning dataset and masks user prompts so the loss is
   applied on the assistant span only.
4. Runs QLoRA fine-tuning via Hugging Face Trainer.
5. Saves LoRA adapters and tokenizer artefacts and optionally merges them into
   an FP16 checkpoint for archival/exports.
6. Optionally exports GGUF weights via Unsloth when available.
7. Wraps the fine-tuned model with LLM2Vec so a single process can both chat
   and generate embeddings.
8. Executes lightweight generation + embedding sanity checks.
9. Writes a configuration file to simplify reloads in downstream projects.

The defaults are tuned for an RTX 2070 (8GB). Configure behaviour through the
environment variables documented near the top of this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ------------------------ User knobs (env or edit) ------------------------
MODEL_ID = os.environ.get("MODEL_ID", "cognitivecomputations/Dolphin3.0-Llama3.1-8B")
DATASET_ID = os.environ.get("DATASET_ID", "yahma/alpaca-cleaned")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs_dolphin8b")).resolve()
OFFLOAD_DIR = Path(os.environ.get("OFFLOAD_DIR", "./offload")).resolve()
VRAM_BUDGET_MB = int(os.environ.get("VRAM_BUDGET_MB", "7300"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "1.0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", "1.0"))
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))
SAVE_MERGED_FP16 = os.environ.get("SAVE_MERGED_FP16", "0") == "1"
EXPORT_GGUF = os.environ.get("EXPORT_GGUF", "0") == "1"
GGUF_DIR = Path(os.environ.get("GGUF_DIR", "gguf_out")).resolve()
GGUF_METHOD = os.environ.get("GGUF_METHOD", "q4_k_m")
RUN_TINY_TESTS = os.environ.get("RUN_TINY_TESTS", "1") == "1"
# -------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64"
)


def _have(module: str) -> bool:
    """Return True when *module* can be imported."""

    from importlib.util import find_spec

    return find_spec(module) is not None


def _pip_install(spec: str) -> None:
    """Install *spec* using pip in the current interpreter."""

    print(f"[deps] Installing missing dependency: {spec}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", spec], check=True)


def ensure_dependencies() -> None:
    """Install runtime dependencies on-demand (no version pins)."""

    required = (
        "torch",
        "transformers",
        "datasets",
        "peft",
        "bitsandbytes",
        "llm2vec",
        "sentencepiece",
    )
    optional = ("unsloth",)

    for module in required:
        if not _have(module):
            _pip_install(module)

    for module in optional:
        if not _have(module):
            try:
                _pip_install(module)
            except (
                subprocess.CalledProcessError
            ) as exc:  # pragma: no cover - optional path
                print(f"[deps] Optional dependency {module} failed to install: {exc}")


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    as_float = float(value)
    while as_float >= 1024 and idx < len(units) - 1:
        as_float /= 1024
        idx += 1
    return f"{as_float:.2f} {units[idx]}"


def print_env_summary() -> None:
    import torch

    print("=== Environment ===")
    print(sys.version)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            "GPU:",
            props.name,
            "VRAM=",
            f"{props.total_memory / (1024 ** 3):.2f} GB",
        )
        print("CUDA:", torch.version.cuda or "none", "Torch:", torch.__version__)
        allocated = torch.cuda.memory_allocated()
        print("Currently allocated:", _format_bytes(allocated))
    else:
        print("CUDA not available; GPU execution is required for this workflow.")


# Ensure dependencies before importing heavy modules so we can use them below.
ensure_dependencies()

import torch
from datasets import Dataset, load_dataset
from llm2vec import LLM2Vec
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# safer SDPA backend for Turing (RTX 2070 -> sm_75)
try:  # pragma: no cover - env specific
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


def bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_4bit(model_id: str, vram_mib: int, offload_dir: Path):
    print(f"[model] Loading {model_id} in 4-bit with {vram_mib} MiB budget")
    max_memory = {0: f"{vram_mib}MiB", "cpu": "48GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        quantization_config=bnb_4bit_config(),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def summarize_device_map(model: torch.nn.Module) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        print("[model] Device map unavailable (likely single-device deployment)")
        return
    counter = Counter(device_map.values())
    total_layers = sum(counter.values())
    gpu_layers = sum(v for k, v in counter.items() if str(k).startswith("cuda"))
    cpu_layers = counter.get("cpu", 0)
    print("[model] Device map summary:", dict(counter))
    if total_layers:
        print(
            f"[model] GPU layers: {gpu_layers}/{total_layers}"
            f" ({(gpu_layers / total_layers) * 100:.1f}%)"
        )
        print(
            f"[model] CPU layers: {cpu_layers}/{total_layers}"
            f" ({(cpu_layers / total_layers) * 100:.1f}%)"
        )


def _to_prompt_completion(ex: dict) -> dict:
    instruction = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ""
    user_input = ex.get("input") or ex.get("context") or ""
    output = ex.get("output") or ex.get("response") or ex.get("answer") or ""
    prompt = (
        f"{instruction}\n\n{user_input}"
        if user_input and user_input.strip()
        else instruction
    )
    return {"prompt": prompt, "completion": output}


def build_sft_dataset(dataset: Dataset, tokenizer, max_seq_len: int) -> Dataset:
    if not ({"prompt", "completion"} <= set(dataset.column_names)):
        dataset = dataset.map(
            _to_prompt_completion, remove_columns=dataset.column_names
        )

    def build(example: dict) -> dict:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt_only = f"<|im_start|>user\n{example['prompt']}\n<|im_end|>\n<|im_start|>assistant\n"
            full_text = f"{prompt_only}{example['completion']}<|im_end|>"

        prompt_enc = tokenizer(
            prompt_only,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        )
        full_enc = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.copy()
        prompt_length = min(len(prompt_enc["input_ids"]), len(labels))
        for idx in range(prompt_length):
            labels[idx] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(
        build,
        remove_columns=dataset.column_names,
        desc="Tokenizing & masking dataset",
    )
    tokenized.set_format(type="torch")
    return tokenized


def _subset_dataset(dataset: Dataset, fraction: float) -> Dataset:
    if 0 < fraction < 1.0:
        take = max(1, int(len(dataset) * fraction))
        print(f"[data] Using subset: {take}/{len(dataset)} examples")
        return dataset.select(range(take))
    return dataset


def prepare_dataset(tokenizer, max_seq_len: int) -> Dataset:
    print(f"[data] Loading dataset: {DATASET_ID}")
    try:
        dataset = load_dataset(DATASET_ID, split="train")
    except Exception:
        dataset = load_dataset(DATASET_ID)
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
    dataset = _subset_dataset(dataset, TRAIN_FRACTION)
    print("[data] Dataset size:", len(dataset))
    return build_sft_dataset(dataset, tokenizer, max_seq_len)


def prepare_model_for_training(model: torch.nn.Module) -> torch.nn.Module:
    print("[train] Preparing model for QLoRA training")
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model


def train_model(model, dataset: Dataset) -> Trainer:
    print("[train] Starting training loop")
    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        logging_steps=25,
        save_steps=250,
        save_total_limit=1,
        report_to=[],
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        torch_empty_cache_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    print("[train] Training complete")
    return trainer


def save_lora_and_tokenizer(trainer: Trainer, tokenizer) -> None:
    print("[save] Writing LoRA adapters and tokenizer to", OUTPUT_DIR)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


def merge_to_fp16(model_id: str, adapters_dir: Path, tokenizer) -> Path | None:
    if not SAVE_MERGED_FP16:
        return None
    print("[merge] Creating merged FP16 checkpoint (CPU)")
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    merged = PeftModel.from_pretrained(base, str(adapters_dir)).merge_and_unload()
    out_dir = adapters_dir / "merged_fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("[merge] Saved merged checkpoint to", out_dir)
    return out_dir


def export_gguf_if_requested(source_dir: Path) -> None:
    if not EXPORT_GGUF:
        return
    if not _have("unsloth"):
        print("[gguf] Skipping GGUF export: unsloth not installed")
        return
    print(f"[gguf] Exporting GGUF to {GGUF_DIR} with method {GGUF_METHOD}")
    try:
        from unsloth import FastModel

        GGUF_DIR.mkdir(parents=True, exist_ok=True)
        model, tokenizer = FastModel.from_pretrained(
            str(source_dir), max_seq_length=MAX_SEQ_LEN
        )
        model.save_pretrained_gguf(
            str(GGUF_DIR),
            tokenizer=tokenizer,
            quantization_method=GGUF_METHOD,
        )
        print("[gguf] GGUF export complete")
    except Exception as exc:  # pragma: no cover - optional path
        print(f"[gguf] Export failed (non-fatal): {exc}")


def wrap_with_llm2vec(model, tokenizer) -> LLM2Vec:
    print("[embed] Initialising LLM2Vec wrapper (shared model instance)")
    return LLM2Vec(
        model, tokenizer, pooling_mode="mean", max_length=min(512, MAX_SEQ_LEN)
    )


def _decode_assistant_reply(tokenizer, generated_ids: torch.Tensor) -> str:
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant")[-1].strip()
    return text.strip()


def run_sanity_checks(model, tokenizer, llm2vec: LLM2Vec) -> None:
    if not RUN_TINY_TESTS:
        return
    print("[tests] Running tiny generation + embedding checks")
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Say hello in seven words."},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "User: Say hello in seven words.\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        model.eval()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        print("[tests] Generation:", _decode_assistant_reply(tokenizer, generated))
    except Exception as exc:  # pragma: no cover - optional path
        print(f"[tests] Generation sanity check failed (non-fatal): {exc}")

    try:
        embeddings = llm2vec.encode(["a lightweight embedding check"])
        print("[tests] Embedding shape:", tuple(embeddings.shape))
    except Exception as exc:  # pragma: no cover - optional path
        print(f"[tests] Embedding sanity check failed (non-fatal): {exc}")


def write_wrapper_config(adapters_dir: Path) -> None:
    config = {
        "base_model_id": MODEL_ID,
        "adapters_dir": str(adapters_dir),
        "quantization": "bnb-4bit nf4 double-quant fp16 compute",
        "max_seq_len": MAX_SEQ_LEN,
        "vram_budget_mb": VRAM_BUDGET_MB,
        "notes": (
            "Reload the base model in 4-bit with BitsAndBytes, then attach the LoRA "
            "adapters from adapters_dir. Wrap the resulting model with LLM2Vec for "
            "shared chat + embedding workloads."
        ),
    }
    config_path = adapters_dir / "wrapper_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print("[save] Wrapper configuration written to", config_path)


def main() -> None:
    print_env_summary()

    model, tokenizer = load_model_4bit(MODEL_ID, VRAM_BUDGET_MB, OFFLOAD_DIR)
    summarize_device_map(model)

    dataset = prepare_dataset(tokenizer, MAX_SEQ_LEN)
    model = prepare_model_for_training(model)
    trainer = train_model(model, dataset)

    save_lora_and_tokenizer(trainer, tokenizer)
    merged_dir = merge_to_fp16(MODEL_ID, OUTPUT_DIR, tokenizer)
    export_source = merged_dir if merged_dir is not None else OUTPUT_DIR
    export_gguf_if_requested(export_source)

    llm2vec_wrapper = wrap_with_llm2vec(trainer.model, tokenizer)
    run_sanity_checks(trainer.model, tokenizer, llm2vec_wrapper)
    write_wrapper_config(OUTPUT_DIR)

    print("\n[done] Artefacts available in", OUTPUT_DIR)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Run examples
# ---------------------------------------------------------------------------
#
# (Recommended) create and activate a virtual environment first:
#   python -m venv .venv && source .venv/bin/activate
#
# Install base dependencies once (the script also installs on-demand):
#   pip install --upgrade transformers datasets peft bitsandbytes llm2vec unsloth sentencepiece
#
# Suggested environment tweaks for 8GB cards:
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
#   export CUDA_VISIBLE_DEVICES=0
#
# Execute with safe defaults:
#   python dolphin_llm2vec_pipeline.py
#
# Useful environment variables:
#   VRAM_BUDGET_MB=7300        # lower if you still see OOM (e.g., 7100 is very safe)
#   MAX_SEQ_LEN=1024           # raise if stable, lower to reduce memory pressure
#   BATCH_SIZE=1               # keep at 1 on 8GB; increase GRAD_ACCUM for larger effective batch
#   GRAD_ACCUM=8               # tunes effective batch size
#   EPOCHS=1                   # or configure MAX_STEPS instead
#   TRAIN_FRACTION=0.1         # quick smoke run using 10% of the dataset
#   SAVE_MERGED_FP16=1         # additionally materialise a merged FP16 checkpoint
#   EXPORT_GGUF=1              # export a GGUF bundle via Unsloth (requires SAVE_MERGED_FP16 for best results)
#
# Artefact layout:
#   outputs_dolphin8b/
#     adapter_config.json, LoRA adapter weights, tokenizer files
#     wrapper_config.json
#     merged_fp16/ (optional)
#     gguf_out/ (optional, via EXPORT_GGUF=1)
#
# Reload pattern in your project:
#   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#   from peft import PeftModel
#   import torch
#
#   base = AutoModelForCausalLM.from_pretrained(
#       MODEL_ID,
#       device_map="auto",
#       max_memory={0: "7300MiB", "cpu": "48GiB"},
#       offload_folder="./offload",
#       quantization_config=BitsAndBytesConfig(
#           load_in_4bit=True,
#           bnb_4bit_use_double_quant=True,
#           bnb_4bit_quant_type="nf4",
#           bnb_4bit_compute_dtype=torch.float16,
#       ),
#       trust_remote_code=True,
#   )
#   tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
#   tok.pad_token = tok.eos_token
#   model = PeftModel.from_pretrained(base, "outputs_dolphin8b")
#
#   from llm2vec import LLM2Vec
#   l2v = LLM2Vec(model, tok, pooling_mode="mean", max_length=512)
#   embeddings = l2v.encode(["hello world"])
# ---------------------------------------------------------------------------
