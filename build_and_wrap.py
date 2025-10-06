#!/usr/bin/env python3
"""Utility to fine-tune Dolphin3.0-Llama3.1-8B with QLoRA and bundle helpers.

This script provides an end-to-end local pipeline designed for an RTX 2070
class GPU (8GB VRAM). It performs the following steps:

1. Installs required dependencies if they are missing (unless disabled).
2. Loads the base model using 4-bit quantisation for memory efficiency.
3. Prepares the model for QLoRA fine-tuning using PEFT.
4. Loads and tokenises an instruction dataset, masking the prompt tokens.
5. Fine-tunes the model with Hugging Face Trainer.
6. Saves the resulting LoRA adapters and tokenizer.
7. Optionally merges the adapters back into a FP16 model.
8. Optionally exports the merged model to GGUF using Unsloth.
9. Generates a reusable wrapper that provides both chat generation and
   embeddings (via LLM2Vec) from a single model instance.
10. Writes integration documentation for downstream consumers.

Environment variables allow overriding defaults without interactive prompts.
The defaults are tuned for quick iteration while fitting inside 8GB of VRAM.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from monGARS.mlops.artifacts import (
    WrapperConfig,
    render_output_bundle_readme,
    write_wrapper_bundle,
)

# -------------------- User-tunable defaults (override with env vars) --------------------
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
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "./offload")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./out")
EXPORT_MERGED_FP16 = os.environ.get("EXPORT_MERGED_FP16", "0") == "1"
EXPORT_GGUF = os.environ.get("EXPORT_GGUF", "0") == "1"
GGUF_DIR = os.environ.get("GGUF_DIR", "./out/gguf")
GGUF_METHOD = os.environ.get("GGUF_METHOD", "q4_k_m")
AUTO_INSTALL = os.environ.get("AUTO_INSTALL", "1") == "1"


# Memory-friendly CUDA allocator
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64"
)
Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# -------------------- Helpers --------------------
def have(mod: str) -> bool:
    """Return True when the given module can be imported."""

    return importlib.util.find_spec(mod) is not None


def pip_install(spec: str) -> None:
    """Install a dependency using pip."""

    print(f"[setup] Installing: {spec}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", spec], check=True)


def ensure_pkgs() -> None:
    """Ensure required (and optional) dependencies are present."""

    required = [
        ("torch", "torch"),
        ("transformers", "transformers>=4.44"),
        ("datasets", "datasets"),
        ("peft", "peft>=0.11"),
        ("bitsandbytes", "bitsandbytes>=0.44.1"),
        ("llm2vec", "llm2vec"),
    ]
    optional = [("unsloth", "unsloth")]

    for module, spec in required:
        if not have(module):
            if AUTO_INSTALL:
                pip_install(spec)
            else:
                print(f"[setup] Missing {module}. Install {spec} and re-run.")
                sys.exit(1)

    for module, spec in optional:
        if not have(module) and AUTO_INSTALL:
            try:
                pip_install(spec)
            except Exception as exc:  # pragma: no cover - best effort install
                print(f"[warn] Could not install optional {spec}: {exc}")
        elif not have(module):
            print(
                f"[info] Optional {module} not installed (skip features that require it)."
            )


def print_env() -> None:
    """Print the runtime environment and validate CUDA availability."""

    print("=== Environment Check ===")
    print(sys.version)
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            print(f"GPU: {props.name}, VRAM={vram_gb:.2f} GB")
            print(f"CUDA: {torch.version.cuda or 'none'}, Torch: {torch.__version__}")
        else:
            print(
                "CUDA not available. This script requires an NVIDIA GPU for finetuning 8B."
            )
            sys.exit(1)

        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:  # pragma: no cover - depends on torch version
            pass
    except Exception as exc:
        print(f"[error] torch import failed: {exc}")
        sys.exit(1)


# -------------------- Main pipeline steps --------------------
def load_model_4bit(model_id: str, vram_mib: int, offload_dir: str):
    """Load a base model in 4-bit precision with an explicit VRAM budget."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    max_memory = {0: f"{vram_mib}MiB", "cpu": "48GiB"}
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"[load] {model_id} in 4-bit with budget {vram_mib} MiB")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_dir,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    return model, tokenizer


def prep_lora(model):
    """Enable gradient checkpointing and attach LoRA adapters."""

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    print("[lora] attached LoRA adapters (r=32, alpha=32).")
    return model


def load_dataset_and_tokenize(
    tokenizer, dataset_name: str, max_seq_len: int, fraction: float
):
    """Load a dataset, convert to prompt/completion, and tokenise."""

    from datasets import load_dataset

    print(f"[data] loading {dataset_name} (fraction={fraction:.2f})")

    if ":" in dataset_name:
        dataset = load_dataset(dataset_name)
        dataset = dataset.get("train") or next(iter(dataset.values()))
    else:
        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception:
            dataset = load_dataset(dataset_name)
            dataset = dataset.get("train") or next(iter(dataset.values()))

    if 0 < fraction < 1.0:
        total = len(dataset)
        take = max(1000, int(total * fraction))
        dataset = dataset.select(range(take))
        print(f"[data] subset: {take}/{total} examples")

    def to_prompt_completion(example: Dict[str, Any]) -> Dict[str, str]:
        instruction = (
            example.get("instruction")
            or example.get("prompt")
            or example.get("question")
            or ""
        )
        input_text = example.get("input") or example.get("context") or ""
        output = (
            example.get("output")
            or example.get("response")
            or example.get("answer")
            or ""
        )
        prompt = (
            f"{instruction}\n\n{input_text}"
            if input_text and input_text.strip()
            else instruction
        )
        return {"prompt": prompt, "completion": output}

    dataset = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)

    def tokenise(example: Dict[str, str]) -> Dict[str, Any]:
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
            prompt_only = f"User: {example['prompt']}\nAssistant:"
            full_text = f"{prompt_only} {example['completion']}"

        enc_prompt = tokenizer(
            prompt_only,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        )
        enc_full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_attention_mask=True,
        )

        input_ids = enc_full["input_ids"]
        attention = enc_full["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(enc_prompt["input_ids"]), len(labels))
        for idx in range(prompt_len):
            labels[idx] = -100

        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    dataset_tok = dataset.map(
        tokenise, remove_columns=dataset.column_names, desc="[data] tokenize+mask"
    )
    dataset_tok.set_format(type="torch")
    print(f"[data] tokenized: {len(dataset_tok)} samples")
    return dataset_tok


def train_lora(model, dataset_tok, out_dir: str):
    """Fine-tune the model using Hugging Face Trainer."""

    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator

    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS if MAX_STEPS <= 0 else 1.0,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        logging_steps=25,
        save_steps=250,
        save_total_limit=1,
        report_to=[],
        bf16=bf16_supported,
        fp16=not bf16_supported,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        torch_empty_cache_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_tok,
        data_collator=default_data_collator,
    )
    print("[train] starting SFT…")
    trainer.train()
    print("[train] done.")
    return trainer


def save_adapters_and_tokenizer(model, tokenizer, out_dir: str) -> None:
    """Persist the LoRA adapters and tokenizer."""

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[save] LoRA adapters + tokenizer saved to: {out_dir}")


def merge_to_fp16_and_save(base_model_id: str, lora_dir: str, out_dir: str) -> bool:
    """Merge LoRA adapters into a full-precision model and save it."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[merge] loading base FP16 on CPU and applying LoRA for merge…")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype="auto", device_map={"": "cpu"}
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    peft_model = PeftModel.from_pretrained(base, lora_dir)
    merged = peft_model.merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[merge] merged model (FP16) saved to: {out_dir}")
    return True


def export_gguf(
    unsloth_available: bool, merged_dir: str, gguf_dir: str, method: str
) -> bool:
    """Export a merged model to GGUF format when Unsloth is present."""

    if not unsloth_available:
        print("[gguf] Unsloth not installed; skipping GGUF export.")
        return False

    try:
        from unsloth import FastModel
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[gguf] Unsloth missing: {exc}")
        return False

    print(f"[gguf] exporting {merged_dir} -> {gguf_dir} ({method})")
    Path(gguf_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastModel.from_pretrained(merged_dir, max_seq_length=MAX_SEQ_LEN)
    model.save_pretrained_gguf(
        gguf_dir, tokenizer=tokenizer, quantization_method=method
    )
    print(f"[gguf] saved GGUF files to: {gguf_dir}")
    return True


def create_wrapper_module(config: WrapperConfig, out_root: str) -> None:
    """Create a wrapper exposing chat generation and embeddings."""

    bundle = write_wrapper_bundle(config, Path(out_root))
    print(f"[wrap] wrote wrapper module to: {bundle['module'].parent}")


def write_run_readme(
    config: WrapperConfig,
    out_root: str,
    *,
    merged_fp16: bool,
    gguf_done: bool,
) -> None:
    """Write a README describing the produced artifacts."""

    readme = render_output_bundle_readme(
        config,
        merged_fp16=merged_fp16,
        gguf_enabled=gguf_done,
        gguf_method=GGUF_METHOD,
    )
    target = Path(out_root) / "README_outputs.md"
    target.write_text(readme, encoding="utf-8")
    print(f"[docs] wrote {target}")


def main() -> None:
    """Run the full fine-tuning and packaging workflow."""

    ensure_pkgs()
    print_env()

    model, tokenizer = load_model_4bit(MODEL_ID, VRAM_BUDGET_MB, OFFLOAD_DIR)
    model = prep_lora(model)
    dataset_tok = load_dataset_and_tokenize(
        tokenizer, DATASET_NAME, MAX_SEQ_LEN, TRAIN_FRACTION
    )
    lora_out_path = Path(OUTPUT_DIR) / "chat_lora"
    lora_out = str(lora_out_path)
    trainer = train_lora(model, dataset_tok, lora_out)
    save_adapters_and_tokenizer(trainer.model, tokenizer, lora_out)

    merged_dir_path = Path(OUTPUT_DIR) / "merged_fp16"
    merged_dir = str(merged_dir_path)
    merged = False
    if EXPORT_MERGED_FP16:
        merged = merge_to_fp16_and_save(MODEL_ID, lora_out, merged_dir)

    gguf_done = False
    if EXPORT_GGUF and EXPORT_MERGED_FP16:
        gguf_done = export_gguf(have("unsloth"), merged_dir, GGUF_DIR, GGUF_METHOD)
    elif EXPORT_GGUF:
        print("[gguf] Skipping: EXPORT_GGUF=1 requires EXPORT_MERGED_FP16=1")

    wrapper_config = WrapperConfig(
        base_model_id=MODEL_ID,
        lora_dir=lora_out_path.resolve(),
        max_seq_len=MAX_SEQ_LEN,
        vram_budget_mb=VRAM_BUDGET_MB,
        offload_dir=Path(OFFLOAD_DIR).resolve(),
    )
    create_wrapper_module(wrapper_config, OUTPUT_DIR)
    write_run_readme(
        wrapper_config,
        OUTPUT_DIR,
        merged_fp16=merged,
        gguf_done=gguf_done,
    )

    print("\n=== ALL DONE ===")
    print(f"- LoRA adapters:        {lora_out}")
    if merged:
        print(f"- Merged FP16 model:    {merged_dir}")
    if gguf_done:
        print(f"- GGUF export:          {GGUF_DIR}")
    wrapper_module = Path(OUTPUT_DIR) / "wrapper" / "project_wrapper.py"
    print(f"- Wrapper module:       {wrapper_module}")
    outputs_readme = Path(OUTPUT_DIR) / "README_outputs.md"
    print(f"- Outputs README:       {outputs_readme}")


if __name__ == "__main__":
    main()
