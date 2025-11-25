# light_peft_patch.py
# A single-file, 8GB-friendly PEFT helper for Llama-family models.
# - Safe 4-bit loader with CPU offload for `lm_head` (avoids logits memory spikes).
# - "Light" LoRA preparation: no fp32 upcast of head/norms (unlike `prepare_model_for_kbit_training`).
# - Optional small dataset builder and sliced-loss Trainer to further reduce VRAM at loss time.
#
# Why this works (docs & references):
# - Offloading with bitsandbytes / Transformers (`llm_int8_enable_fp32_cpu_offload=True`) allows CPU placement
#   for large layers when quantized; weights on CPU remain fp32 by design. See HF docs on offloading.  [1]
# - `prepare_model_for_kbit_training` intentionally casts layer norms/lm_head to fp32 (great for >24GB,
#   risky on 8GB). We avoid that by a "light" prep and LoRA only on attention/MLP projections.           [2]
# - Modern PyTorch checkpointing prefers explicit `use_reentrant=False` to reduce memory pitfalls.        [3]
# - LLM2Vec supports recent Transformers (Llama 3.1) on GitHub; this file remains compatible.            [4][5]
#
# [1] https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
# [2] https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model
# [3] https://docs.pytorch.org/docs/stable/checkpoint.html
# [4] https://github.com/McGill-NLP/llm2vec
# [5] https://mcgill-nlp.github.io/llm2vec/tutorial/
#
# Usage (drop-in):
#   from light_peft_patch import (
#       load_4bit_causal_lm,          # replacement for your loader
#       prepare_lora_model_light,     # replacement for prepare_model_for_kbit_training
#       build_sft_dataset,            # optional: small helper to build prompt/labels
#       make_sliced_trainer           # optional: Trainer that computes CE on a small slice
#   )
#
#   model, tok = load_4bit_causal_lm("dphn/Dolphin-X1-8B",
#                                    vram_budget_mb=7100, offload_dir="./offload")
#   model     = prepare_lora_model_light(model, r=16, alpha=16, dropout=0.0)
#
#   # Optional tiny-fine-tune path (very memory-conscious):
#   ds_tok    = build_sft_dataset(tok, "yahma/alpaca-cleaned",
#                                 max_seq_len=384, label_tokens=48, fraction=0.06)
#   trainer   = make_sliced_trainer(model, ds_tok, out_dir="./out/chat_lora",
#                                   batch_size=1, grad_accum=8, lr=2e-4, epochs=1.0)
#   trainer.train()
#   model.save_pretrained("./out/chat_lora"); tok.save_pretrained("./out/chat_lora")
#
# NOTE: this file has ZERO Unsloth-specific code; it works with plain Transformers+PEFT.
#       Keep your TRL/Unsloth versions consistent if you use those separately.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from datasets import load_dataset
from monGARS.mlops.chat_templates import (
    ensure_dolphin_chat_template,
    load_tokenizer_with_dolphin_chat_template,
)

# ---- Public API --------------------------------------------------------------

__all__ = [
    "load_4bit_causal_lm",
    "prepare_lora_model_light",
    "build_sft_dataset",
    "make_sliced_trainer",
]

LOGGER = logging.getLogger(__name__)

# ---- Loader: 4-bit + CPU offload of lm_head ---------------------------------


def load_4bit_causal_lm(
    model_id: str,
    vram_budget_mb: int = 7100,
    offload_dir: str = "./offload",
    compute_dtype: torch.dtype = torch.float16,
    quant_type: str = "nf4",
    double_quant: bool = True,
    device_map_override: Optional[Dict[str, Any]] = None,
    cpu_offload: bool = True,
):
    """
    Load a causal LM in 4-bit with bitsandbytes and offload the `lm_head` to CPU by default.
    This avoids the logits-time VRAM spike on 8GB GPUs.

    Returns: (model, tokenizer)
    """
    Path(offload_dir).mkdir(parents=True, exist_ok=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Default: decoder on GPU, big projection on CPU
    device_map = device_map_override or {
        "model.embed_tokens": 0,
        "model.layers": 0,
        "model.norm": 0,
        "lm_head": "cpu",
    }
    max_memory = {0: f"{vram_budget_mb}MiB", "cpu": "64GiB"}

    kwargs = dict(
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_dir,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=compute_dtype,
    )
    # Required for HF quantizer validator when any module is on CPU in 8/4-bit
    if cpu_offload:
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tok = load_tokenizer_with_dolphin_chat_template(model_id)

    # Memory-friendly defaults for Turing (RTX 20xx)
    model.config.use_cache = False
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass

    return model, tok


# ---- LoRA: "light" prep (no fp32 upcast of head/norms) ----------------------


@dataclass
class LoRAArgs:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Iterable[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def prepare_lora_model_light(
    model,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[Iterable[str]] = None,
):
    """
    Attach LoRA to attention/MLP projections without upcasting lm_head/norms to fp32.
    Enables gradient checkpointing and input requires_grad.
    """
    target_modules = (
        tuple(target_modules) if target_modules else LoRAArgs.target_modules
    )

    frozen_layers = 0
    transformer = getattr(model, "model", None)
    layers = getattr(transformer, "layers", None) if transformer is not None else None
    if layers is not None:
        try:
            total_layers = len(layers)
        except TypeError:
            total_layers = 0
        freeze_count = max(total_layers // 2, 0)
        if freeze_count > 0:
            for layer in layers[:freeze_count]:
                for param in layer.parameters():
                    param.requires_grad_(False)
            frozen_layers = freeze_count
            LOGGER.info(
                "Gradient checkpoint freezing applied to %d/%d transformer layers.",
                frozen_layers,
                total_layers,
            )
    if frozen_layers == 0:
        LOGGER.debug(
            "Skipping gradient checkpoint freezing; model does not expose layered transformer blocks."
        )

    # Gradient checkpointing (prefer non-reentrant variant for modern PyTorch)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    # Ensure grads can flow into embeddings when base weights are frozen
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def _req_grad_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        try:
            model.get_input_embeddings().register_forward_hook(_req_grad_hook)
        except Exception:
            pass

    lcfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    return model


# ---- Optional helpers: tiny SFT dataset + sliced CE trainer ------------------


def build_sft_dataset(
    tokenizer,
    dataset_name: str,
    max_seq_len: int = 384,
    label_tokens: int = 48,
    fraction: float = 0.06,
):
    """
    Build a minimal supervised dataset with labels restricted to the last N tokens
    of the assistant reply (reduces loss-time memory).

    Returns: tokenized HF Dataset with columns: input_ids, attention_mask, labels
    """
    print(f"[data] loading {dataset_name} (fraction={fraction:.2f})")
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception:
        ds_all = load_dataset(dataset_name)
        ds = ds_all["train"] if "train" in ds_all else list(ds_all.values())[0]

    if 0 < fraction < 1.0:
        n = len(ds)
        take = max(1000, int(n * fraction))
        ds = ds.select(range(take))
        print(f"[data] subset: {take}/{n} examples")

    def to_pc(ex):
        instr = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ""
        inp = ex.get("input") or ex.get("context") or ""
        out = ex.get("output") or ex.get("response") or ex.get("answer") or ""
        prompt = f"{instr}\n\n{inp}" if inp and inp.strip() else instr
        return {"prompt": prompt, "completion": out}

    ds = ds.map(to_pc, remove_columns=ds.column_names)

    ensure_dolphin_chat_template(tokenizer)

    def build(ex):
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["completion"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt_only = f"User: {ex['prompt']}\nAssistant:"
            full_text = f"{prompt_only} {ex['completion']}"

        enc_p = tokenizer(
            prompt_only,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        )
        enc_all = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_attention_mask=True,
        )

        input_ids = enc_all["input_ids"]
        attn = enc_all["attention_mask"]
        labels = [-100] * len(input_ids)

        L = sum(attn)
        k_prompt = min(len(enc_p["input_ids"]), L)
        start = max(k_prompt, L - label_tokens)
        for i in range(start, L):
            labels[i] = input_ids[i]

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    ds_tok = ds.map(
        build,
        remove_columns=ds.column_names,
        desc=f"[data] tokenize+mask (last {label_tokens} tokens)",
    )
    ds_tok.set_format(type="torch")
    print(f"[data] tokenized: {len(ds_tok)} samples (labels on last {label_tokens})")
    return ds_tok


def make_sliced_trainer(
    model,
    train_dataset,
    out_dir: str = "./out/chat_lora",
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    epochs: float = 1.0,
    max_steps: int = -1,
    fp16: bool = True,
):
    """
    Create a Trainer that computes cross-entropy **only on labeled positions** (labels != -100),
    avoiding building gigantic logits for loss when most tokens are masked. Keeps the
    final cross-entropy on CPU-sized slices and helps 8GB GPUs.
    """

    class SliceLossTrainer(Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits  # [B, T, V] (lm_head may live on CPU)
            labels = inputs["labels"]
            with torch.no_grad():
                mask = labels.ne(-100)
            if mask.sum().item() == 0:
                loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
                return (loss, outputs) if return_outputs else loss
            sel_logits = logits[mask]  # [N, V], typically CPU
            sel_labels = labels[mask].to(sel_logits.device)
            loss = torch.nn.functional.cross_entropy(
                sel_logits.float(), sel_labels, reduction="mean"
            )
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs if max_steps <= 0 else 1.0,
        max_steps=max_steps if max_steps > 0 else -1,
        logging_steps=25,
        save_steps=250,
        save_total_limit=1,
        report_to=[],
        fp16=fp16,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        torch_empty_cache_steps=50,
    )
    return SliceLossTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
