#!/usr/bin/env python3
"""
train_dolphin_unsloth_multimodule.py

End-to-end fine-tuning for Dolphin 3.0 (Llama 3.1 8B) using Unsloth + LoRA.
- Ingests your uploaded train/val JSONL (supports prompt/response, instruction/input/output, or messages[])
- Works with module-tagged prompts like: [MOD=Cortex], [MOD=Hippocampus], etc.
- Trains with 4-bit base + LoRA, with safe fallbacks
- Exports a minimal LLM2Vec-style wrapper (generate + embed via mean pooling)

USAGE (typical):
  python scripts/train_dolphin_unsloth_multimodule.py \
    --train-file /path/to/train.jsonl \
    --val-file /path/to/val.jsonl \
    --out-dir out/monGARS_dolphin_multimodule \
    --epochs 2 --lr 1.5e-4 --cutoff-len 4096 \
    --merge-and-save  # optional full-weights export

Requirements:
  pip install "unsloth>=2025.10.1" "transformers>=4.56.0" "datasets>=2.20.0" "accelerate>=0.34.0" "peft>=0.13.0" torch
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from datasets import load_dataset

LOGGER = logging.getLogger("unsloth_multimodule")

BASE_MODEL = "cognitivecomputations/Dolphin3.0-Llama3.1-8B"


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _detect_device_map():
    """Return an automatic device map suitable for most environments."""

    return "auto"


def _load_unsloth(base_model: str, max_len: int, try_4bit: bool = True):
    from unsloth import FastLanguageModel

    kwargs = dict(
        model_name=base_model,
        max_seq_length=max_len,
        dtype=None,
        device_map=_detect_device_map(),
        trust_remote_code=True,
    )
    if try_4bit:
        kwargs["load_in_4bit"] = True
    return FastLanguageModel.from_pretrained(**kwargs)


def _get_peft(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing=True,
    )


# ---------- Data loading / normalization ----------
def _normalize_item(rec: Dict[str, Any]) -> Dict[str, str] | None:
    """
    Accepts any of:
      {prompt, response}
      {instruction, input, output}
      {messages: [{role, content}, ...]}
    Returns dict with keys: instruction, input, output
    """

    if "messages" in rec and isinstance(rec["messages"], list):
        msgs = rec["messages"]
        instr = None
        inp = ""
        out = None
        user_parts = []
        for m in msgs:
            role = (m.get("role") or "").lower()
            content = (m.get("content") or "").strip()
            if role == "user":
                user_parts.append(content)
            elif role == "assistant":
                out = content
        if user_parts:
            instr = "\n\n".join(user_parts)
        if instr and out:
            return {"instruction": instr, "input": "", "output": out}

    if "prompt" in rec and "response" in rec:
        instr = (rec.get("prompt") or "").strip()
        out = (rec.get("response") or "").strip()
        if instr and out:
            return {"instruction": instr, "input": "", "output": out}

    if "instruction" in rec and "output" in rec:
        instr = (rec.get("instruction") or "").strip()
        inp = (
            (rec.get("input") or "").strip()
            if isinstance(rec.get("input"), str)
            else ""
        )
        out = rec.get("output")
        if not isinstance(out, str):
            out = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
        out = out.strip()
        if instr and out:
            return {"instruction": instr, "input": inp, "output": out}

    return None


def _format_prompt_for_chat(example: Dict[str, str]) -> Dict[str, str]:
    """Build a Llama 3 style chat with system + user turns; keep module tags in instruction."""

    instr = example["instruction"]
    inp = example.get("input", "")
    user = instr if not inp else f"{instr}\n\n[INPUT]\n{inp}"
    sysmsg = (
        "You are monGARS internal assistant. "
        "Follow the module contract indicated by tags like [MOD=Cortex], [MOD=Hippocampus], etc. "
        "Do not speculate beyond module specifications."
    )
    prompt = (
        f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {"text": prompt + example["output"] + "<|im_end|>\n"}


def _load_as_dataset(train_file: str, val_file: str | None):
    """Load JSONL files into HF datasets and normalize records."""

    data_files = {"train": train_file}
    if val_file:
        data_files["validation"] = val_file
    raw = load_dataset("json", data_files=data_files)

    for split in list(raw.keys()):
        raw[split] = (
            raw[split]
            .map(
                lambda x: _normalize_item(x),
                remove_columns=raw[split].column_names,
            )
            .filter(lambda r: r is not None)
        )

    for split in list(raw.keys()):
        raw[split] = raw[split].map(
            _format_prompt_for_chat,
            remove_columns=raw[split].column_names,
        )
    return raw


# ---------- Tokenization ----------
def _tokenize(tokenizer, ds, max_len: int):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    out = {}
    for split in list(ds.keys()):
        out[split] = ds[split].map(_tok, batched=True, remove_columns=["text"])
    return out


# ---------- LLM2Vec-style wrapper export ----------
LLM2VEC_PY = r'''# llm2vec_wrapper.py
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


class LLM2Vec:
    """
    Minimal chat + embed wrapper.
    - generate(prompt, ...) -> str
    - embed(texts) -> torch.Tensor [N, hidden_size] (mean-pooled last layer)
    """

    def __init__(self, base_dir, prefer_merged=False, device=None, load_in_4bit=True):
        self.base_dir = str(base_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok_dir = f"{self.base_dir}/tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

        if prefer_merged and (Path(f"{self.base_dir}/merged").exists()):
            model_dir = f"{self.base_dir}/merged"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        else:
            base_model = "cognitivecomputations/Dolphin3.0-Llama3.1-8B"
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_4bit=load_in_4bit,
                device_map="auto",
                trust_remote_code=True,
            )
            if PeftModel is None:
                raise RuntimeError("peft not available; cannot load LoRA adapter.")
            self.model = PeftModel.from_pretrained(self.model, f"{self.base_dir}/lora_adapter")
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=512, temperature=0.2, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.inference_mode()
    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        if hasattr(self.model, "transformer"):
            outputs = self.model.transformer(**batch, output_hidden_states=True)
        else:
            outputs = self.model(**batch, output_hidden_states=True)
        last = outputs.hidden_states[-1]
        mask = batch["attention_mask"].unsqueeze(-1)
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        emb = summed / counts
        return emb
'''


def _export_wrapper(out_dir: Path):
    wrap = out_dir / "wrapper"
    wrap.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer").mkdir(exist_ok=True)
    (wrap / "llm2vec_wrapper.py").write_text(LLM2VEC_PY, encoding="utf-8")
    (wrap / "config.json").write_text(
        json.dumps(
            {
                "name": "monGARS-LLM2Vec",
                "backbone": "Dolphin3.0-Llama3.1-8B",
                "adapter_dir": "lora_adapter",
                "supports_merged": True,
                "embed_strategy": "last_hidden_mean_pool",
                "module_tag_format": "[MOD=<Module>]",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (wrap / "README.md").write_text(
        "Minimal chat+embed wrapper. Usage:\n"
        "from llm2vec_wrapper import LLM2Vec\n"
        "w = LLM2Vec(base_dir='..', prefer_merged=False)\n"
        "print(w.generate('Bonjour [MOD=Hippocampus] Rappelle-moi le dernier contexte.'))\n"
        "vec = w.embed('On va au dépanneur.')\n"
        "print(vec.shape)\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrapper bundle created at %s", wrap)
    return wrap


# ---------- Main train routine ----------
def main():
    _setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--out-dir", default="out/monGARS_dolphin_multimodule")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--cutoff-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--merge-and-save", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading datasets")
    ds = _load_as_dataset(args.train_file, args.val_file)

    LOGGER.info("Loading base model (Unsloth)")
    try:
        model, tokenizer = _load_unsloth(
            args.base_model, args.cutoff_len, try_4bit=True
        )
    except Exception as e:
        LOGGER.warning(
            "4-bit load failed (%s). Retrying without 4-bit on current device.", e
        )
        model, tokenizer = _load_unsloth(
            args.base_model, args.cutoff_len, try_4bit=False
        )

    LOGGER.info(
        "Attaching LoRA adapters r=%d alpha=%d dropout=%.3f",
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    )
    model = _get_peft(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )

    LOGGER.info("Tokenizing")
    tok_ds = _tokenize(tokenizer, ds, args.cutoff_len)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    LOGGER.info("Preparing trainer")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        evaluation_strategy="steps" if "validation" in tok_ds else "no",
        eval_steps=100,
        save_steps=300,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds.get("validation"),
    )

    LOGGER.info("Starting training")
    trainer.train()
    LOGGER.info("Training complete")

    LOGGER.info("Saving LoRA adapter & tokenizer")
    (out_dir / "lora_adapter").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    if args.merge_and_save:
        try:
            from unsloth import FastLanguageModel

            merged = FastLanguageModel.merge_and_unload(model)
            (out_dir / "merged").mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(out_dir / "merged"))
            tokenizer.save_pretrained(str(out_dir / "merged"))
            LOGGER.info("Merged full weights saved")
        except Exception as e:
            LOGGER.warning("Merging failed: %s", e)

    _export_wrapper(out_dir)
    LOGGER.info("All done → %s", out_dir)


if __name__ == "__main__":
    main()
