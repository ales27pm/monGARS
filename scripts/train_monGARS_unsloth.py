#!/usr/bin/env python3
"""Fine-tune Dolphin-X1-8B with Unsloth on the multi-module dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from unsloth import FastLanguageModel

from datasets import load_dataset

BASE_MODEL = "dphn/Dolphin-X1-8B"
SYSTEM_PROMPT = (
    "You are monGARS internal assistant. Follow the module contract indicated by "
    "tags like [MOD=...]."
)


def load_jsonl_as_dataset(dataset_dir: Path):
    data_files: dict[str, str] = {}
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"
    if train_file.exists():
        data_files["train"] = str(train_file)
    if val_file.exists():
        data_files["validation"] = str(val_file)
    if not data_files:
        raise SystemExit(f"No dataset files found in {dataset_dir}.")
    return load_dataset("json", data_files=data_files)


def build_prompt(example: dict) -> dict[str, str]:
    instruction = example.get("instruction", "")
    input_section = example.get("input", "")
    if input_section:
        user_block = f"{instruction}\n\n[INPUT]\n{input_section}"
    else:
        user_block = instruction
    assistant_output = example.get("output", "")
    prompt = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_block}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"text": prompt + assistant_output + "<|im_end|>\n"}


def tokenize_batch(tokenizer: AutoTokenizer, cutoff_len: int, batch: dict) -> dict:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=cutoff_len,
    )


def configure_tokenizer(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/multimodule")
    parser.add_argument("--out_dir", default="out/monGARS_dolphin_finetuned")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--merge_and_save",
        action="store_true",
        help="Merge LoRA adapter back into base weights for full-weight export.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl_as_dataset(data_dir)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=args.cutoff_len,
        load_in_4bit=True,
        dtype=None,
        device_map="auto",
        trust_remote_code=True,
    )

    configure_tokenizer(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing=True,
    )

    formatted_dataset = dataset.map(build_prompt)
    train_columns = formatted_dataset["train"].column_names
    columns_to_remove = [col for col in train_columns if col != "text"]
    tokenized_dataset = formatted_dataset.map(
        lambda batch: tokenize_batch(tokenizer, args.cutoff_len, batch),
        batched=True,
        remove_columns=columns_to_remove,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not supports_bf16

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        evaluation_strategy="steps" if "validation" in tokenized_dataset else "no",
        eval_steps=100,
        save_steps=300,
        save_total_limit=2,
        bf16=supports_bf16,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
    )

    trainer.train()

    model.save_pretrained(str(out_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    if args.merge_and_save:
        merged_model = FastLanguageModel.merge_and_unload(model)
        merged_path = out_dir / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))

    print(f"[DONE] Saved artifacts to {out_dir}")
    print("  - lora_adapter/")
    if args.merge_and_save:
        print("  - merged/")


if __name__ == "__main__":
    main()
