"""Dataset helpers for supervised fine-tuning pipelines."""

from __future__ import annotations

import logging
from typing import Any, Callable

from transformers import PreTrainedTokenizerBase

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


PROMPT_KEYS = ("instruction", "prompt", "question")
INPUT_KEYS = ("input", "context")
OUTPUT_KEYS = ("output", "response", "answer")


def _extract_field(example: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _format_prompt_completion(example: dict[str, Any]) -> dict[str, str]:
    instruction = _extract_field(example, PROMPT_KEYS)
    additional = _extract_field(example, INPUT_KEYS)
    output = _extract_field(example, OUTPUT_KEYS)
    prompt = f"{instruction}\n\n{additional}" if additional else instruction
    return {"prompt": prompt, "completion": output}


def _tokenize_pair(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> Callable[[dict[str, str]], dict[str, Any]]:
    def builder(example: dict[str, str]) -> dict[str, Any]:
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

        prompt_tokens = tokenizer(
            prompt_only,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        )
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = full_tokens["input_ids"]
        attention = full_tokens["attention_mask"]
        labels = list(input_ids)
        k = min(len(prompt_tokens["input_ids"]), len(labels))
        for idx in range(k):
            labels[idx] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
        }

    return builder


def prepare_instruction_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    *,
    train_fraction: float = 1.0,
) -> Dataset:
    """Load and tokenise an instruction dataset for supervised fine-tuning."""

    logger.info(
        "Loading dataset",
        extra={"dataset": dataset_name, "train_fraction": train_fraction},
    )
    dataset: Dataset | DatasetDict = load_dataset(dataset_name)
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get("train") or next(iter(dataset.values()))
    if train_fraction and 0 < train_fraction < 1:
        total = len(dataset)
        desired = max(1, int(total * train_fraction))
        take = min(total, max(1000, desired))
        dataset = dataset.select(range(take))
        logger.info("Dataset subset selected", extra={"take": take, "total": total})

    if "prompt" not in dataset.column_names or "completion" not in dataset.column_names:
        dataset = dataset.map(
            _format_prompt_completion, remove_columns=dataset.column_names
        )

    tokenized = dataset.map(
        _tokenize_pair(tokenizer, max_seq_len),
        remove_columns=dataset.column_names,
        desc="tokenize+mask",
    )
    tokenized.set_format(type="torch")
    logger.info("Tokenised dataset", extra={"size": len(tokenized)})
    return tokenized
