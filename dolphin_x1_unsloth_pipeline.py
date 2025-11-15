#!/usr/bin/env python3
"""Utility script for Unsloth-backed 4-bit Dolphin-X1-8B fine-tuning.

This entrypoint wraps :func:`monGARS.mlops.pipelines.unsloth.run_unsloth_finetune`
with sensible defaults for the ``dphn/Dolphin-X1-8B`` base model. It handles the
following responsibilities:

* Ensure Unsloth, bitsandbytes, llm2vec, and the Transformers stack are
  installed on-demand.
* Load the base model with 4-bit quantisation inside the configured VRAM
  envelope.
* Execute a LoRA fine-tuning run (Hugging Face dataset or local JSON/JSONL).
* Persist adapters, merged checkpoints, and an LLM2Vec-compatible wrapper.
* Emit a JSON summary so CI and operators can reason about the artefacts.

Example usage::

    python dolphin_x1_unsloth_pipeline.py \
        --output-dir outputs/dolphin-x1 \
        --dataset-id yahma/alpaca-cleaned \
        --train-fraction 0.1 \
        --epochs 1

The defaults mirror the Dolphin X1 release announcement: we keep the Llama 3.1
chat template, target 8K context, and assume a 7.5 GiB VRAM envelope which fits
on common consumer GPUs once activation buffers are considered.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from monGARS.mlops.pipelines.unsloth import run_unsloth_finetune

logger = logging.getLogger("dolphin_x1_unsloth")

DEFAULT_MODEL_ID = "dphn/Dolphin-X1-8B"
DEFAULT_DATASET_ID = "yahma/alpaca-cleaned"
DEFAULT_OUTPUT_DIR = Path("outputs_dolphin_x1")
DEFAULT_VRAM_MB = 7500
DEFAULT_ACTIVATION_BUFFER_MB = 1024
DEFAULT_MAX_SEQ_LEN = 8192


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {value!r}")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"Expected non-negative integer, got {value!r}"
        )
    return number


def _fraction(value: str) -> float:
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError(
            f"Train fraction must be within (0, 1], received {value!r}"
        )
    return fraction


def _epochs(value: str) -> float:
    epochs = float(value)
    if epochs <= 0:
        raise argparse.ArgumentTypeError(
            f"Number of epochs must be positive, received {value!r}"
        )
    return epochs


def _dropout(value: str) -> float:
    dropout = float(value)
    if not 0 <= dropout < 1:
        raise argparse.ArgumentTypeError(
            f"Dropout must be within [0, 1), received {value!r}"
        )
    return dropout


def _learning_rate(value: str) -> float:
    lr = float(value)
    if lr <= 0:
        raise argparse.ArgumentTypeError(
            f"Learning rate must be positive, got {value!r}"
        )
    return lr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Dolphin-X1-8B with Unsloth and emit LLM2Vec wrappers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Local JSON/JSONL dataset (overrides --dataset-id)",
    )
    parser.add_argument(
        "--max-seq-len", type=_positive_int, default=DEFAULT_MAX_SEQ_LEN
    )
    parser.add_argument("--vram-budget-mb", type=_positive_int, default=DEFAULT_VRAM_MB)
    parser.add_argument(
        "--activation-buffer-mb",
        type=_non_negative_int,
        default=DEFAULT_ACTIVATION_BUFFER_MB,
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--grad-accum", type=_positive_int, default=8)
    parser.add_argument("--learning-rate", type=_learning_rate, default=2e-4)
    parser.add_argument("--epochs", type=_epochs, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-rank", type=_positive_int, default=32)
    parser.add_argument("--lora-alpha", type=_positive_int, default=32)
    parser.add_argument("--lora-dropout", type=_dropout, default=0.0)
    parser.add_argument("--train-fraction", type=_fraction, default=1.0)
    parser.add_argument("--eval-dataset-id", default=None)
    parser.add_argument(
        "--eval-dataset-path",
        type=Path,
        default=None,
        help="Optional evaluation dataset for validation metrics",
    )
    parser.add_argument("--eval-batch-size", type=_positive_int, default=2)
    parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Disable generation/embedding smoke tests (useful for CI)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing run_metadata.json",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging LoRA adapters into an FP16 checkpoint",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Emit the run summary as JSON to stdout",
    )
    return parser


def _resolve_dataset_id(args: argparse.Namespace) -> str | None:
    if args.dataset_path is not None:
        return None
    if args.dataset_id:
        return str(args.dataset_id)
    return DEFAULT_DATASET_ID


def _normalise_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value.expanduser().resolve()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = _normalise_path(args.output_dir) or DEFAULT_OUTPUT_DIR.resolve()
    dataset_path = _normalise_path(args.dataset_path)
    eval_dataset_path = _normalise_path(args.eval_dataset_path)

    result = run_unsloth_finetune(
        model_id=str(args.model_id),
        output_dir=output_dir,
        dataset_id=_resolve_dataset_id(args),
        dataset_path=dataset_path,
        max_seq_len=int(args.max_seq_len),
        vram_budget_mb=int(args.vram_budget_mb),
        activation_buffer_mb=int(args.activation_buffer_mb),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        learning_rate=float(args.learning_rate),
        epochs=float(args.epochs),
        max_steps=int(args.max_steps),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        train_fraction=float(args.train_fraction),
        eval_dataset_id=(None if dataset_path else args.eval_dataset_id),
        eval_dataset_path=eval_dataset_path,
        eval_batch_size=int(args.eval_batch_size),
        run_smoke_tests=not args.skip_smoke_tests,
        write_metadata=not args.no_metadata,
        merge_to_fp16=not args.no_merge,
    )

    wrapper_dir = result.get("wrapper_dir")
    logger.info(
        "Pipeline finished",
        extra={
            "output_dir": str(result.get("output_dir")),
            "wrapper_dir": str(wrapper_dir) if wrapper_dir else None,
            "merged_dir": (
                str(result.get("merged_dir")) if result.get("merged_dir") else None
            ),
            "quantized_4bit": result.get("quantized_4bit"),
            "dataset_size": result.get("dataset_size"),
            "eval_dataset_size": result.get("eval_dataset_size"),
        },
    )

    if args.json_summary:
        serialisable = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in result.items()
        }
        print(json.dumps(serialisable, indent=2))
    else:
        print("Artifacts written to:", result.get("output_dir"))
        if wrapper_dir:
            print("LLM2Vec wrapper:", wrapper_dir)
        merged_dir = result.get("merged_dir")
        if merged_dir:
            print("Merged FP16 checkpoint:", merged_dir)

    return result


if __name__ == "__main__":
    main()
