#!/usr/bin/env python3
"""Command-line entry point for the monGARS LLM fine-tuning toolchain."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from modules.neurons.registry import update_manifest
    from monGARS.mlops.artifacts import build_adapter_summary
    from monGARS.mlops.code_analysis import (
        render_usage_report,
        scan_llm_usage,
        scan_module_interactions,
    )
    from monGARS.mlops.dataset import (
        build_module_interaction_dataset,
        build_mongars_strategy_dataset,
    )
    from monGARS.mlops.pipelines import run_unsloth_finetune
except ModuleNotFoundError:  # pragma: no cover - allow execution as a script
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from modules.neurons.registry import update_manifest
    from monGARS.mlops.artifacts import build_adapter_summary
    from monGARS.mlops.code_analysis import (
        render_usage_report,
        scan_llm_usage,
        scan_module_interactions,
    )
    from monGARS.mlops.dataset import (
        build_module_interaction_dataset,
        build_mongars_strategy_dataset,
    )
    from monGARS.mlops.pipelines import run_unsloth_finetune

LOGGER = logging.getLogger("monGARS.llm_pipeline")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_analyze(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    LOGGER.info("Scanning repository", extra={"root": str(root)})
    usages = scan_llm_usage(root)
    report = render_usage_report(usages)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    LOGGER.info(
        "Wrote analysis report",
        extra={"output": str(output), "callsite_count": len(usages)},
    )


def cmd_dataset(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    usages = scan_llm_usage(root)
    dataset_path = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else None
    build_mongars_strategy_dataset(
        usages,
        dataset_path,
        metadata_path=metadata_path,
        min_examples=args.min_examples,
    )
    LOGGER.info(
        "Generated custom dataset",
        extra={
            "dataset_path": str(dataset_path),
            "metadata_path": str(metadata_path) if metadata_path else None,
        },
    )


def cmd_module_dataset(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    interactions = scan_module_interactions(
        root, packages=tuple(args.packages), ignore_parts=tuple(args.ignore_parts)
    )
    dataset_path = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else None
    build_module_interaction_dataset(
        interactions,
        dataset_path,
        metadata_path=metadata_path,
        min_examples=args.min_examples,
    )
    LOGGER.info(
        "Generated module interaction dataset",
        extra={
            "dataset_path": str(dataset_path),
            "metadata_path": str(metadata_path) if metadata_path else None,
            "interactions": len(interactions),
        },
    )


def cmd_finetune(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else None
    eval_dataset_path = (
        Path(args.eval_dataset_path).resolve() if args.eval_dataset_path else None
    )
    if args.dataset_id is None and dataset_path is None:
        raise SystemExit("--dataset-id or --dataset-path must be supplied for finetune")
    results = run_unsloth_finetune(
        model_id=args.model_id,
        output_dir=output_dir,
        dataset_id=args.dataset_id,
        dataset_path=dataset_path,
        max_seq_len=args.max_seq_len,
        vram_budget_mb=args.vram_budget_mb,
        activation_buffer_mb=args.activation_buffer_mb,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_fraction=args.train_fraction,
        eval_dataset_id=args.eval_dataset_id,
        eval_dataset_path=eval_dataset_path,
        eval_batch_size=args.eval_batch_size,
        run_smoke_tests=not args.skip_smoke_tests,
        write_metadata=not args.skip_metadata,
        merge_to_fp16=not args.skip_merge,
    )
    LOGGER.info(
        "Fine-tuning completed",
        extra={key: str(value) for key, value in results.items()},
    )

    if args.registry_path:
        registry_path = Path(args.registry_path).resolve()
        metrics = {
            "dataset_size": results.get("dataset_size"),
            "eval_dataset_size": results.get("eval_dataset_size"),
        }
        eval_metrics = results.get("evaluation_metrics") or {}
        if eval_metrics:
            metrics["evaluation"] = eval_metrics
        summary = build_adapter_summary(
            adapter_dir=results["chat_lora_dir"],
            weights_path=None,
            wrapper_dir=results.get("wrapper_dir"),
            status="success",
            labels={"pipeline": "unsloth_llm2vec"},
            metrics={k: v for k, v in metrics.items() if v is not None},
            training={
                "model_id": args.model_id,
                "dataset_id": args.dataset_id,
                "dataset_path": str(dataset_path) if dataset_path else None,
                "eval_dataset_id": args.eval_dataset_id,
                "eval_dataset_path": (
                    str(eval_dataset_path) if eval_dataset_path else None
                ),
                "max_seq_len": args.max_seq_len,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "train_fraction": args.train_fraction,
                "eval_batch_size": args.eval_batch_size,
            },
        )
        artifacts = summary.setdefault("artifacts", {})
        merged_dir = results.get("merged_dir")
        if merged_dir:
            artifacts["merged_fp16"] = str(merged_dir)
        try:
            manifest = update_manifest(registry_path, summary)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to update adapter manifest", exc_info=True)
            raise SystemExit(f"Adapter manifest update failed: {exc}")
        LOGGER.info(
            "Adapter manifest refreshed",
            extra={"manifest_path": str(manifest.path)},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(command=None)
    parser.add_argument(
        "--root",
        default=_default_repo_root(),
        help="Repository root used for analysis commands (default: repo root)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Generate an LLM usage report")
    analyze.add_argument(
        "--output",
        required=True,
        help="Destination Markdown file for the analysis report",
    )
    analyze.set_defaults(func=cmd_analyze)

    dataset = subparsers.add_parser(
        "dataset", help="Build a custom dataset from detected callsites"
    )
    dataset.add_argument(
        "--output",
        required=True,
        help="Destination JSONL dataset path",
    )
    dataset.add_argument(
        "--metadata",
        help="Optional metadata JSON path to describe the dataset",
    )
    dataset.add_argument(
        "--min-examples",
        type=int,
        default=4,
        help="Minimum number of callsites required to create the dataset",
    )
    dataset.set_defaults(func=cmd_dataset)

    module_dataset = subparsers.add_parser(
        "module-dataset",
        help="Build a dataset describing module-to-module dependencies",
    )
    module_dataset.add_argument(
        "--output",
        required=True,
        help="Destination JSONL dataset path",
    )
    module_dataset.add_argument(
        "--metadata",
        help="Optional metadata JSON path to describe the dataset",
    )
    module_dataset.add_argument(
        "--min-examples",
        type=int,
        default=8,
        help="Minimum number of interactions required to create the dataset",
    )
    module_dataset.add_argument(
        "--packages",
        nargs="+",
        default=["monGARS", "modules"],
        help="Root packages considered when scanning for interactions",
    )
    module_dataset.add_argument(
        "--ignore-parts",
        nargs="+",
        default=[".venv", "tests", "build", "dist", "__pycache__"],
        help="Directory names to ignore while scanning",
    )
    module_dataset.set_defaults(func=cmd_module_dataset)

    finetune = subparsers.add_parser(
        "finetune", help="Execute the Unsloth fine-tuning workflow"
    )
    finetune.add_argument(
        "--model-id",
        required=True,
        help="Base model identifier (e.g. dphn/Dolphin3.0-Llama3.1-8B)",
    )
    finetune.add_argument(
        "--dataset-id",
        help="Hugging Face dataset identifier used for training",
    )
    finetune.add_argument(
        "--dataset-path",
        help="Local JSONL dataset file produced by the dataset command",
    )
    finetune.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to store LoRA adapters and wrapper artefacts",
    )
    finetune.add_argument("--max-seq-len", type=int, default=1024)
    finetune.add_argument("--vram-budget-mb", type=int, default=8192)
    finetune.add_argument("--activation-buffer-mb", type=int, default=1024)
    finetune.add_argument("--batch-size", type=int, default=1)
    finetune.add_argument("--grad-accum", type=int, default=8)
    finetune.add_argument("--learning-rate", type=float, default=2e-4)
    finetune.add_argument("--epochs", type=float, default=1.0)
    finetune.add_argument("--max-steps", type=int, default=-1)
    finetune.add_argument("--lora-rank", type=int, default=32)
    finetune.add_argument("--lora-alpha", type=int, default=32)
    finetune.add_argument("--lora-dropout", type=float, default=0.0)
    finetune.add_argument("--train-fraction", type=float)
    finetune.add_argument("--eval-dataset-id")
    finetune.add_argument("--eval-dataset-path")
    finetune.add_argument("--eval-batch-size", type=int)
    finetune.add_argument(
        "--registry-path",
        help="Optional adapter registry directory for manifest updates",
    )
    finetune.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Skip the post-training embedding smoke test",
    )
    finetune.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not write run_metadata.json",
    )
    finetune.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging LoRA adapters into an FP16 checkpoint",
    )
    finetune.set_defaults(func=cmd_finetune)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
