#!/usr/bin/env python3
"""Generate datasets and run Unsloth + LLM2Vec fine-tuning in one step."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from monGARS.mlops.code_analysis import (
        scan_llm_usage,
        scan_module_interactions,
    )
    from monGARS.mlops.dataset import build_unsloth_llm2vec_dataset
    from monGARS.mlops.pipelines import run_unsloth_finetune
except ModuleNotFoundError as exc:  # pragma: no cover - CLI invocation convenience
    raise ModuleNotFoundError(
        "Unable to import monGARS modules. Run this script from the repository root "
        "(e.g. `python -m scripts.auto_unsloth_llm2vec`) or set PYTHONPATH to include "
        f"{REPO_ROOT}"
    ) from exc


LOGGER = logging.getLogger("auto_unsloth_llm2vec")
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "unsloth_auto"
DEFAULT_METADATA = DEFAULT_DATASET_DIR / "metadata.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "unsloth_llm2vec"
DEFAULT_SEED_DATASET = (
    REPO_ROOT / "datasets" / "unsloth" / "monGARS_unsloth_dataset.jsonl"
)
DATASET_PREP_METADATA = "dataset_metadata.json"
DATASET_PREP_CANDIDATES = (
    "unsloth_prompt_completion.jsonl",
    "combined_instruct.unsloth.jsonl",
    "combined_instruct.jsonl",
)


def _default_repo_root() -> Path:
    return REPO_ROOT


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _normalise_paths(paths: Sequence[str]) -> list[Path]:
    return [Path(path).expanduser().resolve() for path in paths]


def _load_dataset_prep_metadata(metadata_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOGGER.warning("Dataset prep metadata %s does not exist", metadata_path)
        return {}
    except json.JSONDecodeError as exc:
        LOGGER.warning("Invalid dataset prep metadata at %s: %s", metadata_path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _discover_dataset_prep_exports(paths: Sequence[Path]) -> list[Path]:
    """Return prompt/completion datasets derived from dataset_prep outputs."""

    discovered: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        base = Path(raw_path).expanduser().resolve()
        candidates: list[Path] = []
        search_roots = [base.parent if base.is_file() else base]

        metadata_files: list[Path] = []
        if base.is_file() and base.suffix == ".json":
            metadata_files.append(base)
        elif base.is_dir():
            metadata_files.append(base / DATASET_PREP_METADATA)

        for metadata_path in metadata_files:
            if not metadata_path.exists():
                continue
            payload = _load_dataset_prep_metadata(metadata_path)
            derived = payload.get("derived_outputs") if payload else {}
            if isinstance(derived, dict):
                for key in ("unsloth_prompt_completion", "instruction_jsonl"):
                    entry = derived.get(key)
                    candidate: Path | None = None
                    if isinstance(entry, str):
                        candidate = Path(entry)
                    elif isinstance(entry, dict):
                        location = entry.get("path")
                        if isinstance(location, str):
                            candidate = Path(location)
                    if candidate is None:
                        continue
                    if not candidate.is_absolute():
                        candidate = (metadata_path.parent / candidate).resolve()
                    else:
                        candidate = candidate.resolve()
                    candidates.append(candidate)

        for root in search_roots:
            for name in DATASET_PREP_CANDIDATES:
                candidate = (root / name).resolve()
                candidates.append(candidate)

        for candidate in candidates:
            if candidate in seen:
                continue
            if not candidate.exists():
                continue
            seen.add(candidate)
            discovered.append(candidate)

    if discovered:
        LOGGER.info(
            "Discovered %d dataset_prep exports",
            len(discovered),
            extra={"paths": [str(path) for path in discovered]},
        )
    return discovered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined monGARS dataset and launch the Unsloth + LLM2Vec pipeline."
        )
    )

    dataset_args: list[tuple[tuple[str, ...], dict[str, object]]] = [
        (("--root",), {"default": _default_repo_root(), "help": "Repository root"}),
        (
            ("--dataset-dir",),
            {
                "default": DEFAULT_DATASET_DIR,
                "help": "Directory where the generated train/validation splits are written.",
            },
        ),
        (
            ("--metadata-path",),
            {
                "default": DEFAULT_METADATA,
                "help": "Optional metadata file describing the dataset composition.",
            },
        ),
        (
            ("--validation-ratio",),
            {
                "type": float,
                "default": 0.1,
                "help": "Fraction of records reserved for validation (must be between 0 and 1).",
            },
        ),
        (
            ("--shuffle-seed",),
            {
                "type": int,
                "default": 13,
                "help": "Deterministic seed used when shuffling merged dataset records.",
            },
        ),
        (
            ("--seed-dataset",),
            {
                "action": "append",
                "default": [],
                "help": "Additional JSONL datasets (prompt/completion) merged into the corpus.",
            },
        ),
        (
            ("--dataset-prep-output",),
            {
                "action": "append",
                "default": [],
                "help": (
                    "Directories or metadata files produced by scripts/dataset_prep.py. "
                    "Derived prompt/completion exports are included automatically."
                ),
            },
        ),
        (
            ("--packages",),
            {
                "nargs": "*",
                "default": ("monGARS", "modules"),
                "help": "Package prefixes considered when scanning module interactions.",
            },
        ),
        (
            ("--ignore-part",),
            {
                "action": "append",
                "default": [".venv", "tests", "build", "dist", "__pycache__"],
                "help": "Directory names ignored during source analysis (may be repeated).",
            },
        ),
        (
            ("--model-id",),
            {
                "default": "dphn/Dolphin-X1-8B",
                "help": "Base model identifier passed to run_unsloth_finetune.",
            },
        ),
        (
            ("--output-dir",),
            {
                "default": DEFAULT_OUTPUT_DIR,
                "help": "Where fine-tuning artefacts (adapters, wrappers) will be written.",
            },
        ),
    ]

    numeric_args = [
        (("--vram-budget-mb",), {"type": int, "default": 7500}),
        (("--activation-buffer-mb",), {"type": int, "default": 1024}),
        (("--max-seq-len",), {"type": int, "default": 4096}),
        (("--batch-size",), {"type": int, "default": 1}),
        (("--grad-accum",), {"type": int, "default": 8}),
        (("--learning-rate",), {"type": float, "default": 2e-4}),
        (("--epochs",), {"type": float, "default": 1.0}),
        (("--max-steps",), {"type": int, "default": -1}),
        (("--lora-rank",), {"type": int, "default": 32}),
        (("--lora-alpha",), {"type": int, "default": 32}),
        (("--lora-dropout",), {"type": float, "default": 0.0}),
        (
            ("--eval-batch-size",),
            {
                "type": int,
                "default": 2,
                "help": "Batch size forwarded to Trainer.evaluate when validation data exists.",
            },
        ),
    ]

    boolean_flags = [
        (
            "--no-default-seed",
            {
                "action": "store_true",
                "help": "Do not automatically include datasets/unsloth/monGARS_unsloth_dataset.jsonl.",
            },
        ),
        (
            "--skip-train",
            {
                "action": "store_true",
                "help": "Generate datasets only; skip the Unsloth fine-tuning phase.",
            },
        ),
        (
            "--skip-smoke-tests",
            {
                "action": "store_true",
                "help": "Disable generation/embedding smoke tests inside run_unsloth_finetune.",
            },
        ),
        (
            "--skip-metadata",
            {
                "action": "store_true",
                "help": "Do not write run_metadata.json next to the training artefacts.",
            },
        ),
        (
            "--skip-merge",
            {
                "action": "store_true",
                "help": "Avoid merging LoRA adapters into an FP16 checkpoint after training.",
            },
        ),
        (
            "--verbose",
            {
                "action": "store_true",
                "help": "Increase logging verbosity for troubleshooting.",
            },
        ),
    ]

    for flags, options in [*dataset_args, *numeric_args]:
        parser.add_argument(*flags, **options)

    for flag, options in boolean_flags:
        parser.add_argument(flag, **options)

    return parser


def _resolve_seed_datasets(args: argparse.Namespace) -> list[Path]:
    seeds = _normalise_paths(args.seed_dataset)
    dataset_prep_roots = _normalise_paths(getattr(args, "dataset_prep_output", []))
    seeds.extend(_discover_dataset_prep_exports(dataset_prep_roots))
    if not args.no_default_seed and DEFAULT_SEED_DATASET.exists():
        seeds.append(DEFAULT_SEED_DATASET.resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in seeds:
        if not path.exists():
            LOGGER.warning("Seed dataset %s does not exist; skipping", path)
            continue
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be within (0, 1)")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise SystemExit("batch-size and grad-accum must be positive integers")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise SystemExit("LoRA rank/alpha must be positive")
    return args


def _build_dataset(args: argparse.Namespace) -> dict[str, Path | None]:
    root = Path(args.root).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    metadata_path = Path(args.metadata_path).resolve() if args.metadata_path else None
    ignore_parts = tuple(args.ignore_part)
    packages = tuple(args.packages)

    LOGGER.info("Scanning repository for LLM callsites", extra={"root": str(root)})
    usages = scan_llm_usage(root, ignore_parts=ignore_parts)
    LOGGER.info("Discovered %d callsites", len(usages))

    interactions = scan_module_interactions(
        root, packages=packages, ignore_parts=ignore_parts
    )
    LOGGER.info("Discovered %d module interactions", len(interactions))

    seeds = _resolve_seed_datasets(args)
    dataset_paths = build_unsloth_llm2vec_dataset(
        usages,
        interactions,
        dataset_dir,
        validation_ratio=float(args.validation_ratio),
        shuffle_seed=int(args.shuffle_seed),
        metadata_path=metadata_path,
        extra_datasets=seeds,
    )
    LOGGER.info(
        "Dataset prepared",
        extra={
            "train": str(dataset_paths["train"]),
            "validation": str(dataset_paths.get("validation")),
        },
    )
    return dataset_paths


def _run_training(
    args: argparse.Namespace, dataset_paths: dict[str, Path | None]
) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting Unsloth fine-tuning", extra={"output": str(output_dir)})
    results = run_unsloth_finetune(
        model_id=str(args.model_id),
        output_dir=output_dir,
        dataset_id=None,
        dataset_path=dataset_paths["train"],
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
        train_fraction=None,
        eval_dataset_id=None,
        eval_dataset_path=dataset_paths.get("validation"),
        eval_batch_size=int(args.eval_batch_size),
        run_smoke_tests=not args.skip_smoke_tests,
        write_metadata=not args.skip_metadata,
        merge_to_fp16=not args.skip_merge,
    )
    LOGGER.info(
        "Fine-tuning completed",
        extra={
            "chat_lora": str(results.get("chat_lora_dir")),
            "wrapper": str(results.get("wrapper_dir")),
        },
    )
    return dict(results)


def main(argv: Sequence[str] | None = None) -> dict[str, object | Path | None]:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    dataset_paths = _build_dataset(args)
    if args.skip_train:
        LOGGER.info("Skip-train flag supplied; training phase disabled")
        return dataset_paths
    return _run_training(args, dataset_paths)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
