#!/usr/bin/env python3
"""Automation workflow for Dolphin fine-tuning with Unsloth and LLM2Vec export.

This utility orchestrates the full data preparation and training pipeline used
by monGARS when producing a Dolphin SFT adapter.  It performs four high-level
steps:

1. Execute :mod:`scripts.ultimate_repo_analyzer` (when requested) to refresh the
   structured dataset generated from the repository's source tree.
2. Merge that dataset with ``datasets/formatted_dataset 2.jsonl`` while
   normalising records, removing duplicates, and creating train/validation
   splits on disk.
3. Launch :mod:`scripts.train_dolphin_unsloth` with 4-bit quantisation enabled
   via Unsloth so LoRA fine-tuning runs automatically.
4. Trigger the LLM2Vec conversion hook exposed by the training script so a
   ready-to-serve embedding wrapper is bundled next to the chat checkpoint.

The workflow embraces the repository conventions for logging, optional
dependencies, and reproducibility.  It intentionally keeps the top-level API
small so CI jobs or operators can run it non-interactively.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger("dolphin_unsloth_workflow")

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_SCRIPT = REPO_ROOT / "scripts" / "ultimate_repo_analyzer.py"
ANALYZER_OUTPUT = (
    REPO_ROOT / "scripts" / "data" / "ultimate" / "processed_repo" / "sft_repo.jsonl"
)
FORMATTED_DATASET = REPO_ROOT / "datasets" / "formatted_dataset 2.jsonl"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "dolphin_unsloth_auto"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dolphin3_unsloth_adapter"


@dataclass(slots=True)
class WorkflowConfig:
    """Typed configuration derived from CLI arguments."""

    refresh_analysis: bool
    skip_analysis: bool
    analyzer_script: Path
    analyzer_output: Path
    formatted_dataset: Path
    dataset_output_dir: Path
    validation_ratio: float
    shuffle_seed: int
    training_output_dir: Path
    max_seq_length: int
    learning_rate: float
    num_train_epochs: float
    gradient_accumulation_steps: int
    hf_token: str | None
    hf_token_source: str | None
    allow_cpu_fallback: bool
    max_retries: int
    minimum_train_records: int
    dry_run: bool


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_repo_analysis(config: WorkflowConfig) -> None:
    """Invoke the repository analyser to refresh the SFT dataset."""

    if config.analyzer_output.exists() and not config.refresh_analysis:
        LOGGER.info(
            "Skipping repo analysis; dataset already present at %s",
            config.analyzer_output,
        )
        return

    if not config.analyzer_script.exists():
        raise FileNotFoundError(
            f"Repository analysis script not found: {config.analyzer_script}"
        )

    LOGGER.info("Running repository analyser via %s", config.analyzer_script)
    env = os.environ.copy()
    env.setdefault("CONFIRM_SCAN", "YES")
    try:
        subprocess.run(
            [sys.executable, str(config.analyzer_script)],
            check=True,
            cwd=str(config.analyzer_script.parent.parent),
            env=env,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - subprocess
        raise RuntimeError("Repository analysis failed") from exc

    if not config.analyzer_output.exists():
        raise FileNotFoundError(
            "Repository analysis did not generate the expected dataset at "
            f"{config.analyzer_output}"
        )


def _normalise_record(record: dict[str, object]) -> dict[str, object]:
    """Return a canonical representation suitable for deduplication."""

    normalised = dict(record)
    instruction = str(normalised.get("instruction", ""))
    input_text = str(normalised.get("input", ""))
    output = str(normalised.get("output", ""))

    normalised["instruction"] = instruction.strip()
    normalised["input"] = input_text.strip()
    normalised["output"] = output.strip()

    if "system" in normalised and normalised["system"] is not None:
        normalised["system"] = str(normalised["system"]).strip()

    return normalised


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} of {path}"
                )
            record.setdefault("input", "")
            records.append(_normalise_record(record))
    LOGGER.info("Loaded %d rows from %s", len(records), path)
    return records


def _write_jsonl_records(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %s", path)


def build_datasets(config: WorkflowConfig) -> tuple[Path, Path | None]:
    """Merge raw datasets and create training/validation splits."""

    repo_records = _load_jsonl_records(config.analyzer_output)
    formatted_records = _load_jsonl_records(config.formatted_dataset)

    combined = repo_records + formatted_records
    if not combined:
        raise RuntimeError("Combined dataset is empty; nothing to train on")

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, object]] = []
    skipped_duplicates = 0
    for record in combined:
        key = (
            str(record.get("instruction", "")),
            str(record.get("input", "")),
            str(record.get("output", "")),
        )
        if key in seen:
            skipped_duplicates += 1
            continue
        seen.add(key)
        deduped.append(record)

    LOGGER.info(
        "Dataset contains %d unique records after deduplication",
        len(deduped),
    )
    if skipped_duplicates:
        LOGGER.info("Skipped %d duplicate records", skipped_duplicates)

    rng = random.Random(config.shuffle_seed)
    rng.shuffle(deduped)

    validation_count = int(len(deduped) * config.validation_ratio)
    if validation_count > 0:
        validation_records = deduped[:validation_count]
        train_records = deduped[validation_count:]
    else:
        validation_records = []
        train_records = deduped

    if len(train_records) < config.minimum_train_records:
        raise RuntimeError(
            "Insufficient training records after splitting. "
            f"Required at least {config.minimum_train_records}, got {len(train_records)}."
        )

    if not train_records:
        raise RuntimeError("Training split is empty after applying validation ratio")

    dataset_dir = config.dataset_output_dir
    if dataset_dir.exists() and dataset_dir.is_file():
        raise RuntimeError(
            f"Dataset output path {dataset_dir} is a file; expected a directory"
        )
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"

    _write_jsonl_records(train_path, train_records)
    if validation_records:
        _write_jsonl_records(validation_path, validation_records)
        validation_file: Path | None = validation_path
    else:
        if validation_path.exists():
            validation_path.unlink()
        validation_file = None

    LOGGER.info(
        "Prepared %d training and %d validation records",
        len(train_records),
        len(validation_records),
    )

    return train_path, validation_file


def run_training(
    config: WorkflowConfig,
    train_file: Path,
    validation_file: Path | None,
) -> None:
    """Invoke the Dolphin Unsloth trainer with the prepared datasets."""

    if config.dry_run:
        LOGGER.info("Dry-run enabled; skipping training invocation")
        return

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        from scripts import train_dolphin_unsloth
    except RuntimeError as exc:  # pragma: no cover - dependency guard
        LOGGER.error("Failed to import Unsloth trainer: %s", exc)
        raise
    except Exception as exc:  # pragma: no cover - unexpected import failure
        LOGGER.exception("Unexpected error while importing train_dolphin_unsloth")
        raise RuntimeError("Unable to import train_dolphin_unsloth") from exc

    if not hasattr(train_dolphin_unsloth, "main"):
        raise AttributeError(
            "train_dolphin_unsloth module does not expose a main() entrypoint"
        )

    training_args: list[str] = [
        "--train-file",
        str(train_file),
        "--output-dir",
        str(config.training_output_dir),
        "--max-seq-length",
        str(config.max_seq_length),
        "--num-train-epochs",
        str(config.num_train_epochs),
        "--learning-rate",
        str(config.learning_rate),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--max-retries",
        str(config.max_retries),
        "--convert-to-llm2vec",
    ]

    if config.allow_cpu_fallback:
        training_args.append("--allow-cpu-fallback")

    if validation_file is not None:
        training_args.extend(["--validation-file", str(validation_file)])

    if config.hf_token:
        training_args.extend(["--hf-token", config.hf_token])

    LOGGER.info("Starting Dolphin fine-tuning via train_dolphin_unsloth")
    LOGGER.debug("Trainer arguments: %s", training_args)
    try:
        train_dolphin_unsloth.main(training_args)
    except SystemExit as exc:  # pragma: no cover - propagate CLI exits with context
        raise RuntimeError("Trainer aborted early") from exc
    except Exception as exc:  # pragma: no cover - guard unexpected failures
        LOGGER.exception("Trainer raised an unexpected error")
        raise


def parse_arguments(argv: Sequence[str] | None = None) -> WorkflowConfig:
    parser = argparse.ArgumentParser(
        description="Automate Dolphin fine-tuning with Unsloth and LLM2Vec export",
    )
    parser.add_argument(
        "--refresh-analysis",
        action="store_true",
        help="Re-run the repository analyser even when cached data is available.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip repository analysis and reuse existing outputs as-is.",
    )
    parser.add_argument(
        "--analyzer-script",
        type=Path,
        default=ANALYZER_SCRIPT,
        help="Path to the repository analysis script that generates SFT data.",
    )
    parser.add_argument(
        "--analyzer-output",
        type=Path,
        default=ANALYZER_OUTPUT,
        help="Expected JSONL output file produced by the analysis step.",
    )
    parser.add_argument(
        "--formatted-dataset",
        type=Path,
        default=FORMATTED_DATASET,
        help="Additional JSONL dataset merged into the training corpus.",
    )
    parser.add_argument(
        "--dataset-output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory where merged train/validation splits will be written.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples reserved for validation (0 disables validation).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed used when shuffling the merged dataset before splitting.",
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the fine-tuned checkpoint and wrappers.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length forwarded to the trainer.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate used by the trainer.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of epochs passed to the trainer.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation factor used during training.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Optional Hugging Face token forwarded to the trainer.",
    )
    parser.add_argument(
        "--minimum-train-records",
        type=int,
        default=25,
        help="Fail if the training split contains fewer records than this threshold.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Forward --allow-cpu-fallback to the trainer for non-GPU systems.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of CUDA OOM retries attempted by the trainer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the actual training step after datasets are prepared.",
    )

    parsed = parser.parse_args(argv)

    if parsed.validation_ratio < 0 or parsed.validation_ratio >= 1:
        raise SystemExit(
            "validation-ratio must be between 0 (inclusive) and 1 (exclusive)"
        )

    if parsed.refresh_analysis and parsed.skip_analysis:
        raise SystemExit(
            "--refresh-analysis and --skip-analysis are mutually exclusive"
        )

    if parsed.minimum_train_records < 1:
        raise SystemExit("--minimum-train-records must be at least 1")

    refresh_analysis = bool(parsed.refresh_analysis and not parsed.skip_analysis)
    analyzer_output = Path(parsed.analyzer_output)

    hf_token = parsed.hf_token
    hf_token_source: str | None = None
    if hf_token:
        hf_token_source = "cli"
    else:
        for env_name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
            env_value = os.environ.get(env_name)
            if env_value:
                hf_token = env_value
                hf_token_source = f"env:{env_name}"
                break

    return WorkflowConfig(
        refresh_analysis=refresh_analysis,
        skip_analysis=bool(parsed.skip_analysis),
        analyzer_script=Path(parsed.analyzer_script),
        analyzer_output=analyzer_output,
        formatted_dataset=Path(parsed.formatted_dataset),
        dataset_output_dir=Path(parsed.dataset_output_dir),
        validation_ratio=float(parsed.validation_ratio),
        shuffle_seed=int(parsed.shuffle_seed),
        training_output_dir=Path(parsed.training_output_dir),
        max_seq_length=int(parsed.max_seq_length),
        learning_rate=float(parsed.learning_rate),
        num_train_epochs=float(parsed.num_train_epochs),
        gradient_accumulation_steps=int(parsed.gradient_accumulation_steps),
        hf_token=hf_token,
        hf_token_source=hf_token_source,
        allow_cpu_fallback=bool(parsed.allow_cpu_fallback),
        max_retries=int(parsed.max_retries),
        minimum_train_records=int(parsed.minimum_train_records),
        dry_run=bool(parsed.dry_run),
    )


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    config = parse_arguments(argv)

    if not Path(config.formatted_dataset).exists():
        raise FileNotFoundError(
            f"Formatted dataset not found: {config.formatted_dataset}"
        )

    if config.skip_analysis and not config.analyzer_output.exists():
        raise FileNotFoundError(
            "--skip-analysis requested but no analyser output is available at "
            f"{config.analyzer_output}"
        )

    if config.training_output_dir.exists() and not config.training_output_dir.is_dir():
        raise RuntimeError(
            f"Training output path {config.training_output_dir} is not a directory"
        )

    if not config.dry_run and config.training_output_dir.exists():
        LOGGER.info(
            "Training output directory %s already exists; results will be overwritten",
            config.training_output_dir,
        )

    if config.hf_token_source:
        if config.hf_token_source == "cli":
            LOGGER.info("Hugging Face token provided via CLI flag")
        else:
            env_name = config.hf_token_source.split(":", 1)[1]
            LOGGER.info(
                "Hugging Face token sourced from environment variable %s", env_name
            )

    if not config.dry_run:
        config.training_output_dir.mkdir(parents=True, exist_ok=True)

    if not config.dry_run and not config.skip_analysis:
        run_repo_analysis(config)
    elif config.skip_analysis:
        LOGGER.info("Repository analysis skipped via CLI flag")

    train_file, validation_file = build_datasets(config)
    run_training(config, train_file, validation_file)


if __name__ == "__main__":
    main()
