#!/usr/bin/env python3
"""One-shot pipeline to train Dolphin-X1-8B on the monGARS dataset and emit wrappers."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - allow invocation as a module or script
    from modules.neurons.registry import update_manifest
    from monGARS.mlops.artifacts import build_adapter_summary
    from monGARS.mlops.pipelines import run_unsloth_finetune
except ModuleNotFoundError:  # pragma: no cover - defensive for direct execution
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from modules.neurons.registry import update_manifest
    from monGARS.mlops.artifacts import build_adapter_summary
    from monGARS.mlops.pipelines import run_unsloth_finetune

LOGGER = logging.getLogger("monGARS.auto_llm_pipeline")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "dphn/Dolphin-X1-8B"
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "datasets" / "unsloth" / "monGARS_unsloth_dataset.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "monGARS_dolphin_x1"
DEFAULT_REGISTRY_PATH = REPO_ROOT / "models" / "encoders"


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


def _dropout(value: str) -> float:
    dropout = float(value)
    if not 0 <= dropout < 1:
        raise argparse.ArgumentTypeError(
            f"Dropout must be within [0, 1), received {value!r}"
        )
    return dropout


def _epochs(value: str) -> float:
    epochs = float(value)
    if epochs <= 0:
        raise argparse.ArgumentTypeError(
            f"Number of epochs must be positive, received {value!r}"
        )
    return epochs


def _learning_rate(value: str) -> float:
    lr = float(value)
    if lr <= 0:
        raise argparse.ArgumentTypeError(
            f"Learning rate must be positive, got {value!r}"
        )
    return lr


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Dolphin-X1-8B on the curated monGARS dataset, export LoRA adapters, "
            "and emit an LLM2Vec-compatible wrapper in a single command."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Optional HuggingFace dataset identifier to load instead of a local JSONL file",
    )
    parser.add_argument("--max-seq-len", type=_positive_int, default=8192)
    parser.add_argument("--vram-budget-mb", type=_positive_int, default=7500)
    parser.add_argument("--activation-buffer-mb", type=_non_negative_int, default=1024)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--grad-accum", type=_positive_int, default=8)
    parser.add_argument("--learning-rate", type=_learning_rate, default=2e-4)
    parser.add_argument("--epochs", type=_epochs, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-rank", type=_positive_int, default=32)
    parser.add_argument("--lora-alpha", type=_positive_int, default=32)
    parser.add_argument("--lora-dropout", type=_dropout, default=0.0)
    parser.add_argument("--train-fraction", type=_fraction, default=1.0)
    parser.add_argument(
        "--eval-dataset-id",
        default=None,
        help="Optional evaluation dataset identifier overriding the training dataset",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=Path,
        default=None,
        help="Optional local evaluation dataset JSONL path",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=_positive_int,
        default=None,
        help="Override per-device evaluation batch size",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help=(
            "Optional adapter registry to update after training. Defaults to"
            " models/encoders relative to the repository root when omitted."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Disable generation and embedding smoke tests after training",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip writing run_metadata.json to the output directory",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Do not merge LoRA adapters into an FP16 checkpoint",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit training results as JSON"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run label stored in adapter manifests and metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Summarise dataset characteristics and exit without training",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON or YAML file providing default values for CLI options",
    )
    return parser


def _normalise_registry_path(path: Path | None) -> Path | None:
    if path is not None:
        return _resolve_path(path)
    default_path = DEFAULT_REGISTRY_PATH
    return default_path.expanduser().resolve()


def _load_config_payload(config_path: Path) -> dict[str, Any]:
    resolved = _resolve_path(config_path)
    if resolved is None or not resolved.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"Config file {resolved} is empty")

    if resolved.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "PyYAML is required to load YAML config files. Install it or use JSON."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise SystemExit(
            f"Config file {resolved} must define a JSON/YAML object with option overrides"
        )
    return data


def _apply_config_overrides(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[argparse.Namespace, dict[str, Any], list[str]]:
    config_path: Path | None = getattr(args, "config", None)
    if not config_path:
        return args, {}, []

    payload = _load_config_payload(config_path)
    defaults: dict[str, Any] = {
        action.dest: action.default for action in parser._actions if action.dest
    }
    applied: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in payload.items():
        if key not in defaults:
            LOGGER.warning("Ignoring unknown config key", extra={"key": key})
            continue
        current = getattr(args, key, None)
        if current != defaults[key]:
            skipped.append(key)
            continue
        setattr(args, key, value)
        applied[key] = value
    return args, applied, skipped


def _round_metric(value: float) -> float:
    return float(round(value, 4))


def _build_length_summary(values: Iterable[int]) -> dict[str, float | int]:
    series = list(values)
    if not series:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "min": int(min(series)),
        "max": int(max(series)),
        "mean": _round_metric(statistics.fmean(series)),
        "median": _round_metric(statistics.median(series)),
    }


def _summarise_local_dataset(dataset_path: Path) -> dict[str, Any]:
    prompts: list[str] = []
    completions: list[str] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise SystemExit(
                    f"Invalid JSON on line {idx} of {dataset_path}: {exc}"
                ) from exc
            prompt = str(record.get("prompt", ""))
            completion = str(record.get("completion", ""))
            if not prompt:
                raise SystemExit(
                    f"Dataset entry {idx} missing 'prompt' in {dataset_path}"
                )
            if not completion:
                raise SystemExit(
                    f"Dataset entry {idx} missing 'completion' in {dataset_path}"
                )
            prompts.append(prompt)
            completions.append(completion)

    if not prompts:
        raise SystemExit(f"Dataset at {dataset_path} is empty")

    prompt_lengths = [len(item) for item in prompts]
    completion_lengths = [len(item) for item in completions]
    prompt_tokens = [len(item.split()) for item in prompts]
    completion_tokens = [len(item.split()) for item in completions]
    unique_prompts = len({item.strip() for item in prompts})
    duplicates = len(prompts) - unique_prompts
    duplicate_ratio = duplicates / len(prompts)

    preview_count = min(3, len(prompts))
    preview_examples = [
        {
            "prompt": prompts[idx][:256],
            "completion": completions[idx][:256],
        }
        for idx in range(preview_count)
    ]

    return {
        "path": str(dataset_path),
        "examples": len(prompts),
        "unique_prompts": unique_prompts,
        "duplicate_examples": duplicates,
        "duplicate_ratio": _round_metric(duplicate_ratio),
        "prompt": {
            "chars": _build_length_summary(prompt_lengths),
            "tokens": _build_length_summary(prompt_tokens),
        },
        "completion": {
            "chars": _build_length_summary(completion_lengths),
            "tokens": _build_length_summary(completion_tokens),
        },
        "preview": preview_examples,
    }


@dataclass(slots=True)
class BundleExecution:
    dataset_id: str | None
    dataset_path: Path | None
    eval_dataset_id: str | None
    eval_dataset_path: Path | None
    output_dir: Path
    registry_path: Path | None
    max_seq_len: int
    vram_budget_mb: int
    activation_buffer_mb: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    epochs: float
    max_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    train_fraction: float
    eval_batch_size: int | None
    skip_smoke_tests: bool
    skip_metadata: bool
    skip_merge: bool
    emit_json: bool
    run_name: str | None
    dry_run: bool
    log_level: str


def _coerce_optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(Path(value))


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _build_execution(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> BundleExecution:
    args, applied_overrides, skipped_overrides = _apply_config_overrides(parser, args)

    # configure logging after merges to honour configured log level
    _configure_logging(getattr(args, "log_level", "INFO"))
    if applied_overrides:
        LOGGER.info(
            "Applied config overrides", extra={"keys": sorted(applied_overrides)}
        )
    if skipped_overrides:
        LOGGER.info(
            "Skipped config overrides in favour of CLI arguments",
            extra={"keys": sorted(skipped_overrides)},
        )

    dataset_id = getattr(args, "dataset_id", None)
    dataset_path = _coerce_optional_path(getattr(args, "dataset_path", None))
    eval_dataset_id = getattr(args, "eval_dataset_id", None)
    eval_dataset_path = _coerce_optional_path(getattr(args, "eval_dataset_path", None))
    output_dir = _resolve_path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR))
    registry_path = _normalise_registry_path(getattr(args, "registry_path", None))

    if dataset_path is None and not dataset_id:
        raise SystemExit("One of --dataset-path or --dataset-id must be provided")
    if dataset_path is not None and not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    exec_config = BundleExecution(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        eval_dataset_id=eval_dataset_id,
        eval_dataset_path=eval_dataset_path,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR.resolve(),
        registry_path=registry_path,
        max_seq_len=int(getattr(args, "max_seq_len")),
        vram_budget_mb=int(getattr(args, "vram_budget_mb")),
        activation_buffer_mb=int(getattr(args, "activation_buffer_mb")),
        batch_size=int(getattr(args, "batch_size")),
        grad_accum=int(getattr(args, "grad_accum")),
        learning_rate=float(getattr(args, "learning_rate")),
        epochs=float(getattr(args, "epochs")),
        max_steps=int(getattr(args, "max_steps")),
        lora_rank=int(getattr(args, "lora_rank")),
        lora_alpha=int(getattr(args, "lora_alpha")),
        lora_dropout=float(getattr(args, "lora_dropout")),
        train_fraction=float(getattr(args, "train_fraction")),
        eval_batch_size=_coerce_optional_int(getattr(args, "eval_batch_size", None)),
        skip_smoke_tests=bool(getattr(args, "skip_smoke_tests")),
        skip_metadata=bool(getattr(args, "skip_metadata")),
        skip_merge=bool(getattr(args, "skip_merge")),
        emit_json=bool(getattr(args, "json")),
        run_name=getattr(args, "run_name", None) or None,
        dry_run=bool(getattr(args, "dry_run")),
        log_level=getattr(args, "log_level", "INFO"),
    )
    return exec_config


def _summarise_results(results: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "output_dir": str(results.get("output_dir")),
        "chat_lora_dir": str(results.get("chat_lora_dir")),
        "wrapper_dir": str(results.get("wrapper_dir")),
        "wrapper_module": str(results.get("wrapper_module")),
        "wrapper_config": str(results.get("wrapper_config")),
    }
    if results.get("merged_dir") is not None:
        summary["merged_dir"] = str(results["merged_dir"])
    if results.get("dataset_size") is not None:
        summary["dataset_size"] = int(results["dataset_size"])
    if results.get("evaluation_metrics"):
        summary["evaluation_metrics"] = results["evaluation_metrics"]
    return summary


def _dataset_metrics_for_manifest(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    prompt = summary.get("prompt", {})
    completion = summary.get("completion", {})
    return {
        "examples": summary.get("examples"),
        "unique_prompts": summary.get("unique_prompts"),
        "duplicate_ratio": summary.get("duplicate_ratio"),
        "prompt_mean_chars": prompt.get("chars", {}).get("mean"),
        "completion_mean_chars": completion.get("chars", {}).get("mean"),
        "prompt_mean_tokens": prompt.get("tokens", {}).get("mean"),
        "completion_mean_tokens": completion.get("tokens", {}).get("mean"),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    execution = _build_execution(parser, args)

    dataset_summary: dict[str, Any] | None = None
    if execution.dataset_path is not None:
        dataset_summary = _summarise_local_dataset(execution.dataset_path)
        LOGGER.info(
            "Dataset characteristics",
            extra={
                "examples": dataset_summary["examples"],
                "unique_prompts": dataset_summary["unique_prompts"],
                "duplicate_ratio": dataset_summary["duplicate_ratio"],
                "prompt_mean_chars": dataset_summary["prompt"]["chars"]["mean"],
                "completion_mean_chars": dataset_summary["completion"]["chars"]["mean"],
            },
        )

    payload: dict[str, Any] = {
        "model_id": DEFAULT_MODEL_ID,
        "dataset_id": execution.dataset_id,
        "dataset_path": str(execution.dataset_path) if execution.dataset_path else None,
        "output_dir": str(execution.output_dir),
        "run_name": execution.run_name,
    }
    if dataset_summary is not None:
        payload["dataset_summary"] = dataset_summary

    if execution.dry_run:
        if execution.emit_json:
            print(json.dumps(payload, indent=2))
        else:
            LOGGER.info("Dry run completed", extra=payload)
        return payload

    execution.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Starting Dolphin-X1-8B fine-tune",
        extra={
            "model_id": DEFAULT_MODEL_ID,
            "dataset_id": execution.dataset_id,
            "dataset_path": (
                str(execution.dataset_path) if execution.dataset_path else None
            ),
            "output_dir": str(execution.output_dir),
            "run_name": execution.run_name,
        },
    )

    try:
        results = run_unsloth_finetune(
            model_id=DEFAULT_MODEL_ID,
            output_dir=execution.output_dir,
            dataset_id=execution.dataset_id,
            dataset_path=execution.dataset_path,
            max_seq_len=execution.max_seq_len,
            vram_budget_mb=execution.vram_budget_mb,
            activation_buffer_mb=execution.activation_buffer_mb,
            batch_size=execution.batch_size,
            grad_accum=execution.grad_accum,
            learning_rate=execution.learning_rate,
            epochs=execution.epochs,
            max_steps=execution.max_steps,
            lora_rank=execution.lora_rank,
            lora_alpha=execution.lora_alpha,
            lora_dropout=execution.lora_dropout,
            train_fraction=execution.train_fraction,
            eval_dataset_id=execution.eval_dataset_id,
            eval_dataset_path=execution.eval_dataset_path,
            eval_batch_size=execution.eval_batch_size,
            run_smoke_tests=not execution.skip_smoke_tests,
            write_metadata=not execution.skip_metadata,
            merge_to_fp16=not execution.skip_merge,
        )
    except Exception as exc:  # pragma: no cover - delegated to pipeline
        LOGGER.exception("Fine-tune failed")
        raise SystemExit(1) from exc

    LOGGER.info(
        "Fine-tune completed",
        extra={
            "dataset_size": results.get("dataset_size"),
            "eval_dataset_size": results.get("eval_dataset_size"),
            "wrapper_dir": str(results.get("wrapper_dir")),
            "merged_dir": (
                str(results.get("merged_dir")) if results.get("merged_dir") else None
            ),
        },
    )

    dataset_summary_path: Path | None = None
    if dataset_summary is not None:
        dataset_summary_path = execution.output_dir / "dataset_summary.json"
        dataset_summary_path.write_text(
            json.dumps(dataset_summary, indent=2),
            encoding="utf-8",
        )
        payload["dataset_summary_path"] = str(dataset_summary_path)

    if not execution.skip_metadata:
        metadata_path = execution.output_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {}
        if dataset_summary is not None:
            metadata["dataset_summary"] = dataset_summary
        if execution.run_name:
            metadata["run_name"] = execution.run_name
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    registry_path = execution.registry_path
    manifest = None
    if registry_path is not None:
        metrics: dict[str, Any] = {}
        dataset_metrics = _dataset_metrics_for_manifest(dataset_summary)
        if dataset_metrics:
            metrics["dataset"] = dataset_metrics
        if results.get("dataset_size") is not None:
            metrics.setdefault("dataset", {})["tokenized_examples"] = int(
                results["dataset_size"]
            )
        if results.get("evaluation_metrics"):
            metrics["evaluation"] = results["evaluation_metrics"]
        labels = {"pipeline": "mongars_unsloth"}
        if execution.run_name:
            labels["run"] = execution.run_name
        summary = build_adapter_summary(
            adapter_dir=results["chat_lora_dir"],
            weights_path=results.get("merged_dir"),
            wrapper_dir=results.get("wrapper_dir"),
            status="success",
            labels=labels,
            metrics=metrics,
            training={
                "model_id": DEFAULT_MODEL_ID,
                "dataset_id": execution.dataset_id,
                "dataset_path": (
                    str(execution.dataset_path) if execution.dataset_path else None
                ),
                "max_seq_len": execution.max_seq_len,
                "batch_size": execution.batch_size,
                "grad_accum": execution.grad_accum,
                "learning_rate": execution.learning_rate,
                "epochs": execution.epochs,
                "max_steps": execution.max_steps,
                "lora_rank": execution.lora_rank,
                "lora_alpha": execution.lora_alpha,
                "lora_dropout": execution.lora_dropout,
                "train_fraction": execution.train_fraction,
            },
        )
        manifest = update_manifest(registry_path, summary)
        LOGGER.info(
            "Adapter manifest updated",
            extra={"manifest_path": str(manifest.path)},
        )

    payload.update(_summarise_results(results))
    if manifest is not None:
        payload["manifest_path"] = str(manifest.path)

    if execution.emit_json:
        print(json.dumps(payload, indent=2))
    else:
        LOGGER.info("Training artifacts", extra=payload)

    return payload


if __name__ == "__main__":
    main()
