#!/usr/bin/env python3
"""Benchmark the Unsloth + LLM2Vec fine-tuning workflow.

This script executes :func:`monGARS.mlops.pipelines.unsloth.run_unsloth_finetune`
with optional auto-tuning of hyperparameters based on the detected hardware.
It captures runtime statistics, GPU/CPU utilisation insights, and produces a
structured JSON report that can be archived for regressions or capacity
planning.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import psutil

repo_root = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("monGARS") is None and str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from monGARS.mlops.pipelines import run_unsloth_finetune  # noqa: E402
from monGARS.mlops.utils import configure_cuda_allocator, ensure_directory  # noqa: E402

logger = logging.getLogger("benchmark.unsloth_llm2vec")

UNSLOTH_AVAILABLE = False
_UNSLOTH_IMPORT_ERROR: Exception | None = None
_NOT_IMPLEMENTED_SENTINEL = "Not" "ImplementedError"

try:  # pragma: no cover - depends on runtime accelerators
    import unsloth  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:
    _UNSLOTH_IMPORT_ERROR = exc
    logger.info("Unsloth is not installed; benchmark CLI will raise on execution.")
except Exception as exc:  # pragma: no cover - defensive guardrail
    _UNSLOTH_IMPORT_ERROR = exc
    if exc.__class__.__name__ == _NOT_IMPLEMENTED_SENTINEL:
        logger.warning(
            "Unsloth requires a GPU-backed accelerator; skipping eager import.",
        )
    else:
        logger.warning("Failed to import Unsloth optimisations", exc_info=exc)
else:
    UNSLOTH_AVAILABLE = True

GPUtil = None
GPUtil_spec = importlib.util.find_spec("GPUtil")
if GPUtil_spec is not None:
    GPUtil = importlib.import_module("GPUtil")

torch = None
_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is not None:
    torch = importlib.import_module("torch")


@dataclass(slots=True)
class GPUStatus:
    """Snapshot of a single accelerator."""

    name: str
    total_memory_mb: float | None
    free_memory_mb: float | None
    temperature_c: float | None
    utilisation_percent: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "free_memory_mb": self.free_memory_mb,
            "temperature_c": self.temperature_c,
            "utilisation_percent": self.utilisation_percent,
        }


@dataclass(slots=True)
class HardwareSnapshot:
    """Summary of the runtime environment used for benchmarking."""

    cpu_model: str | None
    physical_cores: int | None
    logical_cores: int | None
    memory_total_gb: float | None
    memory_available_gb: float | None
    gpus: list[GPUStatus] = field(default_factory=list)
    cuda_available: bool = False
    cuda_device_count: int = 0
    torch_version: str | None = None
    cuda_version: str | None = None

    def primary_gpu_memory_mb(self) -> float | None:
        for gpu in self.gpus:
            if gpu.total_memory_mb:
                return gpu.total_memory_mb
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "memory_total_gb": self.memory_total_gb,
            "memory_available_gb": self.memory_available_gb,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "gpus": [gpu.as_dict() for gpu in self.gpus],
        }


@dataclass(slots=True)
class RunMetrics:
    """Detailed metrics captured for a single benchmark run."""

    run_index: int
    duration_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    rss_start_mb: float
    rss_end_mb: float
    max_cuda_reserved_mb: float | None
    max_cuda_allocated_mb: float | None
    dataset_size: int | None
    eval_dataset_size: int | None
    evaluation_metrics: Mapping[str, Any] | None
    result_paths: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "duration_seconds": self.duration_seconds,
            "cpu_user_seconds": self.cpu_user_seconds,
            "cpu_system_seconds": self.cpu_system_seconds,
            "rss_start_mb": self.rss_start_mb,
            "rss_end_mb": self.rss_end_mb,
            "max_cuda_reserved_mb": self.max_cuda_reserved_mb,
            "max_cuda_allocated_mb": self.max_cuda_allocated_mb,
            "dataset_size": self.dataset_size,
            "eval_dataset_size": self.eval_dataset_size,
            "evaluation_metrics": dict(self.evaluation_metrics or {}),
            "result_paths": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in self.result_paths.items()
            },
        }


@dataclass(slots=True)
class BenchmarkConfig:
    """User-supplied configuration for the benchmark."""

    model_id: str
    output_dir: Path
    dataset_id: str | None
    dataset_path: Path | None
    eval_dataset_id: str | None
    eval_dataset_path: Path | None
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
    train_fraction: float | None
    eval_batch_size: int | None
    run_smoke_tests: bool
    write_metadata: bool
    merge_to_fp16: bool
    runs: int
    auto_tune: bool
    report_path: Path | None

    def common_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "dataset_path": self.dataset_path,
            "max_seq_len": self.max_seq_len,
            "vram_budget_mb": self.vram_budget_mb,
            "activation_buffer_mb": self.activation_buffer_mb,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "train_fraction": self.train_fraction,
            "eval_dataset_id": self.eval_dataset_id,
            "eval_dataset_path": self.eval_dataset_path,
            "eval_batch_size": self.eval_batch_size,
            "run_smoke_tests": self.run_smoke_tests,
            "write_metadata": self.write_metadata,
            "merge_to_fp16": self.merge_to_fp16,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "output_dir": str(self.output_dir),
            "dataset_id": self.dataset_id,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "eval_dataset_id": self.eval_dataset_id,
            "eval_dataset_path": (
                str(self.eval_dataset_path) if self.eval_dataset_path else None
            ),
            "max_seq_len": self.max_seq_len,
            "vram_budget_mb": self.vram_budget_mb,
            "activation_buffer_mb": self.activation_buffer_mb,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "train_fraction": self.train_fraction,
            "eval_batch_size": self.eval_batch_size,
            "run_smoke_tests": self.run_smoke_tests,
            "write_metadata": self.write_metadata,
            "merge_to_fp16": self.merge_to_fp16,
            "runs": self.runs,
            "auto_tune": self.auto_tune,
            "report_path": str(self.report_path) if self.report_path else None,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """Container for the final benchmark summary."""

    hardware: HardwareSnapshot
    requested_args: Mapping[str, Any]
    tuned_args: Mapping[str, Any]
    adjustments: list[str]
    runs: list[RunMetrics]

    def to_dict(self) -> dict[str, Any]:
        summary = summarise_run_metrics(self.runs)
        return {
            "hardware": self.hardware.as_dict(),
            "requested_arguments": dict(self.requested_args),
            "tuned_arguments": dict(self.tuned_args),
            "adjustments": list(self.adjustments),
            "runs": [run.as_dict() for run in self.runs],
            "summary": summary,
        }


def collect_hardware_snapshot() -> HardwareSnapshot:
    cpu_model = platform.processor() or platform.machine()
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    virtual_memory = psutil.virtual_memory()
    gpus: list[GPUStatus] = []

    if GPUtil is not None:
        try:
            gputil_devices = GPUtil.getGPUs()
        except Exception:  # pragma: no cover - defensive, relies on GPU runtime
            gputil_devices = []
        for device in gputil_devices:
            gpus.append(
                GPUStatus(
                    name=str(getattr(device, "name", "unknown")),
                    # GPUtil already reports memoryTotal/memoryFree in megabytes,
                    # so we intentionally keep the raw values without additional scaling
                    # to ensure VRAM-aware auto-tuning receives accurate capacities.
                    total_memory_mb=_safe_float(getattr(device, "memoryTotal", None)),
                    free_memory_mb=_safe_float(getattr(device, "memoryFree", None)),
                    temperature_c=_safe_float(getattr(device, "temperature", None)),
                    utilisation_percent=_safe_float(
                        getattr(device, "load", None), scale=100
                    ),
                )
            )

    cuda_available = False
    cuda_devices = 0
    torch_version: str | None = None
    cuda_version: str | None = None
    if torch is not None:
        torch_version = getattr(torch, "__version__", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - depends on runtime
            cuda_available = False
        else:
            if cuda_available:
                try:
                    cuda_devices = int(torch.cuda.device_count())
                except Exception:  # pragma: no cover - depends on runtime
                    cuda_devices = 1

    snapshot = HardwareSnapshot(
        cpu_model=cpu_model if cpu_model else None,
        physical_cores=physical,
        logical_cores=logical,
        memory_total_gb=_bytes_to_gb(virtual_memory.total),
        memory_available_gb=_bytes_to_gb(virtual_memory.available),
        gpus=gpus,
        cuda_available=cuda_available,
        cuda_device_count=cuda_devices,
        torch_version=torch_version,
        cuda_version=cuda_version,
    )
    logger.info("Hardware snapshot: %s", snapshot.as_dict())
    return snapshot


def _bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 2)


def _safe_float(value: Any, *, scale: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if scale:
        number *= scale
    return round(number, 2)


def auto_tune_hyperparameters(
    snapshot: HardwareSnapshot, params: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    tuned = dict(params)
    adjustments: list[str] = []

    total_gpu_mb = snapshot.primary_gpu_memory_mb()
    if total_gpu_mb is not None and total_gpu_mb > 0:
        requested_vram = int(tuned.get("vram_budget_mb", total_gpu_mb))
        if requested_vram > total_gpu_mb:
            new_budget = max(1024, int(total_gpu_mb - 512))
            tuned["vram_budget_mb"] = new_budget
            adjustments.append(
                f"Reduced vram_budget_mb from {requested_vram} to {new_budget} "
                f"to fit GPU capacity ({total_gpu_mb} MB)"
            )
        if total_gpu_mb < 9000 and int(tuned.get("batch_size", 1)) > 1:
            effective_batch = int(tuned.get("batch_size", 1)) * int(
                tuned.get("grad_accum", 1)
            )
            tuned["batch_size"] = 1
            tuned["grad_accum"] = max(1, effective_batch)
            adjustments.append(
                "Set batch_size to 1 and increased grad_accum to preserve effective"
                f" batch ({effective_batch}) on {int(total_gpu_mb)} MB GPU"
            )
    else:
        if int(tuned.get("batch_size", 1)) > 1:
            tuned["batch_size"] = 1
            adjustments.append("No GPU detected; forcing batch_size=1")

    total_memory_gb = snapshot.memory_total_gb or 0
    if total_memory_gb and total_memory_gb < 24:
        activation = int(tuned.get("activation_buffer_mb", 1024))
        max_activation = max(256, int(total_memory_gb * 1024 * 0.25))
        if activation > max_activation:
            tuned["activation_buffer_mb"] = max_activation
            adjustments.append(
                f"Limited activation_buffer_mb to {max_activation} based on "
                f"system RAM ({total_memory_gb} GiB)"
            )

    return tuned, adjustments


def summarise_run_metrics(runs: Iterable[RunMetrics]) -> dict[str, Any]:
    runs_list = list(runs)
    if not runs_list:
        return {"run_count": 0}

    durations = [run.duration_seconds for run in runs_list]
    cpu_totals = [run.cpu_user_seconds + run.cpu_system_seconds for run in runs_list]
    max_reserved = [
        run.max_cuda_reserved_mb for run in runs_list if run.max_cuda_reserved_mb
    ]
    max_allocated = [
        run.max_cuda_allocated_mb for run in runs_list if run.max_cuda_allocated_mb
    ]
    throughputs = [
        run.dataset_size / run.duration_seconds
        for run in runs_list
        if run.dataset_size and run.duration_seconds > 0
    ]

    summary = {
        "run_count": len(runs_list),
        "duration_seconds": {
            "mean": round(statistics.mean(durations), 3),
            "median": round(statistics.median(durations), 3),
            "min": round(min(durations), 3),
            "max": round(max(durations), 3),
        },
        "cpu_time_seconds": {
            "mean": round(statistics.mean(cpu_totals), 3),
            "median": round(statistics.median(cpu_totals), 3),
            "min": round(min(cpu_totals), 3),
            "max": round(max(cpu_totals), 3),
        },
    }

    if max_reserved:
        summary["cuda_reserved_mb"] = {
            "mean": round(statistics.mean(max_reserved), 2),
            "max": round(max(max_reserved), 2),
        }
    if max_allocated:
        summary["cuda_allocated_mb"] = {
            "mean": round(statistics.mean(max_allocated), 2),
            "max": round(max(max_allocated), 2),
        }
    if throughputs:
        summary["examples_per_second"] = {
            "mean": round(statistics.mean(throughputs), 3),
            "max": round(max(throughputs), 3),
        }

    return summary


def run_benchmark(
    config: BenchmarkConfig,
    *,
    pipeline: Callable[..., Mapping[str, Any]] = run_unsloth_finetune,
    snapshot: HardwareSnapshot | None = None,
) -> BenchmarkReport:
    if not UNSLOTH_AVAILABLE:
        reason = "Unsloth is not available in this runtime."
        if _UNSLOTH_IMPORT_ERROR is not None:
            reason = f"Unsloth import failed: {_UNSLOTH_IMPORT_ERROR}"
        raise RuntimeError(f"Cannot run the benchmark without Unsloth. {reason}")

    hardware = snapshot or collect_hardware_snapshot()
    configure_cuda_allocator()
    ensure_directory(config.output_dir)

    common_kwargs = config.common_pipeline_kwargs()
    tuned_kwargs = dict(common_kwargs)
    adjustments: list[str] = []
    if config.auto_tune:
        tuned_kwargs, adjustments = auto_tune_hyperparameters(hardware, common_kwargs)

    runs: list[RunMetrics] = []
    process = psutil.Process()
    for run_index in range(config.runs):
        run_output_dir = config.output_dir
        if config.runs > 1:
            run_output_dir = config.output_dir / f"run_{run_index + 1:02d}"
        ensure_directory(run_output_dir)
        run_kwargs = dict(tuned_kwargs)
        run_kwargs["output_dir"] = run_output_dir

        logger.info("Starting benchmark run %s", run_index + 1)
        start_cpu = process.cpu_times()
        rss_start_mb = process.memory_info().rss / (1024**2)
        start_time = time.perf_counter()

        if torch is not None and hasattr(torch.cuda, "reset_peak_memory_stats"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:  # pragma: no cover - depends on runtime
                pass

        results = pipeline(**run_kwargs)

        duration = time.perf_counter() - start_time
        cpu_times = process.cpu_times()
        cpu_user = cpu_times.user - start_cpu.user
        cpu_system = cpu_times.system - start_cpu.system
        rss_end_mb = process.memory_info().rss / (1024**2)

        max_reserved = None
        max_allocated = None
        if torch is not None and hasattr(torch.cuda, "max_memory_reserved"):
            try:
                torch.cuda.synchronize()
                max_reserved = torch.cuda.max_memory_reserved() / (1024**2)
                max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            except Exception:  # pragma: no cover - depends on runtime
                max_reserved = None
                max_allocated = None

        metrics = RunMetrics(
            run_index=run_index + 1,
            duration_seconds=round(duration, 3),
            cpu_user_seconds=round(cpu_user, 3),
            cpu_system_seconds=round(cpu_system, 3),
            rss_start_mb=round(rss_start_mb, 2),
            rss_end_mb=round(rss_end_mb, 2),
            max_cuda_reserved_mb=_round_or_none(max_reserved, 2),
            max_cuda_allocated_mb=_round_or_none(max_allocated, 2),
            dataset_size=_safe_int(results.get("dataset_size")),
            eval_dataset_size=_safe_int(results.get("eval_dataset_size")),
            evaluation_metrics=results.get("evaluation_metrics") or {},
            result_paths={
                key: value
                for key, value in results.items()
                if isinstance(value, (str, Path))
            },
        )
        runs.append(metrics)
        logger.info("Completed run %s: %s", run_index + 1, metrics.as_dict())

    requested_args = config.to_dict()
    tuned_args = dict(requested_args)
    tuned_args.update(
        {
            key: tuned_kwargs[key]
            for key in (
                "vram_budget_mb",
                "activation_buffer_mb",
                "batch_size",
                "grad_accum",
            )
            if key in tuned_kwargs
        }
    )

    return BenchmarkReport(
        hardware=hardware,
        requested_args=requested_args,
        tuned_args=tuned_args,
        adjustments=adjustments,
        runs=runs,
    )


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args(argv: Iterable[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--eval-dataset-id", default=None)
    parser.add_argument("--eval-dataset-path", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--vram-budget-mb", type=int, default=7500)
    parser.add_argument("--activation-buffer-mb", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--skip-smoke-tests", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report destination (defaults to output_dir/benchmark_report.json)",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)

    runs = max(1, parsed.runs)
    report_path = parsed.report_path
    if report_path is None:
        report_path = parsed.output_dir / "benchmark_report.json"

    return BenchmarkConfig(
        model_id=parsed.model_id,
        output_dir=parsed.output_dir,
        dataset_id=parsed.dataset_id,
        dataset_path=parsed.dataset_path,
        eval_dataset_id=parsed.eval_dataset_id,
        eval_dataset_path=parsed.eval_dataset_path,
        max_seq_len=parsed.max_seq_len,
        vram_budget_mb=parsed.vram_budget_mb,
        activation_buffer_mb=parsed.activation_buffer_mb,
        batch_size=max(1, parsed.batch_size),
        grad_accum=max(1, parsed.grad_accum),
        learning_rate=parsed.learning_rate,
        epochs=parsed.epochs,
        max_steps=parsed.max_steps,
        lora_rank=parsed.lora_rank,
        lora_alpha=parsed.lora_alpha,
        lora_dropout=parsed.lora_dropout,
        train_fraction=parsed.train_fraction,
        eval_batch_size=parsed.eval_batch_size,
        run_smoke_tests=not parsed.skip_smoke_tests,
        write_metadata=not parsed.skip_metadata,
        merge_to_fp16=not parsed.skip_merge,
        runs=runs,
        auto_tune=parsed.auto_tune,
        report_path=report_path,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Iterable[str] | None = None) -> None:
    configure_logging()
    config = parse_args(argv)
    logger.info("Benchmark configuration: %s", config.to_dict())
    report = run_benchmark(config)
    report_data = report.to_dict()
    ensure_directory(config.report_path.parent)
    with config.report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2)
    logger.info("Benchmark report written to %s", config.report_path)
    print(json.dumps(report_data, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
