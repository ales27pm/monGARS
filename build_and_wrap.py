#!/usr/bin/env python3
"""QLoRA fine-tuning pipeline for Dolphin3.0-Llama3.1-8B."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from modules.neurons.registry import update_manifest
from monGARS.mlops.artifacts import (
    WrapperConfig,
    build_adapter_summary,
    render_output_bundle_readme,
    write_wrapper_bundle,
)
from monGARS.mlops.dataset import prepare_instruction_dataset
from monGARS.mlops.diagnostics.analysis import analyse_cuda_state
from monGARS.mlops.diagnostics.cuda_metrics import gather_cuda_metrics
from monGARS.mlops.exporters import GGUFExportResult, export_to_gguf, merge_lora_adapters
from monGARS.mlops.model import load_4bit_causal_lm, summarise_device_map
from monGARS.mlops.training import (
    LoraHyperParams,
    TrainerConfig,
    disable_training_mode,
    prepare_lora_model_light,
    save_lora_artifacts,
    train_qlora,
)
from monGARS.mlops.utils import (
    configure_cuda_allocator,
    describe_environment,
    ensure_dependencies,
    ensure_directory,
)

OVR_ENV_MAP = {
    "per_device_train_batch_size": "OVR_PER_DEVICE_TRAIN_BATCH_SIZE",
    "gradient_accumulation_steps": "OVR_GRAD_ACCUM_STEPS",
    "per_device_eval_batch_size": "OVR_PER_DEVICE_EVAL_BATCH_SIZE",
    "max_seq_length": "OVR_MAX_SEQ_LEN",
    "eval_max_seq_length": "OVR_EVAL_MAX_SEQ_LEN",
    "torch_dtype": "OVR_TORCH_DTYPE",
    "dtype": "OVR_TORCH_DTYPE",
    "gradient_checkpointing": "OVR_GRAD_CHECKPOINT",
    "attention_implementation": "OVR_ATTN_IMPL",
    "use_4bit": "OVR_USE_4BIT",
    "bnb_4bit_quant_type": "OVR_BNB_QUANT",
    "bnb_4bit_compute_dtype": "OVR_BNB_COMP_DTYPE",
    "lora_r": "OVR_LORA_R",
    "lora_alpha": "OVR_LORA_ALPHA",
    "lora_dropout": "OVR_LORA_DROPOUT",
}


def _load_json_overrides() -> dict[str, Any]:
    path = os.environ.get("TRAINER_OVERRIDES_JSON")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle).get("trainer_overrides", {})
        except Exception:
            return {}
    return {}


_OVR_JSON = _load_json_overrides()


def ovr(key: str, default: Any | None = None) -> Any | None:
    env_key = OVR_ENV_MAP.get(key)
    if env_key and (value := os.environ.get(env_key)) is not None:
        lowered = value.lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        try:
            return int(value)
        except Exception:
            return value
    return _OVR_JSON.get(key, default)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _coerce_dtype(value: str | None, fallback: str) -> str:
    if value is None or value == "":
        return fallback
    return str(value).lower()


_DEFAULT_DTYPE_NAME = os.environ.get("TORCH_DTYPE", "bfloat16")
_DTYPE_NAME = _coerce_dtype(ovr("dtype", ovr("torch_dtype", None)), _DEFAULT_DTYPE_NAME)
_BNB_COMPUTE_NAME = _coerce_dtype(
    ovr("bnb_4bit_compute_dtype", None), _DTYPE_NAME or _DEFAULT_DTYPE_NAME
)

if torch is not None:
    _DTYPE_MAP = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
    }
    DEFAULT_TORCH_DTYPE = _DTYPE_MAP.get(
        _coerce_dtype(_DEFAULT_DTYPE_NAME, "bfloat16"), torch.bfloat16
    )
    SELECTED_TORCH_DTYPE = _DTYPE_MAP.get(_DTYPE_NAME, DEFAULT_TORCH_DTYPE)
    BNB_COMPUTE_DTYPE = _DTYPE_MAP.get(_BNB_COMPUTE_NAME, SELECTED_TORCH_DTYPE)
else:  # pragma: no cover - torch optional in tests
    SELECTED_TORCH_DTYPE = None
    BNB_COMPUTE_DTYPE = None

ATTENTION_IMPLEMENTATION = (ovr("attention_implementation", None) or "").strip() or None

MODEL_ID = os.environ.get("MODEL_ID", "dphn/Dolphin3.0-Llama3.1-8B")
DATASET_NAME = os.environ.get("DATASET", "yahma/alpaca-cleaned")
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", "0.10"))
DEFAULT_MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
MAX_SEQ_LEN = int(ovr("max_seq_length", DEFAULT_MAX_SEQ_LEN))
EVAL_MAX_SEQ_LEN = int(ovr("eval_max_seq_length", MAX_SEQ_LEN))
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
BATCH_SIZE = int(ovr("per_device_train_batch_size", DEFAULT_BATCH_SIZE))
EVAL_BATCH_SIZE = int(ovr("per_device_eval_batch_size", BATCH_SIZE))
DEFAULT_GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
GRAD_ACCUM = int(ovr("gradient_accumulation_steps", DEFAULT_GRAD_ACCUM))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "1.0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
VRAM_BUDGET_MB = int(os.environ.get("VRAM_BUDGET_MB", "7300"))
ACTIVATION_BUFFER_MB = int(
    os.environ.get(
        "ACTIVATION_BUFFER_MB", os.environ.get("VRAM_ACTIVATION_BUFFER_MB", "1024")
    )
)
RUNTIME_BUFFER_MB = int(
    os.environ.get("RUNTIME_BUFFER_MB", os.environ.get("VRAM_RUNTIME_BUFFER_MB", "768"))
)
OFFLOAD_DIR = Path(os.environ.get("OFFLOAD_DIR", "./offload"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./out"))
EXPORT_MERGED_FP16 = os.environ.get("EXPORT_MERGED_FP16", "0") == "1"
EXPORT_GGUF = os.environ.get("EXPORT_GGUF", "0") == "1"
GGUF_DIR = Path(os.environ.get("GGUF_DIR", OUTPUT_DIR / "gguf"))
GGUF_METHOD = os.environ.get("GGUF_METHOD", "q4_k_m")
AUTO_INSTALL = os.environ.get("AUTO_INSTALL", "1") == "1"
OOM_MIN_FREE_GIB = float(os.environ.get("OOM_MIN_FREE_GIB", "1.0"))
OOM_MIN_FREE_RATIO = float(os.environ.get("OOM_MIN_FREE_RATIO", "0.1"))
FAIL_ON_CRITICAL_OOM = os.environ.get("FAIL_ON_CRITICAL_OOM", "1") == "1"
REGISTRY_PATH = os.environ.get("LLM_ADAPTER_REGISTRY_PATH") or os.environ.get(
    "ADAPTER_REGISTRY_PATH"
)
SUMMARY_FILENAME = "training_summary.json"

EXPORT_AWQ = _env_flag("EXPORT_AWQ", EXPORT_MERGED_FP16)
AWQ_DIR = Path(os.environ.get("AWQ_DIR", OUTPUT_DIR / "awq_model"))
AWQ_W_BITS = int(os.environ.get("AWQ_W_BITS", "4"))
AWQ_GROUP_SIZE = int(os.environ.get("AWQ_GROUP_SIZE", "128"))
AWQ_ZERO_POINT = _env_flag("AWQ_ZERO_POINT", True)
AWQ_VERSION = os.environ.get("AWQ_VERSION", "GEMM")
AWQ_CALIB_DATASET = os.environ.get("AWQ_CALIB_DATASET", "wikitext2")
AWQ_CALIB_SAMPLES = int(os.environ.get("AWQ_CALIB_SAMPLES", "128"))
AWQ_CALIB_SEQ_LEN = int(os.environ.get("AWQ_CALIB_SEQ_LEN", "2048"))
AWQ_TRUST_REMOTE_CODE = _env_flag("AWQ_TRUST_REMOTE_CODE", False)


REQUIRED_PACKAGES = [
    "torch",
    "transformers>=4.44",
    "datasets",
    "peft>=0.11",
    "bitsandbytes>=0.44.1",
    "llm2vec",
    "sentencepiece",
    "autoawq",
]
OPTIONAL_PACKAGES = ["unsloth"]


def _gather_default_cuda_metrics(module: Any) -> dict[str, Any] | None:
    """Collect CUDA diagnostics for all available devices."""

    def device_indices() -> list[int]:
        return list(range(module.cuda.device_count()))

    return gather_cuda_metrics(module, device_indices)


def _locate_adapter_weights(adapter_dir: Path) -> Path | None:
    candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def export_awq_quantized_model(
    merged_dir: Path,
    output_dir: Path,
    *,
    w_bits: int,
    group_size: int,
    zero_point: bool,
    version: str,
    calib_dataset: str | None,
    calib_samples: int,
    calib_seq_len: int,
    trust_remote_code: bool,
) -> bool:
    """Export the merged model to AWQ format."""

    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged model directory does not exist: {merged_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_directory(output_dir)

    try:
        from autoawq import AutoAWQForCausalLM  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "autoawq must be installed to export AWQ quantized models"
        ) from exc

    from transformers import AutoTokenizer

    quant_config: dict[str, Any] = {
        "w_bit": int(w_bits),
        "q_group_size": int(group_size),
        "zero_point": bool(zero_point),
        "version": version,
        "calib_samples": int(calib_samples),
        "calib_seqlen": int(calib_seq_len),
    }
    if calib_dataset:
        quant_config["calib_dataset"] = calib_dataset

    logger.info("Loading merged model for AWQ quantization from %s", merged_dir)
    model = AutoAWQForCausalLM.from_pretrained(
        str(merged_dir),
        device_map="auto",
        safetensors=True,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(merged_dir), trust_remote_code=trust_remote_code
    )

    logger.info("Running AWQ quantization with config: %s", quant_config)
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(str(output_dir), safetensors=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved AWQ quantized model to %s", output_dir)
    return True


def _attach_gguf_summary(
    summary: dict[str, Any],
    *,
    gguf_result: GGUFExportResult,
    requested_method: str | None,
) -> None:
    artifacts = summary.setdefault("artifacts", {})
    artifacts["gguf"] = str(gguf_result.path)

    labels = summary.setdefault("labels", {})
    if requested_method:
        labels["gguf_method"] = requested_method
    if gguf_result.quantization_method:
        labels.setdefault("gguf_quantization", gguf_result.quantization_method)
    labels["gguf_export_interface"] = gguf_result.method
    labels["gguf_exporter"] = gguf_result.exporter

    metrics = summary.setdefault("metrics", {})
    if gguf_result.quantization_method:
        metrics.setdefault("gguf_quantization_method", gguf_result.quantization_method)
    try:
        metrics.setdefault("gguf_file_size_bytes", gguf_result.path.stat().st_size)
    except OSError:
        logger.debug(
            "Unable to stat GGUF artifact", extra={"path": str(gguf_result.path)}
        )


def _attach_awq_summary(summary: dict[str, Any], *, awq_dir: Path) -> None:
    artifacts = summary.setdefault("artifacts", {})
    artifacts["awq"] = str(awq_dir)

    labels = summary.setdefault("labels", {})
    labels.setdefault("awq_version", AWQ_VERSION)
    labels.setdefault("awq_precision", f"{AWQ_W_BITS}-bit")

    quant_summary = summary.setdefault("quantization", {})
    awq_config = {
        "w_bit": AWQ_W_BITS,
        "q_group_size": AWQ_GROUP_SIZE,
        "zero_point": AWQ_ZERO_POINT,
        "version": AWQ_VERSION,
        "calib_samples": AWQ_CALIB_SAMPLES,
        "calib_seq_len": AWQ_CALIB_SEQ_LEN,
    }
    if AWQ_CALIB_DATASET:
        awq_config["calib_dataset"] = AWQ_CALIB_DATASET
    quant_summary["awq"] = awq_config


def _assemble_training_summary(
    *,
    adapter_dir: Path,
    weights_path: Path | None,
    wrapper_dir: Path,
    merged_dir: Path,
    merged: bool,
    gguf_enabled: bool,
    gguf_requested_method: str | None,
    gguf_result: GGUFExportResult | None,
    awq_dir: Path,
    awq_enabled: bool,
    dataset_len: int,
    oom_analysis: dict[str, Any],
) -> dict[str, Any]:
    summary = build_adapter_summary(
        adapter_dir=adapter_dir,
        weights_path=weights_path,
        wrapper_dir=wrapper_dir,
        status="success",
        labels={
            "category": "general_baseline",
            "quantization": "bnb_nf4",
            "first_run": "true",
        },
        metrics={
            "dataset_size": dataset_len,
            "train_fraction": TRAIN_FRACTION,
        },
        training={
            "base_model": MODEL_ID,
            "dataset": DATASET_NAME,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "max_steps": MAX_STEPS,
            "eval_max_seq_len": EVAL_MAX_SEQ_LEN,
            "vram_budget_mb": VRAM_BUDGET_MB,
            "activation_buffer_mb": ACTIVATION_BUFFER_MB,
            "runtime_buffer_mb": RUNTIME_BUFFER_MB,
            "quantization_method": "bnb-4bit-nf4",
        },
    )

    artifacts = summary.setdefault("artifacts", {})
    if merged:
        artifacts["merged_fp16"] = str(merged_dir)
    if gguf_enabled and merged and gguf_result is not None:
        _attach_gguf_summary(
            summary,
            gguf_result=gguf_result,
            requested_method=gguf_requested_method,
        )
    if awq_enabled:
        _attach_awq_summary(summary, awq_dir=awq_dir)

    summary.setdefault("analysis", {})["oom_risk"] = oom_analysis

    return summary


def _save_summary(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2))


def _maybe_update_registry(registry_path: str | None, summary: dict[str, Any]) -> None:
    if registry_path:
        try:
            manifest = update_manifest(registry_path, summary)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"! Failed to update adapter manifest: {exc}")
        else:
            print(f"- Adapter manifest updated: {manifest.path}")
    else:
        print(
            "- Adapter manifest not updated (set LLM_ADAPTER_REGISTRY_PATH to register output)"
        )


def _log_oom_report(analysis: dict[str, Any]) -> None:
    status = analysis.get("status", "unknown")
    logger.info("OOM risk classification: %s", status)

    for device in analysis.get("devices", []) or []:
        index = device.get("index", "?")
        free = device.get("free_gib")
        ratio = device.get("free_ratio")
        device_status = device.get("status", "unknown")
        if free is not None and ratio is not None:
            logger.info(
                "Device %s: %.2f GiB free (%.1f%%) -> %s",
                index,
                free,
                ratio * 100,
                device_status,
            )
        else:
            logger.info("Device %s: status %s", index, device_status)
        for recommendation in device.get("recommendations", []):
            logger.info("Device %s recommendation: %s", index, recommendation)


def _raise_on_critical(analysis: dict[str, Any], fail_on_critical: bool) -> None:
    if not fail_on_critical or analysis.get("status") != "critical":
        return

    recommendations: list[str] = [
        recommendation
        for device in analysis.get("devices", []) or []
        for recommendation in device.get("recommendations", [])
    ]
    guidance = "\n".join(f"  - {text}" for text in recommendations)
    if not guidance:
        guidance = "  - See diagnose_unsloth guidance."

    raise RuntimeError(
        "Insufficient CUDA headroom for QLoRA fine-tuning.\n"
        "VRAM diagnostics flagged a critical OOM risk.\n"
        f"Recommended mitigations:\n{guidance}"
    )


def evaluate_oom_headroom(
    *,
    min_free_gib: float = OOM_MIN_FREE_GIB,
    min_free_ratio: float = OOM_MIN_FREE_RATIO,
    fail_on_critical: bool = FAIL_ON_CRITICAL_OOM,
    torch_module: Any | None = None,
    gather_metrics: Callable[[Any], dict[str, Any] | None] | None = None,
    analyse_state: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assess CUDA headroom and optionally abort if the risk is critical."""

    skip_reason: str | None = None
    cuda_payload: dict[str, Any] | None = None

    torch_module = torch_module if torch_module is not None else torch

    if torch_module is None:
        skip_reason = "torch_missing"
    elif not hasattr(torch_module, "cuda"):
        skip_reason = "cuda_interface_missing"
    elif not callable(getattr(torch_module.cuda, "is_available", None)):
        skip_reason = "cuda_interface_missing"
    elif not torch_module.cuda.is_available():
        skip_reason = "cuda_unavailable"
    else:
        metrics_fn = gather_metrics
        if metrics_fn is None:
            metrics_fn = _gather_default_cuda_metrics
        cuda_payload = metrics_fn(torch_module)
        if cuda_payload is None:
            skip_reason = "cuda_metrics_unavailable"

    analyser = analyse_state or analyse_cuda_state
    analysis = analyser(
        cuda_payload,
        min_free_gib=min_free_gib,
        min_free_ratio=min_free_ratio,
        skip_reason=skip_reason,
    )

    _log_oom_report(analysis)
    _raise_on_critical(analysis, fail_on_critical)

    return analysis


def main() -> None:
    configure_cuda_allocator()
    ensure_directory(OFFLOAD_DIR)
    ensure_directory(OUTPUT_DIR)
    ensure_dependencies(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, auto_install=AUTO_INSTALL)
    describe_environment()

    model, tokenizer = load_4bit_causal_lm(
        MODEL_ID,
        vram_budget_mb=VRAM_BUDGET_MB,
        activation_buffer_mb=ACTIVATION_BUFFER_MB,
        runtime_buffer_mb=RUNTIME_BUFFER_MB,
        offload_dir=OFFLOAD_DIR,
        dtype=SELECTED_TORCH_DTYPE,
        compute_dtype=BNB_COMPUTE_DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )
    summarise_device_map(model)
    oom_analysis = evaluate_oom_headroom()
    default_lora = LoraHyperParams()
    lora_params = LoraHyperParams(
        r=int(ovr("lora_r", default_lora.r)),
        alpha=int(ovr("lora_alpha", default_lora.alpha)),
        dropout=float(ovr("lora_dropout", default_lora.dropout)),
        target_modules=default_lora.target_modules,
    )
    model = prepare_lora_model_light(model, lora_params)

    dataset = prepare_instruction_dataset(
        DATASET_NAME,
        tokenizer,
        MAX_SEQ_LEN,
        train_fraction=TRAIN_FRACTION,
    )

    trainer = train_qlora(
        model,
        dataset,
        config=TrainerConfig(
            output_dir=OUTPUT_DIR / "chat_lora",
            batch_size=BATCH_SIZE,
            grad_accum=GRAD_ACCUM,
            learning_rate=LR,
            epochs=EPOCHS,
            max_steps=MAX_STEPS,
        ),
        extra_args={
            "per_device_eval_batch_size": max(1, EVAL_BATCH_SIZE),
            "gradient_checkpointing": bool(ovr("gradient_checkpointing", True)),
        },
    )

    adapters_dir = OUTPUT_DIR / "chat_lora"
    save_lora_artifacts(trainer.model, tokenizer, adapters_dir)
    disable_training_mode(trainer.model)
    weights_path = _locate_adapter_weights(adapters_dir)

    merged_dir = OUTPUT_DIR / "merged_fp16"
    merged = False
    gguf_export: GGUFExportResult | None = None
    if EXPORT_MERGED_FP16:
        merged = merge_lora_adapters(MODEL_ID, adapters_dir, output_dir=merged_dir)

    awq_exported = False
    if EXPORT_AWQ and not EXPORT_MERGED_FP16:
        logger.warning(
            "EXPORT_AWQ is enabled but EXPORT_MERGED_FP16 is disabled; skipping AWQ export."
        )
    if merged and EXPORT_AWQ:
        awq_exported = export_awq_quantized_model(
            merged_dir,
            AWQ_DIR,
            w_bits=AWQ_W_BITS,
            group_size=AWQ_GROUP_SIZE,
            zero_point=AWQ_ZERO_POINT,
            version=AWQ_VERSION,
            calib_dataset=AWQ_CALIB_DATASET,
            calib_samples=AWQ_CALIB_SAMPLES,
            calib_seq_len=AWQ_CALIB_SEQ_LEN,
            trust_remote_code=AWQ_TRUST_REMOTE_CODE,
        )

    if EXPORT_GGUF:
        if not merged:
            raise RuntimeError("EXPORT_GGUF requires EXPORT_MERGED_FP16=1")
        gguf_export = export_to_gguf(
            str(merged_dir),
            str(GGUF_DIR),
            quantization_method=GGUF_METHOD,
        )

    wrapper_dir = OUTPUT_DIR / "wrapper"
    bundle_config = WrapperConfig(
        base_model_id=MODEL_ID,
        lora_dir=adapters_dir.resolve(),
        max_seq_len=MAX_SEQ_LEN,
        vram_budget_mb=VRAM_BUDGET_MB,
        offload_dir=OFFLOAD_DIR.resolve(),
        activation_buffer_mb=ACTIVATION_BUFFER_MB,
    )
    write_wrapper_bundle(bundle_config, wrapper_dir)

    readme = render_output_bundle_readme(
        bundle_config,
        merged_fp16=merged,
        gguf_enabled=EXPORT_GGUF and merged,
        gguf_method=GGUF_METHOD,
    )
    (OUTPUT_DIR / "README_outputs.md").write_text(readme)

    summary = _assemble_training_summary(
        adapter_dir=adapters_dir,
        weights_path=weights_path,
        wrapper_dir=wrapper_dir,
        merged_dir=merged_dir,
        merged=merged,
        gguf_enabled=EXPORT_GGUF,
        gguf_requested_method=GGUF_METHOD if EXPORT_GGUF else None,
        gguf_result=gguf_export,
        awq_dir=AWQ_DIR,
        awq_enabled=awq_exported,
        dataset_len=len(dataset),
        oom_analysis=oom_analysis,
    )
    summary_path = OUTPUT_DIR / SUMMARY_FILENAME
    _save_summary(summary, summary_path)

    print("=== ALL DONE ===")
    print(f"- LoRA adapters: {adapters_dir}")
    if merged:
        print(f"- Merged FP16 model: {merged_dir}")
    if awq_exported:
        print(f"- AWQ quantized model: {AWQ_DIR}")
    if EXPORT_GGUF and merged:
        gguf_path_display = gguf_export.path if gguf_export is not None else GGUF_DIR
        print(f"- GGUF export: {gguf_path_display}")
    print(f"- Wrapper module: {wrapper_dir / 'project_wrapper.py'}")
    print(f"- Training summary: {summary_path}")
    _maybe_update_registry(REGISTRY_PATH, summary)


if __name__ == "__main__":
    main()
