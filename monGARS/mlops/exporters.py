"""Utilities for persisting and exporting trained artefacts."""

from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


_GGUF_EXPORTER_CANDIDATES: tuple[str, ...] = (
    "transformers.GgufExporter",
    "transformers.GGUFExporter",
    "transformers.exporters.GgufExporter",
    "transformers.exporters.GGUFExporter",
    "transformers.exporters.gguf.GgufExporter",
    "transformers.exporters.gguf.GGUFExporter",
)


@dataclass(frozen=True)
class GGUFExporterInfo:
    """Metadata about a discovered GGUF exporter implementation."""

    qualified_name: str
    factory: Callable[..., Any]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the exporter using the recorded factory."""

        return self.factory(*args, **kwargs)


@dataclass(frozen=True)
class GGUFExportResult:
    """Details about a GGUF export invocation."""

    path: Path
    exporter: str
    method: str
    quantization_method: str | None

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:  # pragma: no cover - debug convenience
        return str(self.path)


def _locate_symbol(path: str) -> Any | None:
    """Import ``path`` and return the referenced attribute if it exists."""

    module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:  # pragma: no cover - defensive guard
        return None
    return getattr(module, attribute, None)


def _normalise_candidates(candidates: Iterable[str] | None) -> tuple[str, ...]:
    if candidates is None:
        return _GGUF_EXPORTER_CANDIDATES
    if isinstance(candidates, tuple):
        return candidates
    return tuple(candidates)


@lru_cache(maxsize=8)
def _load_gguf_exporter_cached(candidates: tuple[str, ...]) -> GGUFExporterInfo | None:
    for path in candidates:
        exporter = _locate_symbol(path)
        if exporter is None or not callable(exporter):
            continue
        return GGUFExporterInfo(qualified_name=path, factory=exporter)
    return None


def _load_gguf_exporter(
    candidates: Iterable[str] | None = None,
) -> GGUFExporterInfo | None:
    """Return metadata for the first available GGUF exporter implementation."""

    normalised = _normalise_candidates(candidates)
    return _load_gguf_exporter_cached(normalised)


def merge_lora_adapters(
    base_model_id: str,
    adapters_dir: Path,
    *,
    output_dir: Path,
) -> bool:
    """Merge LoRA adapters into the base model and persist FP16 weights."""

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency missing
        logger.warning("Unable to import transformers/peft for merge", exc_info=True)
        raise RuntimeError(
            "transformers and peft are required to merge adapters"
        ) from exc

    logger.info(
        "Merging adapters into base model",
        extra={"base_model": base_model_id, "adapters_dir": str(adapters_dir)},
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype="auto",
        device_map={"": "cpu"},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    merged = PeftModel.from_pretrained(base_model, str(adapters_dir)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged FP16 model saved", extra={"output_dir": str(output_dir)})
    return True


def export_gguf(
    source_dir: Path,
    *,
    gguf_dir: Path,
    quantization_method: str,
) -> bool:
    """Export a model directory to GGUF via Unsloth when available."""

    try:
        from unsloth import FastModel
    except Exception as exc:  # pragma: no cover - optional dependency missing
        raise RuntimeError("Unsloth is required for GGUF export") from exc

    logger.info(
        "Exporting GGUF",
        extra={
            "source_dir": str(source_dir),
            "gguf_dir": str(gguf_dir),
            "method": quantization_method,
        },
    )
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastModel.from_pretrained(str(source_dir))
    model.save_pretrained_gguf(
        str(gguf_dir), tokenizer=tokenizer, quantization_method=quantization_method
    )
    logger.info("GGUF export complete", extra={"gguf_dir": str(gguf_dir)})
    return True


def export_to_gguf(
    model_name: str,
    output_path: str,
    *,
    quantization_method: str | None = None,
) -> GGUFExportResult:
    """Load ``model_name`` with Transformers and export it in GGUF format."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("transformers is required for GGUF export") from exc

    exporter_info = _load_gguf_exporter()
    if exporter_info is None:
        raise RuntimeError(
            "No GGUF exporter available. Install a transformers release that "
            "includes GgufExporter or provide llama.cpp conversion tools."
        )

    logger.info(
        "Preparing GGUF export",
        extra={
            "model": model_name,
            "output": output_path,
            "exporter": exporter_info.qualified_name,
        },
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    try:  # pragma: no branch - defensive CPU migration
        model.to("cpu")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - optional step
        logger.debug("Unable to move model to CPU before export", exc_info=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    destination = Path(output_path)
    if destination.suffix.lower() != ".gguf":
        destination.mkdir(parents=True, exist_ok=True)
        model_stub = model_name.replace("/", "_").strip() or "model"
        destination = destination / f"{model_stub}.gguf"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        exporter = exporter_info.create(model=model, tokenizer=tokenizer)
    except TypeError as exc:  # pragma: no cover - unexpected init signature
        raise RuntimeError(
            "Unable to instantiate GGUF exporter; unexpected constructor signature"
        ) from exc

    def _invoke(method_name: str) -> bool:
        method = getattr(exporter, method_name, None)
        if method is None:
            return False

        kwargs: dict[str, str] = {}
        if quantization_method:
            kwargs["quantization_method"] = quantization_method

        try:
            method(str(destination), **kwargs)
        except TypeError as exc:
            if quantization_method and "quantization_method" in str(exc):
                method(str(destination))
            else:
                raise
        return True

    method_used: str | None = None
    for method_name in ("export", "export_model", "save_pretrained"):
        if _invoke(method_name):
            method_used = method_name
            break

    if method_used is None:  # pragma: no cover - unexpected interface
        raise RuntimeError(
            "Unsupported GGUF exporter interface; expected export/export_model/save_pretrained"
        )

    logger.info(
        "GGUF model written",
        extra={
            "path": str(destination),
            "export_method": method_used,
            "exporter": exporter_info.qualified_name,
        },
    )
    return GGUFExportResult(
        path=destination,
        exporter=exporter_info.qualified_name,
        method=method_used,
        quantization_method=quantization_method,
    )


def export_to_ollama(model_name: str, output_dir: str | Path) -> Path:
    """Export an Ollama model snapshot to ``output_dir`` using ``ollama export``."""

    if not model_name:
        raise ValueError("model_name must be provided for Ollama export")

    if shutil.which("ollama") is None:
        raise RuntimeError(
            "Ollama CLI is not available. Install Ollama and ensure the `ollama` "
            "binary is on PATH."
        )

    destination = Path(output_dir)
    if destination.suffix:
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        model_stub = model_name.replace("/", "_").strip() or "model"
        destination = destination / f"{model_stub}.bin"

    logger.info(
        "Exporting Ollama model",
        extra={"model": model_name, "output": str(destination)},
    )

    command = ["ollama", "export", model_name]
    try:
        with destination.open("wb") as export_handle:
            result = subprocess.run(
                command,
                check=True,
                stdout=export_handle,
                stderr=subprocess.PIPE,
            )
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Ollama CLI is not available. Install Ollama and ensure the `ollama` "
            "binary is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore")
        logger.error(
            "Ollama export failed",
            extra={"model": model_name, "output": str(destination), "stderr": stderr},
        )
        raise RuntimeError(f"ollama export failed: {stderr.strip() or exc}") from exc

    stderr_output = (result.stderr or b"").decode("utf-8", errors="ignore")
    if stderr_output.strip():
        logger.debug(
            "Ollama export stderr",
            extra={
                "model": model_name,
                "output": str(destination),
                "stderr": stderr_output,
            },
        )

    logger.info("Ollama model exported", extra={"path": str(destination)})
    return destination


def run_generation_smoke_test(model: Any, tokenizer: Any, prompt: str) -> str | None:
    """Generate a short sample from ``model`` for validation."""

    if not prompt:
        return None
    try:
        batch = tokenizer(prompt, return_tensors="pt")
        target_device = getattr(model, "device", None)
        if target_device is not None:
            batch = batch.to(target_device)
        output = model.generate(
            **batch,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    except Exception:  # pragma: no cover - defensive guard
        logger.warning("Generation smoke test failed", exc_info=True)
        return None
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()
