#!/usr/bin/env python3
"""GitHub Actions helper to inspect accelerator availability.

This module checks for CUDA-capable GPUs and records the detected accelerator
into the `GITHUB_OUTPUT` file while printing diagnostic information. It mirrors
the inline logic previously baked into the workflow but avoids YAML heredocs so
the workflow stays simple and maintainable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def emit(level: str, message: str) -> None:
    """Emit a GitHub Actions command message."""

    print(f"::{level}::{message}")


def ensure_github_output() -> Path:
    """Resolve the GitHub output path, raising a clear error if missing."""

    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        emit("error", "GITHUB_OUTPUT environment variable is not set")
        raise SystemExit(1)
    return Path(output_path)


def inspect_accelerator() -> None:
    accelerator = "gpu"
    cuda_available = False
    device_count = 0

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            subprocess.run(
                [nvidia_smi], check=True, stdout=sys.stdout, stderr=sys.stderr
            )
        except subprocess.CalledProcessError as exc:
            emit(
                "warning",
                f"nvidia-smi exited with {exc.returncode}; continuing with CPU fallback",
            )
            accelerator = "cpu"
    else:
        emit("warning", "nvidia-smi not found; continuing with CPU fallback")
        accelerator = "cpu"

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - defensive logging in CI
        emit("error", f"Failed to import torch: {exc}")
        raise SystemExit(1) from exc

    print(f"torch version {torch.__version__}")

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - CUDA probing edge case
        emit(
            "warning",
            f"torch.cuda.is_available() raised {exc}; assuming CPU fallback",
        )
        accelerator = "cpu"
    else:
        try:
            device_count = int(torch.cuda.device_count()) if cuda_available else 0
        except Exception as exc:  # pragma: no cover - CUDA probing edge case
            emit(
                "warning",
                f"torch.cuda.device_count() raised {exc}; switching to CPU fallback",
            )
            accelerator = "cpu"
            cuda_available = False

    print(f"cuda available {cuda_available}")
    print(f"cuda device count {device_count}")

    if not cuda_available or device_count <= 0:
        if accelerator != "cpu":
            emit(
                "warning", "CUDA device not detected; running fine-tune on CPU fallback"
            )
        accelerator = "cpu"

    output_path = ensure_github_output()
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"accelerator={accelerator}\n")
        handle.write(f"cuda_available={'true' if cuda_available else 'false'}\n")
        handle.write(f"cuda_device_count={device_count}\n")


def main() -> None:
    inspect_accelerator()


if __name__ == "__main__":
    main()
