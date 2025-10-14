#!/usr/bin/env python3
"""Resolve pipeline overrides for the Unsloth workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def emit(level: str, message: str) -> None:
    """Emit a GitHub Actions logging command."""

    print(f"::{level}::{message}")


DEFAULTS: dict[str, Any] = {
    "max_seq_len": "2048",
    "vram_budget_mb": "8192",
    "activation_buffer_mb": "1024",
    "batch_size": "1",
    "grad_accum": "8",
    "learning_rate": "2e-4",
    "epochs": "1",
    "max_steps": "-1",
    "lora_rank": "32",
    "lora_alpha": "32",
    "lora_dropout": "0.05",
    "train_fraction": "",
    "eval_batch_size": "",
    "retention_days": "14",
    "skip_smoke_tests": False,
    "skip_metadata": False,
    "skip_merge": False,
}

BOOLEAN_KEYS = {"skip_smoke_tests", "skip_metadata", "skip_merge"}
TRUTHY_VALUES = {
    True,
    "true",
    "True",
    "TRUE",
    "1",
    1,
    "yes",
    "YES",
    "Yes",
    "on",
    "ON",
    "On",
    "t",
    "T",
}


def parse_raw_overrides(raw: str) -> dict[str, Any]:
    """Parse overrides from JSON or key=value pairs."""

    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        overrides: dict[str, Any] = {}
        tokens = [token.strip() for token in raw.replace("\n", ",").split(",")]
        for token in tokens:
            if not token:
                continue
            if "=" not in token:
                emit(
                    "error",
                    f"Invalid override entry '{token}'. Use key=value or JSON object.",
                )
                raise SystemExit(1)
            key, value = token.split("=", 1)
            overrides[key.strip()] = value.strip()
        return overrides
    else:
        if not isinstance(parsed, dict):
            emit("error", "pipeline_overrides must decode to a JSON object.")
            raise SystemExit(1)
        return parsed


def merge_overrides(raw_overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge user overrides with defaults and normalise values."""

    merged = DEFAULTS.copy()
    for key, value in raw_overrides.items():
        if key not in merged:
            emit(
                "error",
                f"Unknown override key '{key}'. Allowed keys: {sorted(merged.keys())}",
            )
            raise SystemExit(1)
        if key in BOOLEAN_KEYS:
            merged[key] = value in TRUTHY_VALUES
        else:
            merged[key] = "" if value is None else str(value)
    return merged


def write_outputs(merged: dict[str, Any]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        emit("error", "GITHUB_OUTPUT environment variable is not set")
        raise SystemExit(1)

    path = Path(output_path)
    with path.open("a", encoding="utf-8") as handle:
        for key, value in merged.items():
            if key in BOOLEAN_KEYS:
                rendered = "true" if bool(value) else "false"
            else:
                rendered = value
            handle.write(f"{key}={rendered}\n")


def main() -> None:
    raw = os.environ.get("RAW_OVERRIDES", "").strip()
    overrides = parse_raw_overrides(raw)
    merged = merge_overrides(overrides)

    print("Resolved pipeline overrides:")
    for key in sorted(merged):
        print(f"- {key}: {merged[key]}")

    write_outputs(merged)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        emit("error", f"Unexpected failure while resolving overrides: {exc}")
        raise SystemExit(1) from exc
