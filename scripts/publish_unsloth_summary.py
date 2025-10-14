#!/usr/bin/env python3
"""Publish summary information for the Unsloth workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path


def emit(level: str, message: str) -> None:
    print(f"::{level}::{message}")


def load_metadata(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        emit("warning", f"Failed to parse {path.name}: {exc}")
        return []
    lines = json.dumps(payload, indent=2).splitlines()
    output: list[str] = [""]
    output.append("### run_metadata.json (first 200 lines)")
    output.append("```json")
    output.extend(lines[:200])
    if len(lines) > 200:
        output.append("... truncated ...")
    output.append("```")
    return output


def main() -> None:
    output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
    registry_path = Path(os.environ["REGISTRY_PATH"]).resolve()
    summary_path = Path(os.environ["SUMMARY_PATH"])
    hf_source = os.environ.get("HF_TOKEN_SOURCE", "none")

    lines = [
        "## Unsloth fine-tune summary",
        f"- Output directory: `{output_dir}`",
        f"- Registry path: `{registry_path}`",
        f"- Hugging Face token source: `{hf_source}`",
    ]

    lines.extend(load_metadata(output_dir / "run_metadata.json"))

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except KeyError as exc:  # pragma: no cover - environment contract
        emit("error", f"Missing required environment variable: {exc.args[0]}")
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - defensive
        emit("error", f"Unexpected failure while writing summary: {exc}")
        raise SystemExit(1) from exc
