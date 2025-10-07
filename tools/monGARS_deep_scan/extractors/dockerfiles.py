from __future__ import annotations

import io
from pathlib import Path
from typing import List

try:  # pragma: no cover - optional dependency
    from dockerfile_parse import DockerfileParser
except ModuleNotFoundError:  # pragma: no cover
    DockerfileParser = None  # type: ignore[assignment]

from ..utils.text_clean import normalise_whitespace


def _fallback_structure(text: str) -> List[dict]:
    structure: List[dict] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        instruction = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""
        structure.append(
            {
                "instruction": instruction,
                "value": value,
                "startline": idx - 1,
                "endline": idx - 1,
            }
        )
    return structure


from .types import ExtractionRecord


def extract(path: Path, text: str) -> List[ExtractionRecord]:
    if DockerfileParser is not None:
        parser = DockerfileParser(fileobj=io.StringIO(text))
        structure = parser.structure
    else:
        structure = _fallback_structure(text)
    records: List[ExtractionRecord] = []
    for entry in structure:
        instruction = entry.get("instruction")
        value = entry.get("value") or ""
        start_line = entry.get("startline", 0) + 1
        end_line = entry.get("endline", start_line) + 1
        if instruction == "RUN" and value:
            records.append(
                ExtractionRecord.for_agent(
                    instruction=f"Execute Docker RUN step '{value.splitlines()[0].strip()}'",
                    output={"instruction": instruction, "value": value},
                    source_file=str(path),
                    start_line=start_line,
                    end_line=end_line,
                    type_label="docker_run",
                )
            )
        elif instruction in {"CMD", "ENTRYPOINT"} and value:
            records.append(
                ExtractionRecord.for_agent(
                    instruction=f"Configure container {instruction.lower()}",
                    output={"instruction": instruction, "value": value},
                    source_file=str(path),
                    start_line=start_line,
                    end_line=end_line,
                    type_label="docker_entry",
                )
            )
        if value and len(value) >= 60:
            records.append(
                ExtractionRecord.for_embedding(
                    text=normalise_whitespace(value),
                    source_file=str(path),
                    start_line=start_line,
                    end_line=end_line,
                    type_label="docker_instruction",
                )
            )
    return records
